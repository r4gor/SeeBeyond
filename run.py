"""
SeeBeyond — unified squat evaluator entry point.

Wires together:
  OAK 4 D skeleton stream  (squat-evaluator/src/pipeline.py)
  → RepCounter             (squat-evaluator/src/rep_counter.py)
  → feature extraction     (squat-evaluator/src/features.py)
  → form classifier        (squat-evaluator/src/classifier.py)
  → Coach:
      · trigger_rep + send_display_update → MQTT → Core2  [sync, instant]
      · llm_assistant_response → ElevenLabs TTS → MQTT → Core2  [async thread]

Usage:
    python run.py                             # full pipeline; reads .env for keys/broker
    python run.py --broker 192.168.1.100      # explicit MQTT broker IP
    python run.py --port 1883                 # MQTT port (default 1883)
    python run.py --fps 24                    # camera FPS (default 24)
    python run.py --no-sound                  # skip TTS / MQTT  (offline / no broker)
    python run.py --no-overlay                # skip OpenCV debug window

Keys (OpenCV debug window):
    Q / ESC   quit
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path wiring — no package installs needed; works from the repo root.
# ---------------------------------------------------------------------------

_REPO    = Path(__file__).resolve().parent
_EVA     = _REPO / "squat-evaluator"
_SOUND   = _REPO / "coach" / "backend" / "sound"

sys.path.insert(0, str(_EVA))
sys.path.insert(0, str(_SOUND))

# ---------------------------------------------------------------------------
# squat-evaluator imports
# ---------------------------------------------------------------------------

import numpy as np
import cv2

from src.pipeline      import open_oak_skeleton_stream, COCO_SKELETON_EDGES
from src.rep_counter   import RepCounter, RepData
from src.features      import extract_features
from src.classifier    import classify_and_explain

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# COCO-17 joint indices used for display / depth stats
L_SHO, R_SHO = 5, 6
L_HIP, R_HIP = 11, 12
L_KNE, R_KNE = 13, 14
L_ANK, R_ANK = 15, 16
SQUAT_JOINTS  = {L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK}

CONF_THR   = 0.25
WINDOW     = "squat-debug"
WINDOW_W   = 480          # camera feed width after downscale
PANEL_W    = 150          # right-side depth-stats panel width
WINDOW_POS = (12, 12)

FEEDBACK_FLASH_S = 5.0   # seconds to keep the verdict strip on the HUD after a rep

# HUD colours (BGR)
GREEN  = (0, 220, 120)
YELLOW = (0, 255, 255)
ORANGE = (0, 110, 255)
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
RED    = (0,  60, 220)
CYAN   = (255, 200,  0)

STATE_COLOR = {
    "STANDING":   (180, 180, 180),
    "DESCENDING": (  0, 200, 255),
    "BOTTOM":     (  0, 100, 255),
    "ASCENDING":  (  0, 255, 180),
}

# ---------------------------------------------------------------------------
# Sound / MQTT — lazy-loaded so --no-sound skips all heavy deps
# ---------------------------------------------------------------------------

_voice = None  # module reference, set by _load_voice()


def _load_voice(broker: str, port: int) -> None:
    """Import voice.py and point it at the correct broker via env vars."""
    global _voice
    os.environ["MQTT_BROKER"] = broker
    os.environ["MQTT_PORT"]   = str(port)
    import voice as _v
    _voice = _v


_audio_lock = threading.Lock()


def _speak_async(verdict: str, feedback: str, knee_angle: float) -> None:
    """Fire-and-forget: Mistral → ElevenLabs TTS → WAV → MQTT PCM → Core2 speaker."""
    def _worker() -> None:
        with _audio_lock:
            try:
                prompt      = (
                    f"verdict={verdict}, feedback={feedback}, "
                    f"knee_angle={knee_angle:.0f}"
                )
                instruction = _voice.llm_assistant_response(prompt)
                print(f"[sound] spoken cue: {instruction!r}")
                wav = _voice.transcribe_message(instruction)
                _voice.play_sound(wav)
            except Exception as exc:
                print(f"[sound] ERROR: {exc}")

    threading.Thread(target=_worker, daemon=True).start()


# ---------------------------------------------------------------------------
# Depth helpers
# ---------------------------------------------------------------------------

def _z_to_color(z_cm: float) -> tuple:
    """BGR color gradient: orange (close, ≤150 cm) → green (200 cm) → cyan (far, ≥280 cm)."""
    if z_cm <= 0 or np.isnan(z_cm):
        return (80, 80, 80)  # gray for invalid
    t = float(np.clip((z_cm - 150) / 130, 0.0, 1.0))  # 0=close, 1=far
    b = int(255 * t)
    g = int(140 + 80 * t)
    r = int(255 * (1 - t))
    return (b, g, r)


def _depth_stats(kp3d: np.ndarray, conf: np.ndarray):
    """Return (hip_z, knee_z, parallel_delta_cm, trunk_deg) from 3D keypoints.

    parallel_delta = hip_y_cm − knee_y_cm  (+Y down, so positive = hip below knee = GOOD).
    trunk_deg = angle of shoulder→hip vector from world-up.
    All may be NaN when joints are occluded.
    """
    valid = lambda idx: conf[idx] >= CONF_THR and kp3d[idx, 2] > 0

    hip_zs = [kp3d[i, 2] for i in (L_HIP, R_HIP) if valid(i)]
    kne_zs = [kp3d[i, 2] for i in (L_KNE, R_KNE) if valid(i)]
    hip_ys = [kp3d[i, 1] for i in (L_HIP, R_HIP) if valid(i)]
    kne_ys = [kp3d[i, 1] for i in (L_KNE, R_KNE) if valid(i)]

    hip_z = float(np.mean(hip_zs)) if hip_zs else float("nan")
    kne_z = float(np.mean(kne_zs)) if kne_zs else float("nan")

    hip_y = float(np.mean(hip_ys)) if hip_ys else float("nan")
    kne_y = float(np.mean(kne_ys)) if kne_ys else float("nan")
    par_delta = hip_y - kne_y if not (np.isnan(hip_y) or np.isnan(kne_y)) else float("nan")

    # Trunk angle: shoulder-midpoint → hip-midpoint vector vs world-up (0, -1, 0)
    trunk_deg = float("nan")
    sho_pts = [(kp3d[i, 0], kp3d[i, 1], kp3d[i, 2]) for i in (L_SHO, R_SHO) if valid(i)]
    hip_pts = [(kp3d[i, 0], kp3d[i, 1], kp3d[i, 2]) for i in (L_HIP, R_HIP) if valid(i)]
    if sho_pts and hip_pts:
        sho_m = np.mean(sho_pts, axis=0)
        hip_m = np.mean(hip_pts, axis=0)
        trunk_vec = sho_m - hip_m
        n = np.linalg.norm(trunk_vec)
        if n > 1e-6:
            world_up = np.array([0.0, -1.0, 0.0])
            trunk_deg = float(np.degrees(
                np.arccos(np.clip(np.dot(trunk_vec / n, world_up), -1.0, 1.0))
            ))

    return hip_z, kne_z, par_delta, trunk_deg


# ---------------------------------------------------------------------------
# OpenCV drawing helpers
# ---------------------------------------------------------------------------

def _put_text(
    canvas,
    text: str,
    org: tuple,
    scale: float = 0.6,
    color: tuple = WHITE,
    thick: int = 1,
) -> None:
    """White text with a dark outline — readable on any background."""
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, BLACK, thick + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def _draw_skeleton(canvas, skel) -> None:
    """Draw COCO-17 bones (Z-depth colored) + joints (squat joints highlighted)."""
    kp2d  = np.array([[k.x_px, k.y_px]         for k in skel.keypoints], np.float32)
    kp3d  = np.array([[k.x_cm, k.y_cm, k.z_cm]  for k in skel.keypoints], np.float32)
    conf  = np.array([k.confidence               for k in skel.keypoints], np.float32)
    depth_ok = (kp3d[:, 2] > 0) & (conf >= CONF_THR)

    # Bones — color each segment by the average Z of its two endpoints
    for i, j in COCO_SKELETON_EDGES:
        if conf[i] < CONF_THR or conf[j] < CONF_THR:
            continue
        is_squat  = (i in SQUAT_JOINTS) and (j in SQUAT_JOINTS)
        # Z-depth gradient coloring (works for all bones, more saturated on squat bones)
        avg_z = float((kp3d[i, 2] + kp3d[j, 2]) / 2)
        color = _z_to_color(avg_z) if depth_ok[i] and depth_ok[j] else (100, 100, 100)
        thickness = 3 if is_squat else 1
        cv2.line(canvas,
                 (int(kp2d[i, 0]), int(kp2d[i, 1])),
                 (int(kp2d[j, 0]), int(kp2d[j, 1])),
                 color, thickness, cv2.LINE_AA)

    # Joints
    for k in range(17):
        if conf[k] < CONF_THR:
            continue
        center = (int(kp2d[k, 0]), int(kp2d[k, 1]))
        is_squat = k in SQUAT_JOINTS
        color  = YELLOW if depth_ok[k] else ORANGE
        radius = 6 if is_squat else 3
        cv2.circle(canvas, center, radius, color, -1, cv2.LINE_AA)

        # Z-depth label on squat joints that have valid depth
        if is_squat and depth_ok[k]:
            _put_text(
                canvas,
                f"{kp3d[k, 2]:.0f}",
                (int(kp2d[k, 0]) + 8, int(kp2d[k, 1]) - 6),
                scale=0.55,  # larger so it survives 0.75× downscale to 480px
                color=color,
            )


def _draw_hud(
    canvas,
    counter: RepCounter,
    skel,
    fps: float,
    verdict: str,
    feedback: str,
    show_feedback: bool,
) -> None:
    h, w  = canvas.shape[:2]
    state = counter.current_state
    angle = counter.latest_angle
    bar   = STATE_COLOR.get(state, WHITE)

    # Top status bar
    cv2.rectangle(canvas, (0, 0), (w, 82), (0, 0, 0), -1)
    cv2.rectangle(canvas, (0, 0), (w, 82), bar, 2)

    angle_str = f"{angle:5.1f}°" if angle is not None else "  --  "
    _put_text(canvas, f"REPS {counter.rep_count}",             (12, 30),  0.9, WHITE, 2)
    _put_text(canvas, f"STATE {state}",                        (160, 30), 0.7, bar,   2)
    _put_text(canvas, f"KNEE {angle_str}",                     (12, 62),  0.7, WHITE, 2)
    _put_text(canvas, f"FPS {fps:4.1f}",                       (230, 62), 0.55, (160, 160, 160))
    _put_text(canvas, f"DET {skel.detection_confidence:.2f}",  (340, 62), 0.55, (160, 160, 160))

    # Bottom feedback strip (flashes for FEEDBACK_FLASH_S after each rep)
    if show_feedback and verdict:
        strip_color = (0, 130, 0) if verdict == "good" else (20, 20, 160)
        cv2.rectangle(canvas, (0, h - 44), (w, h), strip_color, -1)
        cv2.rectangle(canvas, (0, h - 44), (w, h), bar, 1)
        label = f"{verdict.upper()}: {feedback}"
        _put_text(canvas, label, (10, h - 12), 0.65, WHITE, 2)


def _draw_depth_panel(panel: np.ndarray, angle, hip_z, kne_z, par_delta, trunk_deg) -> None:
    """Draw the right-side depth-stats panel in place.

    Layout is adaptive: metrics fill the top portion, gauge takes remaining height.
    """
    ph, pw = panel.shape[:2]
    panel[:] = (15, 15, 25)  # very dark blue-black

    DIM  = (140, 140, 140)
    SEP  = (55, 55, 75)

    def _sep(y):
        cv2.line(panel, (4, y), (pw - 4, y), SEP, 1)

    # --- Header ---
    _put_text(panel, "DEPTH", (6, 16), 0.5, CYAN, 1)
    _sep(22)

    # --- Z distances (one line each: label + value combined) ---
    hip_z_str = f"HIP  {hip_z:.0f}cm" if not np.isnan(hip_z) else "HIP  ---"
    kne_z_str = f"KNE  {kne_z:.0f}cm" if not np.isnan(kne_z) else "KNE  ---"
    _put_text(panel, hip_z_str, (6, 38), 0.44, WHITE)
    _put_text(panel, kne_z_str, (6, 54), 0.44, WHITE)
    _sep(60)

    # --- Parallel delta (uses real 3D depth: hip_y - knee_y in cm) ---
    if not np.isnan(par_delta):
        sign      = "+" if par_delta >= 0 else ""
        par_color = (0, 200, 80) if par_delta >= 0 else (60, 60, 220)  # green/red
        par_str   = f"PAR {sign}{par_delta:.1f}cm"
        _put_text(panel, par_str, (6, 76), 0.44, par_color)
    else:
        _put_text(panel, "PAR ---", (6, 76), 0.44, DIM)
    _sep(82)

    # --- Trunk angle ---
    if not np.isnan(trunk_deg):
        trunk_color = (0, 220, 255) if trunk_deg < 30 else (0, 140, 255)
        _put_text(panel, f"TRK {trunk_deg:.0f}\xb0", (6, 98), 0.44, trunk_color)
    else:
        _put_text(panel, "TRK ---", (6, 98), 0.44, DIM)
    _sep(104)

    # --- Depth gauge fills the remaining vertical space ---
    _put_text(panel, "GAUGE", (6, 116), 0.4, DIM)
    gauge_top = 120
    gauge_bot = ph - 6
    gauge_h   = max(gauge_bot - gauge_top, 4)
    gx        = pw // 2 - 10
    gw        = 20

    cv2.rectangle(panel, (gx, gauge_top), (gx + gw, gauge_bot), (70, 70, 90), 1)

    if angle is not None and not np.isnan(angle):
        # 160° (standing) → 0 %, 90° (parallel) → 100 %
        progress  = float(np.clip((160.0 - angle) / 70.0, 0.0, 1.0))
        fill_px   = int(gauge_h * progress)
        if fill_px > 0:
            # BGR gradient: red (not there yet) → yellow (halfway) → green (parallel)
            t  = progress
            fg = (0, int(255 * min(2 * t, 1.0)), int(255 * (1 - t)))
            cv2.rectangle(panel,
                          (gx + 1, gauge_bot - fill_px),
                          (gx + gw - 1, gauge_bot - 1),
                          fg, -1)

        # Parallel threshold marker: dashed line at the top of gauge range
        par_y = gauge_top + 1
        cv2.line(panel, (gx - 4, par_y), (gx + gw + 4, par_y), (0, 180, 255), 2)

        # "PAR" badge when hip is genuinely below knee in 3D (depth-driven check)
        if not np.isnan(par_delta) and par_delta > 0:
            _put_text(panel, "PAR", (gx - 2, gauge_top - 3), 0.35, (0, 220, 100))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(_REPO / "coach" / "backend" / ".env")

    ap = argparse.ArgumentParser(description="SeeBeyond squat evaluator")
    ap.add_argument("--broker",      default=os.environ.get("MQTT_BROKER", "192.168.1.100"),
                    help="MQTT broker IP (default: env MQTT_BROKER or 192.168.1.100)")
    ap.add_argument("--port",        type=int,
                    default=int(os.environ.get("MQTT_PORT", 1883)),
                    help="MQTT port (default 1883)")
    ap.add_argument("--fps",         type=int, default=24,
                    help="Camera / pipeline FPS (default 24)")
    ap.add_argument("--no-sound",    action="store_true",
                    help="Disable TTS + MQTT audio (offline / no broker)")
    ap.add_argument("--no-overlay",  action="store_true",
                    help="Disable OpenCV debug window")
    args = ap.parse_args()

    # ---------- load sound module ----------
    if not args.no_sound:
        try:
            _load_voice(args.broker, args.port)
            print(f"[run] sound ON  — broker {args.broker}:{args.port}")
        except Exception as exc:
            print(f"[run] WARNING: sound disabled — {exc}")

    # ---------- shared HUD state (written in coach thread, read in draw loop) ----------
    _last: dict = {"verdict": "", "feedback": "", "t": 0.0}

    # ---------- coach worker thread ----------
    _rep_queue: queue.Queue = queue.Queue()

    def _coach_worker() -> None:
        while True:
            rep = _rep_queue.get()
            if rep is None:
                break
            try:
                features        = extract_features(rep.trajectory, rep.bottom_frame_idx)
                verdict, fb_msg = classify_and_explain(features)
                n               = rep.rep_number
                min_angle       = rep.min_knee_angle

                print(
                    f"\n[REP {n:>3}]  verdict={verdict:<14s}  "
                    f"angle={min_angle:5.1f}°  →  {fb_msg}"
                )

                _last["verdict"]  = verdict
                _last["feedback"] = fb_msg
                _last["t"]        = time.monotonic()

                if _voice is not None:
                    try:
                        _voice.trigger_rep(good=(verdict == "good"))
                        # rep is complete — person is back to STANDING
                        _voice.send_display_update(n, verdict, fb_msg, int(min_angle), "STANDING")
                    except Exception as exc:
                        print(f"[run] MQTT error: {exc}")

                if _voice is not None:
                    _speak_async(verdict, fb_msg, min_angle)
            except Exception as exc:
                print(f"[coach] ERROR: {exc}")
            finally:
                _rep_queue.task_done()

    _coach_thread = threading.Thread(target=_coach_worker, daemon=True)
    _coach_thread.start()

    def _on_rep(rep: RepData) -> None:
        _rep_queue.put(rep)

    # ---------- set up ----------
    counter    = RepCounter(on_rep_complete=_on_rep)
    prev_state = "STANDING"

    if not args.no_overlay:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, WINDOW_W + PANEL_W, int(WINDOW_W * 9 / 16))
        cv2.moveWindow(WINDOW, *WINDOW_POS)

    frame_count = 0
    fps         = 0.0
    fps_t0      = time.monotonic()

    # ---------- main loop ----------
    print("[run] opening OAK 4 D pipeline …")
    with open_oak_skeleton_stream(fps=args.fps, with_frames=True) as stream:
        print("[run] tracking — Q or ESC in the debug window to quit.\n")
        for frame, skel in stream:
            counter.update(skel)

            # Push state-transition updates to Core2 in real time
            curr_state = counter.current_state
            if curr_state != prev_state and _voice is not None:
                angle = counter.latest_angle or 0
                try:
                    _voice.send_display_update(
                        counter.rep_count,
                        _last.get("verdict", "---"),
                        _last.get("feedback", "..."),
                        int(angle),
                        curr_state,
                    )
                except Exception as exc:
                    print(f"[run] MQTT state update error: {exc}")
                prev_state = curr_state

            # rolling FPS
            frame_count += 1
            now = time.monotonic()
            if now - fps_t0 >= 1.0:
                fps         = frame_count / (now - fps_t0)
                frame_count = 0
                fps_t0      = now

            if args.no_overlay:
                continue

            # Build 3D arrays for drawing
            kp3d = np.array(
                [[k.x_cm, k.y_cm, k.z_cm] for k in skel.keypoints], np.float32
            )
            conf = np.array([k.confidence for k in skel.keypoints], np.float32)

            # Camera canvas with skeleton
            canvas = frame.copy()
            _draw_skeleton(canvas, skel)
            show_fb = (now - _last["t"]) < FEEDBACK_FLASH_S
            _draw_hud(
                canvas, counter, skel, fps,
                _last["verdict"], _last["feedback"], show_fb,
            )

            # Downscale camera feed
            h, w    = canvas.shape[:2]
            cam_disp = cv2.resize(canvas, (WINDOW_W, int(h * WINDOW_W / w)),
                                  interpolation=cv2.INTER_AREA)

            # Depth-stats panel
            hip_z, kne_z, par_delta, trunk_deg = _depth_stats(kp3d, conf)
            panel = np.zeros((cam_disp.shape[0], PANEL_W, 3), dtype=np.uint8)
            _draw_depth_panel(panel, counter.latest_angle, hip_z, kne_z, par_delta, trunk_deg)

            # Combine camera + panel side by side
            combined = np.hstack([cam_disp, panel])
            cv2.imshow(WINDOW, combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

    if not args.no_overlay:
        cv2.destroyAllWindows()
    _rep_queue.put(None)
    print("\n[run] bye")


if __name__ == "__main__":
    main()
