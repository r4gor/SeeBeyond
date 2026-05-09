"""
SeeBeyond — unified squat evaluator entry point.

Wires together:
  OAK 4 D skeleton stream  (squat-evaluator/src/pipeline.py)
  → RepCounter             (squat-evaluator/src/rep_counter.py)
  → feature extraction     (squat-evaluator/src/features.py)
  → form classifier        (squat-evaluator/src/classifier.py)
  → Coach:
      · trigger_rep + display  → MQTT → Core2 beep + rep count  [sync, instant]
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
from src.classifier    import classify_and_explain, STATIC_FEEDBACK_PHRASES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# COCO-17 squat-relevant joint indices
L_HIP, R_HIP = 11, 12
L_KNE, R_KNE = 13, 14
L_ANK, R_ANK = 15, 16
SQUAT_JOINTS  = {L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK}

CONF_THR   = 0.25
WINDOW     = "squat-debug"
WINDOW_W   = 480
WINDOW_POS = (12, 12)

# HUD colours (BGR)
GREEN  = (0, 220, 120)
YELLOW = (0, 255, 255)
ORANGE = (0, 110, 255)
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
RED    = (0,  60, 220)

STATE_COLOR = {
    "STANDING":   (180, 180, 180),
    "DESCENDING": (  0, 200, 255),
    "BOTTOM":     (  0, 100, 255),
    "ASCENDING":  (  0, 255, 180),
}

FEEDBACK_FLASH_S = 5.0   # seconds to keep the verdict on the HUD after a rep

# ---------------------------------------------------------------------------
# Sound / MQTT — lazy-loaded so --no-sound skips all heavy deps
# ---------------------------------------------------------------------------

_voice = None  # module reference, set by _load_voice()


def _load_voice(broker: str, port: int) -> None:
    """Import voice.py and point it at the correct broker via env vars."""
    global _voice
    # voice.py reads MQTT_BROKER / MQTT_PORT from the environment.
    os.environ["MQTT_BROKER"] = broker
    os.environ["MQTT_PORT"]   = str(port)
    import voice as _v
    _voice = _v


def _speak_async(verdict: str, feedback: str, knee_angle: float) -> None:
    """
    Fire-and-forget: sends coaching text to Core2 via MQTT.
    Core2 calls ElevenLabs directly and plays the result.
    """
    def _worker() -> None:
        try:
            cue = _voice.coaching_cue(feedback)
            print(f"[sound] cue: {cue!r}  (knee {knee_angle:.0f}deg)")
            _voice.send_tts_text(cue)
        except Exception as exc:
            print(f"[sound] ERROR: {exc}")

    threading.Thread(target=_worker, daemon=True).start()


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
    """Draw COCO-17 bones + joints.  Squat joints are highlighted."""
    kp2d  = np.array([[k.x_px, k.y_px]        for k in skel.keypoints], np.float32)
    kp3d  = np.array([[k.x_cm, k.y_cm, k.z_cm] for k in skel.keypoints], np.float32)
    conf  = np.array([k.confidence              for k in skel.keypoints], np.float32)
    depth_ok = (kp3d[:, 2] > 0) & (conf >= CONF_THR)

    # Bones
    for i, j in COCO_SKELETON_EDGES:
        if conf[i] < CONF_THR or conf[j] < CONF_THR:
            continue
        is_squat  = (i in SQUAT_JOINTS) and (j in SQUAT_JOINTS)
        color     = GREEN if is_squat else (160, 160, 160)
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
        color  = YELLOW if depth_ok[k] else ORANGE
        radius = 6 if k in SQUAT_JOINTS else 3
        cv2.circle(canvas, center, radius, color, -1, cv2.LINE_AA)


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

    angle_str = f"{angle:5.1f}deg" if angle is not None else "  --  "
    _put_text(canvas, f"REPS {counter.rep_count}",              (12, 30),  0.9, WHITE, 2)
    _put_text(canvas, f"STATE {state}",                         (160, 30), 0.7, bar,   2)
    _put_text(canvas, f"KNEE {angle_str}",                      (12, 62),  0.7, WHITE, 2)
    _put_text(canvas, f"FPS {fps:4.1f}",                        (230, 62), 0.55, (160, 160, 160))
    _put_text(canvas, f"DET {skel.detection_confidence:.2f}",   (340, 62), 0.55, (160, 160, 160))

    # Bottom feedback strip (flashes for FEEDBACK_FLASH_S after each rep)
    if show_feedback and verdict:
        strip_color = (0, 180, 0) if verdict == "good" else (20, 20, 180)
        cv2.rectangle(canvas, (0, h - 36), (w, h), strip_color, -1)
        label = f"{verdict.upper()}: {feedback}"
        _put_text(canvas, label, (10, h - 10), 0.65, WHITE, 2)


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
            _voice.display(0)
            _warmup_phrases = list(STATIC_FEEDBACK_PHRASES) + [
                f"Hip stopped {n} cm above parallel"            for n in range(1, 26)
            ] + [
                f"Trunk leaned {n}° forward"               for n in range(45, 81)
            ] + [
                f"Heels lifted ~{n} cm — drive through midfoot" for n in range(5, 21)
            ] + [
                f"L/R knees differ by {n}° — even the load" for n in range(15, 41)
            ]
            _voice.warmup_tts_cache(_warmup_phrases)
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
                    f"angle={min_angle:5.1f}\u00b0  \u2192  {fb_msg}"
                )

                _last["verdict"]  = verdict
                _last["feedback"] = fb_msg
                _last["t"]        = time.monotonic()

                if _voice is not None:
                    try:
                        _voice.trigger_rep(good=(verdict == "good"))
                        _voice.display(n)
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
    counter = RepCounter(on_rep_complete=_on_rep)

    if not args.no_overlay:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, WINDOW_W, int(WINDOW_W * 9 / 16))
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

            # rolling FPS
            frame_count += 1
            now = time.monotonic()
            if now - fps_t0 >= 1.0:
                fps         = frame_count / (now - fps_t0)
                frame_count = 0
                fps_t0      = now

            if args.no_overlay:
                continue

            # Draw
            canvas = frame.copy()
            _draw_skeleton(canvas, skel)
            show_fb = (now - _last["t"]) < FEEDBACK_FLASH_S
            _draw_hud(
                canvas, counter, skel, fps,
                _last["verdict"], _last["feedback"], show_fb,
            )

            # Downscale to corner window
            h, w    = canvas.shape[:2]
            display = cv2.resize(canvas, (WINDOW_W, int(h * WINDOW_W / w)),
                                 interpolation=cv2.INTER_AREA)
            cv2.imshow(WINDOW, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

    if not args.no_overlay:
        cv2.destroyAllWindows()
    print("\n[run] bye")


if __name__ == "__main__":
    main()
