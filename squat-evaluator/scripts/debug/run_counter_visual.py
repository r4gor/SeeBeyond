"""
Phase 3 visual debug: live RGB frame with 2D skeleton overlay,
knee-angle readout, state-machine state, and rep counter.

Run from project root:
    python -m scripts.run_counter_visual
Press Q in the video window to quit.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipeline import open_oak_skeleton_stream, COCO_SKELETON_EDGES
from src.rep_counter import (
    RepCounter, RepData,
    L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
    KP_CONF_MIN,
)

# Joints we care most about for squats — drawn larger / in a different color.
SQUAT_JOINTS = {L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE}

STATE_COLOR = {
    "STANDING":   (180, 180, 180),
    "DESCENDING": (  0, 200, 255),  # amber
    "BOTTOM":     (  0, 100, 255),  # red-orange
    "ASCENDING":  (  0, 255, 180),  # green-cyan
}


def draw_skeleton(frame, skel):
    """Draw COCO edges + keypoints. Faded if confidence < KP_CONF_MIN."""
    kps = skel.keypoints

    # edges first (so dots sit on top)
    for a, b in COCO_SKELETON_EDGES:
        ka, kb = kps[a], kps[b]
        if ka.confidence < KP_CONF_MIN or kb.confidence < KP_CONF_MIN:
            continue
        if ka.z_cm <= 0 or kb.z_cm <= 0:
            continue
        pa = (int(ka.x_px), int(ka.y_px))
        pb = (int(kb.x_px), int(kb.y_px))
        is_squat_edge = a in SQUAT_JOINTS and b in SQUAT_JOINTS
        color = (0, 255, 0) if is_squat_edge else (200, 200, 200)
        thickness = 3 if is_squat_edge else 1
        cv2.line(frame, pa, pb, color, thickness)

    # keypoints
    for i, kp in enumerate(kps):
        valid = kp.confidence >= KP_CONF_MIN and kp.z_cm > 0
        color = (0, 255, 255) if i in SQUAT_JOINTS else (255, 200, 0)
        if not valid:
            color = (80, 80, 80)  # grey for occluded
        radius = 6 if i in SQUAT_JOINTS else 3
        cv2.circle(frame, (int(kp.x_px), int(kp.y_px)), radius, color, -1)

        # depth label on the squat joints
        if i in SQUAT_JOINTS and valid:
            cv2.putText(
                frame, f"{kp.z_cm:.0f}",
                (int(kp.x_px) + 8, int(kp.y_px) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )


def draw_hud(frame, counter, skel, fps):
    h, w = frame.shape[:2]
    state = counter.current_state
    angle = counter.latest_angle

    # top-left HUD
    bar_color = STATE_COLOR.get(state, (255, 255, 255))
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (w, 70), bar_color, 2)

    angle_str = f"{angle:5.1f}°" if angle is not None else "  --  "
    cv2.putText(frame, f"REPS {counter.rep_count}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"STATE {state}",
                (160, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bar_color, 2)
    cv2.putText(frame, f"KNEE {angle_str}",
                (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS  {fps:4.1f}",
                (200, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(frame, f"DET {skel.detection_confidence:.2f}",
                (320, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)


def main():
    last_rep_text = ""
    last_rep_until = 0.0

    def on_rep_started():
        print("  [rep] descent started")

    def on_rep(rep: RepData):
        nonlocal last_rep_text, last_rep_until
        msg = f"REP {rep.rep_number}: min={rep.min_knee_angle:.1f}°  frames={len(rep.trajectory)}"
        print(msg)
        last_rep_text = msg
        last_rep_until = time.monotonic() + 2.0  # flash on screen for 2 s

    counter = RepCounter(on_rep_complete=on_rep, on_rep_started=on_rep_started)

    print("[visual] opening OAK 4 D pipeline... (press Q in window to quit)")
    cv2.namedWindow("squat eval", cv2.WINDOW_NORMAL)

    frame_count = 0
    fps = 0.0
    fps_t0 = time.monotonic()

    # NOTE: with_frames=True returns (frame, skeleton) per yield
    with open_oak_skeleton_stream(fps=24, with_frames=True) as stream:
        for frame, skel in stream:
            counter.update(skel)
            draw_skeleton(frame, skel)
            draw_hud(frame, counter, skel, fps)

            # flash the latest rep on screen for 2 s
            if time.monotonic() < last_rep_until:
                cv2.putText(frame, last_rep_text,
                            (12, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("squat eval", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):  # Q or Esc
                break

            # rolling FPS
            frame_count += 1
            now = time.monotonic()
            if now - fps_t0 >= 1.0:
                fps = frame_count / (now - fps_t0)
                frame_count = 0
                fps_t0 = now

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()