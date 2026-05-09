"""
Phase 5: tiny on-stage debug overlay.

For me, on stage, to confirm the vision pipeline is tracking. The
audience-facing UI lives on the Core2 — this is debug only.

Window contents:
  * Camera passthrough with COCO skeleton drawn over it
  * Live L/R knee angle in the top-left corner (3D, in degrees)
  * Joint color shows depth validity:
      yellow = keypoint has valid depth
      orange = keypoint detected in 2D but no usable depth
    On stage this is enough to spot framing problems instantly
    (subject too close, off the stereo baseline, etc.).

Explicitly NOT here: rep counter, verdict, feedback — those go to
the Core2 over serial, this script doesn't touch any of it.

Run from project root:
    python scripts/run_debug_overlay.py

Keys:
    Q or ESC   quit
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.pipeline import COCO_SKELETON_EDGES, open_oak_skeleton_stream

# COCO indices we touch directly (everything else is just edges).
L_HIP, R_HIP = 11, 12
L_KNE, R_KNE = 13, 14
L_ANK, R_ANK = 15, 16

WINDOW = "squat-debug"
WINDOW_W = 480              # display width; downscale from passthrough
WINDOW_POS = (12, 12)       # top-left of the laptop screen; drag to move
CONF_THR = 0.25             # joints below this conf are not drawn
FPS = 24

# BGR colors
GREEN  = (0, 220, 120)
YELLOW = (0, 255, 255)
ORANGE = (0, 110, 255)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)


# --------------------------------------------------------------------------- 

def knee_angle_3d(hip, knee, ankle) -> float:
    """Hip-knee-ankle angle in degrees from 3D coords. NaN-safe."""
    if any(np.isnan(v).any() for v in (hip, knee, ankle)):
        return float("nan")
    v1, v2 = hip - knee, ankle - knee
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    cos = float(np.dot(v1, v2)) / (n1 * n2)
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos)))))


def draw_skeleton(canvas, kp2d, conf, depth_valid):
    """Draw COCO bones + joints. Joint color encodes depth validity."""
    for i, j in COCO_SKELETON_EDGES:
        if conf[i] < CONF_THR or conf[j] < CONF_THR:
            continue
        p1 = (int(kp2d[i, 0]), int(kp2d[i, 1]))
        p2 = (int(kp2d[j, 0]), int(kp2d[j, 1]))
        cv2.line(canvas, p1, p2, GREEN, 2, cv2.LINE_AA)

    for k in range(17):
        if conf[k] < CONF_THR:
            continue
        center = (int(kp2d[k, 0]), int(kp2d[k, 1]))
        color = YELLOW if depth_valid[k] else ORANGE
        cv2.circle(canvas, center, 4, color, -1, cv2.LINE_AA)


def put_text(canvas, text, org, scale=0.6, color=WHITE, thick=1):
    """White text with a black outline for readability on any background."""
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                BLACK, thick + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, cv2.LINE_AA)


def main():
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WINDOW, *WINDOW_POS)

    print("[debug] opening OAK 4 D pipeline...")
    first = True
    with open_oak_skeleton_stream(fps=FPS, with_frames=True) as stream:
        print("[debug] running. Q or ESC to exit.")
        for frame, skel in stream:
            if first:
                print(f"[debug] tracking ({skel.frame_w}x{skel.frame_h})")
                first = False

            # gather arrays once per frame
            kp2d = np.array([[k.x_px, k.y_px] for k in skel.keypoints],
                            dtype=np.float32)
            kp3d = np.array([[k.x_cm, k.y_cm, k.z_cm] for k in skel.keypoints],
                            dtype=np.float32)
            conf = np.array([k.confidence for k in skel.keypoints],
                            dtype=np.float32)

            # depth validity: z must be positive and confidence acceptable
            depth_valid = (kp3d[:, 2] > 0) & (conf >= CONF_THR)
            kp3d_for_angle = kp3d.copy()
            kp3d_for_angle[~depth_valid] = np.nan

            canvas = frame.copy()
            draw_skeleton(canvas, kp2d, conf, depth_valid)

            l_ang = knee_angle_3d(kp3d_for_angle[L_HIP],
                                  kp3d_for_angle[L_KNE],
                                  kp3d_for_angle[L_ANK])
            r_ang = knee_angle_3d(kp3d_for_angle[R_HIP],
                                  kp3d_for_angle[R_KNE],
                                  kp3d_for_angle[R_ANK])
            l_str = "  --" if np.isnan(l_ang) else f"{l_ang:5.1f}"
            r_str = "  --" if np.isnan(r_ang) else f"{r_ang:5.1f}"
            put_text(canvas, f"L knee: {l_str} deg", (10, 24))
            put_text(canvas, f"R knee: {r_str} deg", (10, 48))

            # downscale to a corner-friendly size
            h, w = canvas.shape[:2]
            display = cv2.resize(canvas,
                                 (WINDOW_W, int(h * WINDOW_W / w)),
                                 interpolation=cv2.INTER_AREA)
            cv2.imshow(WINDOW, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 27 = ESC
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyAllWindows()
    print("[debug] bye")


if __name__ == "__main__":
    main()