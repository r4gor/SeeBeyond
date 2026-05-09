"""
Phase 1 sanity check #2: 3D matplotlib plot of one captured skeleton.

Captures the first detection with confidence > THRESHOLD, then renders the
17 keypoints + skeleton edges in 3D. Use this to verify:

    * scale is realistic (hip-to-knee ~40 cm, knee-to-ankle ~40 cm,
                          shoulder-to-hip ~50 cm, total height ~150-190 cm)
    * vertical ordering is right (head above hips above knees above ankles)
    * left/right sides are not swapped
    * z (depth) values are plausible (~150-300 cm if the athlete is ~2 m
                                       from the camera)

Run from the project root:
    python scripts/plot_skeleton_3d.py
or, if you want to overlay multiple frames for comparison:
    python scripts/plot_skeleton_3d.py --n 5
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import (
    COCO_KEYPOINT_NAMES,
    COCO_SKELETON_EDGES,
    open_oak_skeleton_stream,
)


CONF_THRESHOLD = 0.5
LEFT_SIDE = {5, 7, 9, 11, 13, 15}      # left shoulder/elbow/wrist/hip/knee/ankle
RIGHT_SIDE = {6, 8, 10, 12, 14, 16}


def _capture(n_frames: int):
    grabbed = []
    with open_oak_skeleton_stream(fps=24) as stream:
        for skeleton in stream:
            if skeleton.detection_confidence < CONF_THRESHOLD:
                continue
            grabbed.append(skeleton)
            print(f"[capture] grabbed frame {len(grabbed)}/{n_frames} "
                  f"(conf={skeleton.detection_confidence:.2f})")
            if len(grabbed) >= n_frames:
                break
    return grabbed


def _plot(skeletons):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for s_idx, skel in enumerate(skeletons):
        pts = np.array([[kp.x_cm, kp.y_cm, kp.z_cm] for kp in skel.keypoints])

        # Camera frame is X right, Y down, Z forward. For visualization we
        # want "up" to actually point up, so we flip Y.
        xs = pts[:, 0]
        ys_up = -pts[:, 1]
        zs = pts[:, 2]

        # Color: left side blue, right side red, midline gray.
        colors = []
        for i in range(len(pts)):
            if i in LEFT_SIDE:
                colors.append("tab:blue")
            elif i in RIGHT_SIDE:
                colors.append("tab:red")
            else:
                colors.append("tab:gray")

        # Slightly fade older frames if multiple skeletons.
        alpha = 1.0 if len(skeletons) == 1 else 0.3 + 0.7 * (s_idx / max(1, len(skeletons) - 1))

        ax.scatter(xs, zs, ys_up, c=colors, s=40, alpha=alpha)

        # Draw skeleton edges.
        for a, b in COCO_SKELETON_EDGES:
            ax.plot(
                [xs[a], xs[b]],
                [zs[a], zs[b]],
                [ys_up[a], ys_up[b]],
                color="black", alpha=alpha * 0.6, linewidth=1.5,
            )

        # Label key joints on the last (or only) skeleton.
        if s_idx == len(skeletons) - 1:
            for i, name in enumerate(COCO_KEYPOINT_NAMES):
                if name in {"left_hip", "left_knee", "left_ankle",
                            "left_shoulder", "nose"}:
                    ax.text(xs[i], zs[i], ys_up[i], f"  {name}", fontsize=8)

    ax.set_xlabel("X (cm)  - left/right")
    ax.set_ylabel("Z (cm)  - depth from camera")
    ax.set_zlabel("Height (cm)  - up")
    ax.set_title("OAK 4 D 3D skeleton  (Phase 1 sanity check)")

    # Print quick numerical sanity stats too.
    skel = skeletons[-1]
    by = {kp.name: np.array([kp.x_cm, kp.y_cm, kp.z_cm]) for kp in skel.keypoints}
    def dist(a, b): return float(np.linalg.norm(by[a] - by[b]))
    print()
    print("[sanity]  distances on the last frame (cm):")
    print(f"    left  hip-knee   = {dist('left_hip', 'left_knee'):6.1f}    (expect 35-50)")
    print(f"    left  knee-ankle = {dist('left_knee', 'left_ankle'):6.1f}    (expect 35-50)")
    print(f"    left  shoulder-hip = {dist('left_shoulder', 'left_hip'):6.1f}  (expect 40-60)")
    print(f"    head-to-feet     = {dist('nose', 'left_ankle'):6.1f}    (expect 140-185)")
    print(f"    median z (depth) = {np.median([kp.z_cm for kp in skel.keypoints]):6.1f}    "
          f"(expect ~your camera-to-athlete distance, e.g. 200)")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1,
                        help="how many frames to capture and overlay")
    args = parser.parse_args()

    skels = _capture(args.n)
    if not skels:
        print("[error] no detection above confidence threshold; is someone in frame?")
        return
    _plot(skels)


if __name__ == "__main__":
    main()
