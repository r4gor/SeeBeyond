"""
Phase 1 sanity check #1: console stream.

Runs the pipeline, prints frames-per-second and the 3D positions of a few
key joints. Quick way to confirm the OAK is producing sensible numbers
before we bother with a 3D plot.

Run from the project root:
    python -m scripts.run_stream
or:
    python scripts/run_stream.py
"""

import sys
import time
from pathlib import Path

# Allow running as a plain script (without `python -m`).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipeline import open_oak_skeleton_stream


WATCH = ["left_hip", "left_knee", "left_ankle"]  # squat-relevant joints


def main():
    print("[run_stream] opening OAK 4 D pipeline...")
    last_log = time.monotonic()
    frames_in_window = 0

    with open_oak_skeleton_stream(fps=24) as stream:
        for skeleton in stream:
            frames_in_window += 1
            now = time.monotonic()

            # One status line per second.
            if now - last_log >= 1.0:
                fps = frames_in_window / (now - last_log)
                kps = {kp.name: kp for kp in skeleton.keypoints}

                parts = [f"fps={fps:5.1f}", f"conf={skeleton.detection_confidence:.2f}"]
                for name in WATCH:
                    kp = kps[name]
                    parts.append(
                        f"{name}=({kp.x_cm:+6.1f},{kp.y_cm:+6.1f},{kp.z_cm:6.1f})cm"
                    )
                print(" | ".join(parts))

                last_log = now
                frames_in_window = 0


if __name__ == "__main__":
    main()
