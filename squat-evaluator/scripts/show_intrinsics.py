"""
Phase 1 sanity check #3: print factory-calibrated camera intrinsics.

You don't need this to run the pipeline (the on-device SpatialLocationCalculator
already uses these intrinsics for deprojection), but it's a useful first-touch
check that the OAK is communicating and its calibration is readable.

Run:
    python scripts/show_intrinsics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import get_color_intrinsics


def main():
    for w, h in [(1920, 1080), (640, 352)]:
        fx, fy, cx, cy = get_color_intrinsics(w, h)
        print(f"CAM_A @ {w}x{h}:  fx={fx:7.2f}  fy={fy:7.2f}  cx={cx:7.2f}  cy={cy:7.2f}")


if __name__ == "__main__":
    main()
