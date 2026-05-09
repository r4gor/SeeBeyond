"""
Phase 3 smoke test: live rep counting from the OAK 4 D 3D skeleton stream.
Stand side-on to the camera and squat. Each completed rep prints to the
terminal. Ctrl-C to exit.

Run from the project root:
    python -m scripts.run_counter
or:
    python scripts/run_counter.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import open_oak_skeleton_stream
from src.rep_counter import RepCounter, RepData


def on_rep_started():
    print("  [rep] descent started")


def on_rep(rep: RepData):
    nan_frames = int(sum(1 for f in rep.trajectory if any(c != c for c in f.flatten())))
    print(
        f"REP {rep.rep_number:>3}  "
        f"min_angle={rep.min_knee_angle:5.1f}°  "
        f"frames={len(rep.trajectory):3d} (nan={nan_frames})  "
        f"bottom_idx={rep.bottom_frame_idx:3d}  "
        f"dur={rep.duration_s:4.2f}s"
    )


def main():
    counter = RepCounter(on_rep_complete=on_rep, on_rep_started=on_rep_started)

    last_state = None
    last_log = time.monotonic()

    print("[run_counter] opening OAK 4 D pipeline...")
    with open_oak_skeleton_stream(fps=24) as stream:
        print("running. squat away. Ctrl-C to stop.")
        for skel in stream:
            counter.update(skel)

            if counter.current_state != last_state:
                print(f"  [state] {last_state} -> {counter.current_state}")
                last_state = counter.current_state

            now = time.monotonic()
            if now - last_log >= 0.5 and counter.latest_angle is not None:
                print(f"  angle={counter.latest_angle:5.1f}°  state={counter.current_state}")
                last_log = now


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")