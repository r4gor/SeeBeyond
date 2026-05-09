"""
Replay a saved rep: video frame + 3D skeleton, side by side (matplotlib).

Usage:
  python scripts/view_rep.py                  # cycle through all reps in manifest
  python scripts/view_rep.py rep_0023         # jump to a specific rep
  python scripts/view_rep.py --label shallow  # cycle reps with this label only

Controls (matplotlib window):
  slider          scrub frames
  SPACE           play / pause
  RIGHT / LEFT    next / previous rep
  d               DELETE current rep (files + manifest row), then advance
  q               quit

The 3D skeleton uses your camera frame: x right, y down, z forward (cm).
Plot maps that to: x = x, y_axis = depth z, z_axis = up (-y).
You can rotate the 3D plot with click-drag.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
MANIFEST_HEADER = ["rep_id", "label", "n_frames", "fps", "mp4", "npy", "timestamp"]

# COCO 17 keypoint connections
COCO_EDGES = [
    (5, 7), (7, 9),                  # left arm
    (6, 8), (8, 10),                 # right arm
    (5, 6),                          # shoulders
    (5, 11), (6, 12),                # torso sides
    (11, 12),                        # hips
    (11, 13), (13, 15),              # left leg
    (12, 14), (14, 16),              # right leg
    (0, 1), (0, 2), (1, 3), (2, 4),  # face
]


# ---------------------------------------------------------------- manifest

def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        sys.exit(f"no manifest at {MANIFEST_PATH}")
    with open(MANIFEST_PATH, newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(rows: list[dict]) -> None:
    with open(MANIFEST_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_HEADER)
        w.writeheader()
        w.writerows(rows)


def filter_rows(rows, rep_id=None, label=None):
    out = rows
    if rep_id:
        # Accept "rep_0023" or full filename "rep_0023_good"
        parts = rep_id.split("_")
        norm = "_".join(parts[:2]) if len(parts) >= 2 else rep_id
        out = [r for r in out if r["rep_id"] == norm]
    if label:
        out = [r for r in out if r["label"] == label]
    return out


def load_rep(row):
    cap = cv2.VideoCapture(str(DATA_DIR / row["mp4"]))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    kp3d = np.load(DATA_DIR / row["npy"])
    n = min(len(frames), kp3d.shape[0])  # guard against drift
    return frames[:n], kp3d[:n]


def delete_rep_on_disk(row):
    for rel in (row["mp4"], row["npy"]):
        p = DATA_DIR / rel
        if p.exists():
            p.unlink()
    full = load_manifest()
    full = [r for r in full if r["rep_id"] != row["rep_id"]]
    write_manifest(full)


# ---------------------------------------------------------------- viewer

def viewer(rows: list[dict]):
    state = {"idx": 0, "frame": 0, "playing": False, "loaded": None}

    fig = plt.figure(figsize=(13, 6))
    fig.canvas.manager.set_window_title("squat rep viewer")
    ax_img = fig.add_subplot(1, 2, 1)
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
    plt.subplots_adjust(bottom=0.16, wspace=0.05)
    slider_ax = fig.add_axes([0.13, 0.05, 0.74, 0.03])
    slider = Slider(slider_ax, "frame", 0, 1, valinit=0, valstep=1)

    def render():
        if state["loaded"] is None or not rows:
            return
        frames, kp3d = state["loaded"]
        f = state["frame"]
        row = rows[state["idx"]]

        ax_img.clear()
        ax_img.imshow(frames[f])
        ax_img.set_title(
            f"{row['rep_id']}  [{row['label']}]   "
            f"frame {f+1}/{len(frames)}   fps={row['fps']}    "
            f"({state['idx']+1}/{len(rows)})"
        )
        ax_img.axis("off")

        ax_3d.clear()
        kp = kp3d[f]
        valid = ~np.isnan(kp).any(axis=1)

        # Camera frame (x right, y down, z forward) -> plot (x, depth, up)
        xs = kp[:, 0]
        ys_plot = kp[:, 2]   # depth
        zs_plot = -kp[:, 1]  # up

        ax_3d.scatter(xs[valid], ys_plot[valid], zs_plot[valid],
                      c="tab:blue", s=22, depthshade=False)
        for i, j in COCO_EDGES:
            if valid[i] and valid[j]:
                ax_3d.plot([xs[i], xs[j]],
                           [ys_plot[i], ys_plot[j]],
                           [zs_plot[i], zs_plot[j]],
                           color="tab:blue", linewidth=2)

        if valid.any():
            ks = kp[valid]
            ranges = np.array([
                ks[:, 0].max() - ks[:, 0].min(),
                ks[:, 2].max() - ks[:, 2].min(),
                ks[:, 1].max() - ks[:, 1].min(),
            ])
            r = ranges.max() / 2 + 15
            mid_x = (ks[:, 0].max() + ks[:, 0].min()) / 2
            mid_y = (ks[:, 2].max() + ks[:, 2].min()) / 2
            mid_z = -(ks[:, 1].max() + ks[:, 1].min()) / 2
            ax_3d.set_xlim(mid_x - r, mid_x + r)
            ax_3d.set_ylim(mid_y - r, mid_y + r)
            ax_3d.set_zlim(mid_z - r, mid_z + r)

        ax_3d.set_xlabel("x (cm)")
        ax_3d.set_ylabel("depth z (cm)")
        ax_3d.set_zlabel("up -y (cm)")
        ax_3d.set_title(f"3D skeleton   valid {valid.sum()}/17")
        try:
            ax_3d.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        fig.canvas.draw_idle()

    def load_current():
        if not rows:
            plt.close(fig)
            return
        row = rows[state["idx"]]
        print(f"[view] {state['idx']+1}/{len(rows)}  "
              f"{row['rep_id']} ({row['label']})")
        state["loaded"] = load_rep(row)
        state["frame"] = 0
        n = len(state["loaded"][0])
        slider.valmax = max(0, n - 1)
        slider.ax.set_xlim(slider.valmin, slider.valmax)
        slider.eventson = False
        slider.set_val(0)
        slider.eventson = True
        render()

    def on_slider(val):
        state["frame"] = int(val)
        render()

    def on_key(event):
        if event.key == "q":
            plt.close(fig)
        elif event.key == " ":
            state["playing"] = not state["playing"]
        elif event.key == "right" and state["idx"] + 1 < len(rows):
            state["idx"] += 1
            load_current()
        elif event.key == "left" and state["idx"] > 0:
            state["idx"] -= 1
            load_current()
        elif event.key == "d":
            row = rows[state["idx"]]
            delete_rep_on_disk(row)
            print(f"[view] DELETED {row['rep_id']} ({row['label']})")
            rows.pop(state["idx"])
            if not rows:
                print("[view] no reps left")
                plt.close(fig)
                return
            state["idx"] = min(state["idx"], len(rows) - 1)
            load_current()

    slider.on_changed(on_slider)
    fig.canvas.mpl_connect("key_press_event", on_key)
    load_current()

    timer = fig.canvas.new_timer(interval=33)  # ~30 fps playback

    def tick():
        if state["playing"] and state["loaded"] is not None:
            n = len(state["loaded"][0])
            state["frame"] = (state["frame"] + 1) % n
            slider.eventson = False
            slider.set_val(state["frame"])
            slider.eventson = True
            render()

    timer.add_callback(tick)
    timer.start()
    plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rep_id", nargs="?", default=None,
                   help="rep id like 'rep_0023' (optional)")
    p.add_argument("--label", default=None,
                   help="filter to one label, e.g. 'shallow'")
    args = p.parse_args()

    rows = filter_rows(load_manifest(), rep_id=args.rep_id, label=args.label)
    if not rows:
        sys.exit("no matching reps")
    viewer(rows)


if __name__ == "__main__":
    main()