"""
Capture labeled squat reps for classifier training (Phase 2).

Pre-set the label, then record. Manifest CSV is the single source of truth.

Keys:
  1   label = good
  2   label = shallow
  3   label = forward_lean
  4   label = heels_lifting     (swap to knees_caving in LABEL_KEYS if your
                                 trainer fakes that more convincingly)
  SPACE   start / stop recording   (auto-saves on stop)
  A       abort current recording  (no save)
  D       discard last saved rep   (delete files + remove manifest row)
  Q       quit

Output layout:
  data/manifest.csv
  data/reps/rep_0001_good.mp4
  data/reps/rep_0001_good.npy   shape (n_frames, 17, 3) float32
                                 NaN for invalid/occluded joints

Run from project root:
  python scripts/capture_reps.py
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

COCO_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def draw_skeleton(canvas, kp2d):
    if kp2d is None:
        return
    for i, j in COCO_EDGES:
        if not (np.isnan(kp2d[i]).any() or np.isnan(kp2d[j]).any()):
            p1 = tuple(kp2d[i].astype(int))
            p2 = tuple(kp2d[j].astype(int))
            cv2.line(canvas, p1, p2, (0, 255, 120), 2)
    for i in range(17):
        if not np.isnan(kp2d[i]).any():
            cv2.circle(canvas, tuple(kp2d[i].astype(int)), 4, (0, 255, 255), -1)

# ===== ADAPTER: edit if your pipeline interface differs ====================
# Expected:
#   `with Pipeline() as pipe:` works
#   pipe.read() -> (rgb_bgr: np.uint8 HxWx3, kp3d: np.float32 (17,3))
#   kp3d coordinates in cm. NaN entries mark invalid joints.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline import open_oak_skeleton_stream

class Pipeline:
    """Wraps open_oak_skeleton_stream() to match the (rgb, kp3d) read() contract."""

    def __init__(self, fps: int = 24, conf_threshold: float = 0.3):
        self._fps = fps
        self._conf = conf_threshold
        self._cm = None
        self._stream = None

    def __enter__(self):
        self._cm = open_oak_skeleton_stream(fps=self._fps, with_frames=True)
        self._stream = self._cm.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._cm.__exit__(exc_type, exc, tb)

    def read(self):
        try:
            frame, skel = next(self._stream)
        except StopIteration:
            return None, None, None
        kps = skel.keypoints
        kp3d = np.array([[k.x_cm, k.y_cm, k.z_cm] for k in kps], dtype=np.float32)
        kp2d = np.array([[k.x_px, k.y_px] for k in kps], dtype=np.float32)
        confs = np.array([k.confidence for k in kps], dtype=np.float32)
        invalid = (kp3d[:, 2] <= 0) | (confs < self._conf)
        kp3d[invalid] = np.nan
        kp2d[invalid] = np.nan
        return frame, kp3d, kp2d
# ===========================================================================

LABEL_KEYS = {
    ord("1"): "good",
    ord("2"): "shallow",
    ord("3"): "forward_lean",
    ord("4"): "heels_lifting",
}
DEFAULT_LABEL = "good"
MIN_FRAMES = 5  # below this, ignore as accidental press

DATA_DIR = ROOT / "data"
REPS_DIR = DATA_DIR / "reps"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
MANIFEST_HEADER = ["rep_id", "label", "n_frames", "fps", "mp4", "npy", "timestamp"]
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
WINDOW = "capture"


# ---------------------------------------------------------------- manifest

def ensure_layout() -> None:
    REPS_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "w", newline="") as f:
            csv.writer(f).writerow(MANIFEST_HEADER)


def read_manifest() -> list[dict]:
    with open(MANIFEST_PATH, newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(rows: list[dict]) -> None:
    with open(MANIFEST_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_HEADER)
        w.writeheader()
        w.writerows(rows)


def append_manifest(row: dict) -> None:
    with open(MANIFEST_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=MANIFEST_HEADER).writerow(row)


def next_rep_id(rows: list[dict]) -> int:
    used = []
    for r in rows:
        try:
            used.append(int(r["rep_id"].split("_")[1]))
        except (IndexError, ValueError):
            pass
    return max(used) + 1 if used else 1


def label_counts(rows: list[dict]) -> dict[str, int]:
    counts = {lbl: 0 for lbl in LABEL_KEYS.values()}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    return counts


# ---------------------------------------------------------------- save / discard

def save_rep(frames_bgr, kps3d, label, t_elapsed, rows) -> dict:
    n = len(frames_bgr)
    fps = n / t_elapsed if t_elapsed > 0 else 30.0
    rep_id = f"rep_{next_rep_id(rows):04d}"
    base = f"{rep_id}_{label}"
    mp4_rel = f"reps/{base}.mp4"
    npy_rel = f"reps/{base}.npy"

    h, w = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(str(DATA_DIR / mp4_rel), FOURCC, fps, (w, h))
    if not writer.isOpened():
        print(f"[capture] WARNING: VideoWriter failed to open ({mp4_rel}); "
              f"check your OpenCV mp4v codec. .npy will still save.")
    for fr in frames_bgr:
        writer.write(fr)
    writer.release()

    arr = np.stack([np.asarray(k, dtype=np.float32) for k in kps3d], axis=0)
    np.save(DATA_DIR / npy_rel, arr)

    row = {
        "rep_id": rep_id,
        "label": label,
        "n_frames": n,
        "fps": f"{fps:.2f}",
        "mp4": mp4_rel,
        "npy": npy_rel,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    append_manifest(row)
    return row


def discard_last(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    last = rows[-1]
    for rel in (last["mp4"], last["npy"]):
        p = DATA_DIR / rel
        if p.exists():
            p.unlink()
    write_manifest(rows[:-1])
    return last


# ---------------------------------------------------------------- UI overlay

def overlay_ui(canvas, *, label, recording, rec_n, rec_t, counts, last_id, valid):
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (w, 56), (20, 20, 20), -1)
    cv2.putText(canvas, f"label: {label}", (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2)
    counts_str = "  ".join(f"{k[:5]}:{v}" for k, v in counts.items())
    cv2.putText(canvas, counts_str, (260, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    if recording:
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
        cv2.circle(canvas, (w - 90, 28), 9, (0, 0, 255), -1)
        cv2.putText(canvas, f"REC {rec_n}f {rec_t:4.1f}s",
                    (w - 250, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    color = (0, 220, 0) if valid >= 14 else (0, 165, 255) if valid >= 10 else (0, 0, 255)
    cv2.putText(canvas, f"valid joints: {valid}/17", (12, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if last_id:
        cv2.putText(canvas, f"last saved: {last_id}", (240, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(canvas, "1-4=label  SPACE=rec  A=abort  D=discard last  Q=quit",
                (max(0, w - 540), h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)


# ---------------------------------------------------------------- main loop

def main() -> None:
    ensure_layout()
    rows = read_manifest()
    last_id = rows[-1]["rep_id"] if rows else None
    label = DEFAULT_LABEL
    recording = False
    buf_bgr: list = []
    buf_kp3d: list = []
    t0 = 0.0

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print(f"[capture] {len(rows)} reps in manifest; "
          f"next id = rep_{next_rep_id(rows):04d}")
    print(f"[capture] keys: 1-4 label  SPACE rec  A abort  D discard last  Q quit")

    with Pipeline() as pipe:
        while True:
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

            rgb, kp3d, kp2d = pipe.read()
            if rgb is None:
                cv2.waitKey(1)
                continue

            kp_arr = np.asarray(kp3d, dtype=np.float32) if kp3d is not None else None
            valid = (
                int(np.sum(~np.isnan(kp_arr).any(axis=1)))
                if kp_arr is not None and kp_arr.shape == (17, 3)
                else 0
            )

            if recording and kp_arr is not None and kp_arr.shape == (17, 3):
                buf_bgr.append(rgb.copy())
                buf_kp3d.append(kp_arr.copy())

            preview = rgb.copy()
            
            draw_skeleton(preview, kp2d)

            overlay_ui(
                preview,
                label=label,
                recording=recording,
                rec_n=len(buf_bgr),
                rec_t=time.time() - t0 if recording else 0.0,
                counts=label_counts(rows),
                last_id=last_id,
                valid=valid,
            )
            cv2.imshow(WINDOW, preview)
            key = cv2.waitKey(1) & 0xFF

            if key == 0xFF:
                continue
            if key == ord("q"):
                if recording:
                    print("[capture] aborting in-progress rep on quit")
                break
            if key in LABEL_KEYS:
                label = LABEL_KEYS[key]
                print(f"[capture] label = {label}")
            elif key == ord(" "):
                if not recording:
                    recording = True
                    buf_bgr, buf_kp3d = [], []
                    t0 = time.time()
                    print(f"[capture] REC start (label = {label})")
                else:
                    recording = False
                    elapsed = time.time() - t0
                    if len(buf_bgr) < MIN_FRAMES:
                        print(f"[capture] only {len(buf_bgr)} frames — discarded")
                    else:
                        row = save_rep(buf_bgr, buf_kp3d, label, elapsed, rows)
                        rows.append(row)
                        last_id = row["rep_id"]
                        print(f"[capture] saved {row['rep_id']} "
                              f"label={label} frames={row['n_frames']} fps={row['fps']}")
                    buf_bgr, buf_kp3d = [], []
            elif key == ord("a"):
                if recording:
                    recording = False
                    buf_bgr, buf_kp3d = [], []
                    print("[capture] aborted")
            elif key == ord("d"):
                if recording:
                    print("[capture] D ignored while recording (use A first)")
                else:
                    removed = discard_last(rows)
                    if removed:
                        rows = read_manifest()
                        last_id = rows[-1]["rep_id"] if rows else None
                        print(f"[capture] discarded {removed['rep_id']}")
                    else:
                        print("[capture] nothing to discard")

    cv2.destroyAllWindows()
    final = label_counts(read_manifest())
    print(f"[capture] done. counts: {final}  total: {sum(final.values())}")


if __name__ == "__main__":
    main()