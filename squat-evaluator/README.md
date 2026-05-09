# Squat Form Evaluator — Phase 1

DepthAI v3 pipeline that streams 3D skeletons from an OAK 4 D, on-device.

```
squat-evaluator/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   └── pipeline.py            ← the pipeline + Skeleton generator (Phase 2-4 imports this)
├── scripts/
│   ├── show_intrinsics.py     ← sanity #1: read factory calibration
│   ├── run_stream.py          ← sanity #2: console FPS + key joints
│   └── plot_skeleton_3d.py    ← sanity #3: matplotlib 3D plot of one frame
└── data/                      ← (empty for now; Phase 4 trainer reps go here)
```

## Install (one-time)

```bash
cd ~/squat-evaluator
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

`depthai>=3.4` is on PyPI, so a plain `pip install` works — no extra index URL needed.

## Run order (Phase 1 sanity gate)

Plug in the OAK 4 D via the PoE switch, then in the project root with `.venv` active:

**1. Confirm the camera is reachable and calibration loads.**
```bash
python scripts/show_intrinsics.py
```
You should see four numbers per resolution. If this hangs or errors, the OAK isn't being discovered — fix that before doing anything else.

**2. Stream skeletons to the console.** Stand 2 m from the camera in the frame.
```bash
python scripts/run_stream.py
```
Once a second you should see something like:
```
fps= 22.1 | conf=0.91 | left_hip=( -3.4,-12.7, 198.4)cm | left_knee=( -4.1, +28.5, 196.2)cm | left_ankle=( -5.0, +71.8, 195.0)cm
```
Sanity checks: hip Y should be smaller than knee Y, which should be smaller than ankle Y (because Y is "down" in camera frame). Z should match how far away you actually are.

**3. Visual sanity check — the 3D plot.**
```bash
python scripts/plot_skeleton_3d.py
```
A matplotlib window should open showing a recognizable skeleton. The terminal also prints expected-range distances:
```
left  hip-knee   =  41.3    (expect 35-50)
left  knee-ankle =  39.7    (expect 35-50)
left  shoulder-hip = 48.2    (expect 40-60)
head-to-feet     = 168.4    (expect 140-185)
median z (depth) = 197.5    (expect ~your camera-to-athlete distance)
```
If the numbers are way off (e.g. hip-knee = 800 cm), the depth alignment isn't working — re-check that `dai.node.ImageAlign` is wired in the RVC4 branch.

If you want to overlay several frames to see motion:
```bash
python scripts/plot_skeleton_3d.py --n 5
```

## What Phase 2/3/4 will import

```python
from src.pipeline import open_oak_skeleton_stream

with open_oak_skeleton_stream(fps=24) as stream:
    for skeleton in stream:
        # skeleton.timestamp_s
        # skeleton.detection_confidence
        # skeleton.keypoints  - list of 17 Keypoint(name, x_px, y_px, x_cm, y_cm, z_cm, confidence)
        ...
```

Each `Skeleton` is one frame containing one athlete. Frames with no detection are skipped automatically.
