# SeeBeyond — Technical Documentation
**GDG AI Hack Polimi Hackathon**

---

## System Architecture

Three physical components communicate over a local WiFi network via MQTT:

```
OAK-4D Camera ──(USB3)──► Laptop (Python pipeline) ──(MQTT/WiFi)──► M5Stack Core2
                                  │                                        │
                           OpenCV overlay                          TFT display + speaker
```

---

## Hardware

**Luxonis OAK-4D** — RVC4 SoC, 48 TOPS on-device DSP. Runs YOLOv8-large-pose (640×352) directly on the chip at 24 fps. Stereo RGB+IR modules provide metric depth via hardware-accelerated block matching. Factory-calibrated intrinsics enable sub-centimetre deprojection accuracy at 3 m.

**M5Stack Core2** — ESP32-D0WDQ6, 320×240 IPS TFT, built-in speaker (NS4168 I2S amp), 8 MB PSRAM, 16 MB flash. Runs custom Arduino firmware with FreeRTOS.

---

## Camera Pipeline (DepthAI v3)

```
ColorCamera (640×352, BGR)
    │
    ├──► DetectionNetwork  — YOLOv8-large-pose, COCO-17 keypoints
    │        │
    │    StereoDepth  — disparity → metric depth map
    │        │
    │    ImageAlign  — aligns depth frame to RGB frame
    │        │
    │    SpatialLocationCalculator  — reprojects each of the 17 keypoints
    │                                 from pixel → 3D world (mm) on-device
    └──► Passthrough frame  — raw 640×352 BGR for overlay
```

Coordinate frame: **+X right, +Y down, +Z forward** (camera optical axis). All joint positions arrive in centimetres after `/10` conversion. `z_cm = 0` means no valid stereo match at that pixel.

---

## Rep Counter

State machine with hysteresis to reject noise:

| Transition | Condition |
|---|---|
| STANDING → DESCENDING | knee angle < 160° |
| DESCENDING → BOTTOM | knee angle < 100° |
| BOTTOM → ASCENDING | knee angle > 105° |
| ASCENDING → STANDING | knee angle > 155° → **rep complete** |

Minimum dwell time: 200 ms per state. Knee angle is the mean of left and right knee joint angles derived from 3D positions (hip→knee→ankle vectors).

---

## Feature Extraction & Classification

**10 depth-derived features** computed at the bottom frame (frame of minimum knee angle):

- `hip_minus_knee_y_at_bottom` — hip Y − knee Y in cm; **positive = hip below knee = parallel reached**
- `trunk_angle_at_bottom` — shoulder-midpoint→hip-midpoint vector vs world-up `(0,−1,0)`
- `knee_angle_at_bottom` — 3D joint angle (hip–knee–ankle)
- `ankle_y_drift_max` — maximum vertical heel rise during descent
- `hip_depth_z_at_bottom` — absolute camera distance (cm) to hips
- `lateral_knee_spread`, `knee_symmetry`, `hip_symmetry`, `descent_smoothness`, `rep_duration`

**Classifier:** scikit-learn `RandomForestClassifier` — 300 trees, `min_samples_leaf=2`, `class_weight='balanced'`. Trained on 65 labelled reps (25 good, 20 shallow, 20 forward lean). In-sample accuracy 100%; thresholds auto-calibrated from the good-class distribution (p10/p90 ± σ). A rule-based layer on top converts the verdict into a specific feedback string (e.g. *"Hip 3 cm above parallel"*).

---

## M5Stack Firmware

Four-zone 320×240 display updated via `core2/display` (compact JSON, keys: `r` reps, `v` verdict, `f` feedback, `a` angle, `s` state). Parsed with ArduinoJson v6.

**Audio:** A dedicated FreeRTOS task (`pcm_audio`, 8 KB stack, PSRAM-backed 2 MB buffer via `ps_malloc`) waits on `ulTaskNotifyTake`. The MQTT callback signals it with `xTaskNotifyGive` on `core2/play/pcm/end` — playback is fully async so `mqttLoop()` keeps processing display updates during audio.

**Beeps:** Python generates 880 Hz (good) and 330 Hz (bad) sine tones at startup (44100 Hz 16-bit mono, 180 ms with 8 ms fade), sent as raw PCM over MQTT — no SD card files required.

---

## MQTT Protocol

| Topic | Direction | Payload |
|---|---|---|
| `core2/display` | laptop → Core2 | JSON `{"r":7,"v":"good","f":"Hip 2cm below parallel","a":88,"s":"STANDING"}` |
| `core2/play/pcm/start` | laptop → Core2 | ASCII byte count (reserves PSRAM buffer) |
| `core2/play/pcm/data` | laptop → Core2 | raw PCM chunks ≤ 60 KB each |
| `core2/play/pcm/end` | laptop → Core2 | empty (triggers FreeRTOS playback task) |

All messages QoS 0. MQTT buffer on Core2 set to 60 KB + 256 B to accommodate max chunk size.

---

## Laptop Overlay (OpenCV)

480 px camera feed + 150 px depth panel side-by-side. Skeleton bones coloured by average Z-depth of endpoints (orange=close, green=mid, cyan=far). Squat joints show Z labels in cm. Depth panel: live hip/knee Z distance, parallel delta (hip Y − knee Y, green if positive), trunk angle, and a vertical gauge (red→green) that fills as the athlete descends toward parallel.
