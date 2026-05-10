# FORMA

**GDG AI Hack Polimi Hackathon**

AI-powered squat coach using a depth camera. The system watches you squat in real time, counts your reps, scores your form, and gives instant feedback on a wearable display — all without any wearable sensors.

---

## How it works

A **Luxonis OAK-4D** depth camera runs YOLOv8-large-pose on-device to extract a full 3D skeleton at 24 fps. The depth dimension is the key: the camera deprojects each joint into real-world centimetres, so the system can measure *exactly* how far your hips are below your knees — not just a 2D angle proxy.

A **Random Forest classifier** trained on 65 labelled reps decides the verdict (good / shallow / forward lean) using 10 depth-derived features: hip-to-knee depth delta, trunk angle, knee angle at bottom, ankle drift, and more.

An **M5Stack Core2** (ESP32, 320×240 display, speaker) connects over WiFi/MQTT and shows live state, rep count, verdict, and feedback in four zones. It plays a tone after each rep — high pitch for good, low pitch for shallow.

The laptop runs a debug overlay with a Z-depth-coloured skeleton and a live depth panel showing hip/knee distances, parallel delta, trunk angle, and a squat depth gauge.

---

## Stack

| Layer | Tech |
|---|---|
| Depth camera | Luxonis OAK-4D (RVC4, 48 TOPS, stereo depth) |
| Pose model | YOLOv8-large-pose — runs on-device |
| 3D keypoints | SpatialLocationCalculator — on-device deprojection |
| Rep counting | State machine (STANDING → DESCENDING → BOTTOM → ASCENDING) |
| Classification | scikit-learn Random Forest (300 trees, balanced classes) |
| Display | M5Stack Core2 — ArduinoJson, PubSubClient, FreeRTOS audio task |
| Transport | MQTT (paho-mqtt ↔ mosquitto) |
| Overlay | OpenCV — depth-gradient skeleton, live depth panel |

---

## Running it

```bash
# install dependencies
pip install -r requirements.txt   # or: uv sync

# run (needs OAK camera connected + MQTT broker running)
python run.py --broker <broker-ip>

# flags
--no-sound      # disable MQTT audio (no Core2 needed)
--no-overlay    # disable OpenCV debug window
--fps 24        # camera FPS (default 24)
```

Flash the Core2 firmware with PlatformIO:

```bash
cd coach/core2
pio run --target upload
```

Copy your WiFi credentials and broker IP into `coach/core2/src/config.h` (see `config_example.h`).

---

## Project structure

```
run.py                        # entry point — wires camera → classifier → MQTT
squat-evaluator/
  src/pipeline.py             # OAK pipeline, 3D keypoint extraction
  src/rep_counter.py          # state machine rep counter
  src/features.py             # 10 depth-derived features
  src/classifier.py           # Random Forest predict + rule-based feedback
  data/                       # training reps (.npy) + feature stats
coach/
  core2/                      # M5Stack Core2 Arduino firmware
  backend/sound/voice.py      # ElevenLabs TTS + MQTT audio helpers
```
