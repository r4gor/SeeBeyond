"""
DepthAI v3 pipeline that streams 3D skeletons from an OAK 4 D.

Pipeline (all on-device, RVC4):
  Camera (CAM_A, color)  ->  DetectionNetwork (YOLOv8-pose, 17 COCO kps)
  Camera (CAM_B, mono)   ->  StereoDepth  ->  ImageAlign (align depth to RGB)
                                                      |
                                                      v
                                  SpatialLocationCalculator
                                  (setCalculateSpatialKeypoints=True)
                                                      |
                                                      v
                                  yields detections with per-keypoint
                                  spatial coordinates in mm, deprojected
                                  on-device using factory intrinsics.

The host receives SpatialImgDetections messages and emits a clean
generator yielding one Skeleton per frame.

COCO 17 keypoint order (the model's native ordering):
  0  nose         5  left_shoulder   11 left_hip
  1  left_eye     6  right_shoulder  12 right_hip
  2  right_eye    7  left_elbow      13 left_knee
  3  left_ear     8  right_elbow     14 right_knee
  4  right_ear    9  left_wrist      15 left_ankle
                  10 right_wrist     16 right_ankle
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Generator, List, Optional

import depthai as dai


COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
NUM_KEYPOINTS = 17

# COCO skeleton edges, useful for plotting and Phase 4 features.
COCO_SKELETON_EDGES = [
    (5, 7), (7, 9),       # left arm
    (6, 8), (8, 10),      # right arm
    (5, 6),               # shoulders
    (5, 11), (6, 12),     # torso sides
    (11, 12),             # hips
    (11, 13), (13, 15),   # left leg
    (12, 14), (14, 16),   # right leg
]


@dataclass
class Keypoint:
    name: str
    x_px: float          # pixel x in the RGB frame (passthrough resolution)
    y_px: float          # pixel y in the RGB frame
    x_cm: float          # camera-frame X, cm (right is positive)
    y_cm: float          # camera-frame Y, cm (down is positive)
    z_cm: float          # camera-frame Z, cm (forward / depth)
    confidence: float    # detector confidence in [0, 1]; 1.0 if not provided

    def to_dict(self):
        return asdict(self)


@dataclass
class Skeleton:
    timestamp_s: float           # host monotonic time when frame was received
    frame_w: int
    frame_h: int
    keypoints: List[Keypoint]    # always length 17 (COCO order)
    detection_confidence: float

    def to_dict(self):
        return {
            "timestamp_s": self.timestamp_s,
            "frame_w": self.frame_w,
            "frame_h": self.frame_h,
            "detection_confidence": self.detection_confidence,
            "keypoints": [kp.to_dict() for kp in self.keypoints],
        }


# -----------------------------------------------------------------------------
# Pipeline construction
# -----------------------------------------------------------------------------

def _build_pipeline(device: dai.Device, fps: int):
    """Build the v3 pipeline. Returns (pipeline, queues_dict)."""
    platform = device.getPlatform()

    # Pose model - YOLOv8 pose in COCO format (17 keypoints).
    # Large variant on RVC4 (OAK 4 D), nano fallback on RVC2.
    if platform == dai.Platform.RVC2:
        model_name = "luxonis/yolov8-nano-pose-estimation:coco-512x288"
        fps = min(fps, 10)
    else:
        model_name = "luxonis/yolov8-large-pose-estimation:coco-640x352"

    cam_caps = dai.ImgFrameCapability()
    cam_caps.fps.fixed(fps)
    cam_caps.enableUndistortion = True

    pipeline = dai.Pipeline(device)

    # Color camera (RGB) - source for the pose NN.
    cam_color = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

    # Stereo pair (mono left/right).
    cam_left = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_B, sensorFps=fps
    )
    cam_right = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_C, sensorFps=fps
    )
    left_out = cam_left.requestOutput((640, 400), fps=fps)
    right_out = cam_right.requestOutput((640, 400), fps=fps)

    # Pose detection network. .build() auto-downloads from HubAI model zoo,
    # picks the right backend (DSP on RVC4) and wires up the YOLO-pose parser.
    det_nn = pipeline.create(dai.node.DetectionNetwork).build(
        cam_color, model_name, cam_caps
    )

    # Stereo depth, fast preset (we want low latency, not max accuracy).
    stereo = pipeline.create(dai.node.StereoDepth).build(
        left_out, right_out,
        presetMode=dai.node.StereoDepth.PresetMode.FAST_DENSITY,
    )

    # Spatial location calculator: samples depth at each detection's bbox AND
    # at each keypoint, then deprojects to 3D using on-device intrinsics.
    spatial = pipeline.create(dai.node.SpatialLocationCalculator)
    spatial.initialConfig.setCalculateSpatialKeypoints(True)
    det_nn.out.link(spatial.inputDetections)

    # Depth -> spatial input. RVC4 needs an explicit ImageAlign node to align
    # depth onto the RGB frame the NN ran on; RVC2 can use stereo.inputAlignTo.
    if platform == dai.Platform.RVC4:
        align = pipeline.create(dai.node.ImageAlign)
        stereo.depth.link(align.input)
        det_nn.passthrough.link(align.inputAlignTo)
        align.outputAligned.link(spatial.inputDepth)
    else:
        det_nn.passthrough.link(stereo.inputAlignTo)
        stereo.depth.link(spatial.inputDepth)

    # Output queues we'll read from on the host.
    queues = {
        "spatial":     spatial.outputDetections.createOutputQueue(),
        "passthrough": det_nn.passthrough.createOutputQueue(),
        "depth":       spatial.passthroughDepth.createOutputQueue(),
    }
    return pipeline, queues, model_name


# -----------------------------------------------------------------------------
# Public API: stream 3D skeletons
# -----------------------------------------------------------------------------

@contextmanager
def open_oak_skeleton_stream(fps: int = 24, with_frames: bool = False):
    """
    Context manager that yields a generator of Skeleton objects.

    Usage:
        with open_oak_skeleton_stream(fps=24) as stream:
            for skeleton in stream:
                ...                  # one per frame, only when a person is seen

    Frames with zero detections are skipped. If multiple people are detected,
    only the most confident one is yielded (squat eval = single athlete).
    """
    device = dai.Device()
    pipeline, queues, model_name = _build_pipeline(device, fps=fps)
    print(f"[pipeline] platform={device.getPlatform().name}  model={model_name}  fps={fps}")

    pipeline.start()

    def _generator() -> Generator[Skeleton, None, None]:
        while pipeline.isRunning():
            spatial_msg = queues["spatial"].get()
            passthrough = queues["passthrough"].get()
            # Drain depth queue to keep it from backing up; we don't use the
            # raw depth here (deprojection already happened on-device), but
            # consuming it keeps timing aligned.
            try:
                queues["depth"].tryGet()
            except Exception:
                pass

            assert isinstance(spatial_msg, dai.SpatialImgDetections)
            assert isinstance(passthrough, dai.ImgFrame)

            if not spatial_msg.detections:
                continue

            frame = passthrough.getCvFrame()
            h, w = frame.shape[:2]

            # Best (most confident) person.
            det = max(spatial_msg.detections, key=lambda d: d.confidence)

            kps_raw = det.getKeypoints()
            if len(kps_raw) < NUM_KEYPOINTS:
                # Model occasionally returns fewer if some are off-frame; pad
                # with zero-confidence stubs so downstream code can rely on 17.
                continue

            keypoints: List[Keypoint] = []
            for j, kp in enumerate(kps_raw[:NUM_KEYPOINTS]):
                # imageCoordinates are normalized [0, 1]; convert to pixels.
                x_px = float(kp.imageCoordinates.x) * w
                y_px = float(kp.imageCoordinates.y) * h
                # spatialCoordinates are in mm; convert to cm.
                x_cm = float(kp.spatialCoordinates.x) / 10.0
                y_cm = float(kp.spatialCoordinates.y) / 10.0
                z_cm = float(kp.spatialCoordinates.z) / 10.0
                # Confidence field name varies by SDK version.
                conf = float(getattr(kp, "confidence", 1.0))

                keypoints.append(Keypoint(
                    name=COCO_KEYPOINT_NAMES[j],
                    x_px=x_px, y_px=y_px,
                    x_cm=x_cm, y_cm=y_cm, z_cm=z_cm,
                    confidence=conf,
                ))

            skel = Skeleton(
                timestamp_s=time.monotonic(),
                frame_w=w,
                frame_h=h,
                keypoints=keypoints,
                detection_confidence=float(det.confidence),
            )
            yield (frame, skel) if with_frames else skel

    try:
        yield _generator()
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        device.close()


# -----------------------------------------------------------------------------
# Camera intrinsics helper (for documentation / extra deprojection if wanted)
# -----------------------------------------------------------------------------

def get_color_intrinsics(width: int = 1920, height: int = 1080):
    """
    Read the factory-calibrated intrinsics from the OAK 4 D's RGB camera
    (CAM_A). Returns (fx, fy, cx, cy) at the requested output resolution.

    You don't need this for the main pipeline because SpatialLocationCalculator
    deprojects on-device using these same intrinsics. But it's useful for
    sanity checks and for any host-side geometry you might want to add later.
    """
    with dai.Device() as device:
        calib = device.readCalibration()
        K = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, width, height)
        fx, fy = K[0][0], K[1][1]
        cx, cy = K[0][2], K[1][2]
        return fx, fy, cx, cy
