"""
Phase 3: 3D knee-angle state machine for squat rep counting.

Input:  Skeleton from src.pipeline.open_oak_skeleton_stream (one per frame).
Output: on_rep_complete callback fires once per completed rep with the full
        3D trajectory and bottom-of-rep frame index for the Phase 4 classifier.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np

from src.pipeline import Skeleton, Keypoint

# COCO-17 indices (matches pipeline.COCO_KEYPOINT_NAMES order)
L_HIP, R_HIP     = 11, 12
L_KNEE, R_KNEE   = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# State thresholds in degrees. Hysteresis: tighter to enter STANDING/BOTTOM,
# looser to leave, so a noisy value sitting on a boundary doesn't oscillate.
STANDING_ENTER = 160.0
STANDING_EXIT  = 155.0
BOTTOM_ENTER   = 100.0
BOTTOM_EXIT    = 105.0
MIN_DWELL_S    = 0.20

# A keypoint is "occluded" if either confidence is low or depth was unavailable
# on-device (the SpatialLocationCalculator returns z=0 in that case).
KP_CONF_MIN = 0.30

STANDING, DESCENDING, BOTTOM, ASCENDING = "STANDING", "DESCENDING", "BOTTOM", "ASCENDING"


@dataclass
class RepData:
    rep_number: int
    trajectory: np.ndarray   # (n_frames, 17, 3) cm. Invalid keypoints are NaN.
    bottom_frame_idx: int    # index of min-knee-angle frame within trajectory
    min_knee_angle: float    # degrees
    duration_s: float        # wall-clock duration of the rep


# ---------- helpers ----------

def _kp_valid(kp: Keypoint) -> bool:
    return kp.confidence >= KP_CONF_MIN and kp.z_cm > 0.0


def _kp_to_xyz(kp: Keypoint) -> np.ndarray:
    if not _kp_valid(kp):
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    return np.array([kp.x_cm, kp.y_cm, kp.z_cm], dtype=np.float32)


def _skeleton_to_array(skel: Skeleton) -> np.ndarray:
    """(17, 3) cm. NaN where the keypoint is occluded / has no valid depth."""
    return np.stack([_kp_to_xyz(kp) for kp in skel.keypoints], axis=0)


def _angle_at_vertex(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b in degrees. NaN if any input is NaN or vectors degenerate."""
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return float("nan")
    ba, bc = a - b, c - b
    nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
    if nba < 1e-6 or nbc < 1e-6:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0))))


def _knee_angle(arr: np.ndarray, side: str) -> float:
    if side == "L":
        return _angle_at_vertex(arr[L_HIP], arr[L_KNEE], arr[L_ANKLE])
    return _angle_at_vertex(arr[R_HIP], arr[R_KNEE], arr[R_ANKLE])


def _best_knee_angle(arr: np.ndarray) -> Optional[float]:
    """
    Single representative knee angle for state-machine input.
    Both sides valid -> mean (more stable). One valid -> use it. Neither -> None.

    Phase 4 will likely want per-side angles to compute asymmetry features;
    those are easy to recompute from rep.trajectory after the fact.
    """
    aL = _knee_angle(arr, "L")
    aR = _knee_angle(arr, "R")
    valid = [a for a in (aL, aR) if not np.isnan(a)]
    if not valid:
        return None
    return float(np.mean(valid))


# ---------- state machine ----------

class RepCounter:
    def __init__(
        self,
        on_rep_complete: Callable[[RepData], None],
        on_rep_started: Optional[Callable[[], None]] = None,
    ):
        self.on_rep_complete = on_rep_complete
        self.on_rep_started = on_rep_started
        self.rep_count = 0
        self.state = STANDING
        self._state_entered_t: Optional[float] = None
        self._buffer: List[np.ndarray] = []
        self._angles: List[float] = []
        self._rep_start_t: Optional[float] = None
        self._latest_angle: Optional[float] = None

    @property
    def current_state(self) -> str:
        return self.state

    @property
    def latest_angle(self) -> Optional[float]:
        return self._latest_angle

    def update(self, skel: Skeleton) -> None:
        """Feed one frame. Fires on_rep_started / on_rep_complete on boundaries."""
        t = skel.timestamp_s
        if self._state_entered_t is None:
            self._state_entered_t = t

        arr = _skeleton_to_array(skel)
        angle = _best_knee_angle(arr)
        self._latest_angle = angle

        # Buffer every frame between the start of descent and return to standing,
        # including occluded ones (stored as NaN). Phase 4 gets a continuous
        # trajectory and decides whether to interpolate or reject the rep.
        if self.state != STANDING:
            self._buffer.append(arr)
            self._angles.append(angle if angle is not None else float("nan"))

        if angle is None:
            return  # no valid angle -> can't drive transitions
        if (t - self._state_entered_t) < MIN_DWELL_S:
            return  # dwell guard: let current state settle before flipping

        new_state = self._next_state(angle)
        if new_state != self.state:
            self._on_transition(self.state, new_state, arr, angle, t)
            self.state = new_state
            self._state_entered_t = t

    def _next_state(self, angle: float) -> str:
        if self.state == STANDING:
            if angle < STANDING_EXIT:
                return DESCENDING
        elif self.state == DESCENDING:
            if angle < BOTTOM_ENTER:
                return BOTTOM
            if angle > STANDING_ENTER:
                return STANDING            # aborted descent
        elif self.state == BOTTOM:
            if angle > BOTTOM_EXIT:
                return ASCENDING
        elif self.state == ASCENDING:
            if angle > STANDING_ENTER:
                return STANDING            # rep complete
            if angle < BOTTOM_ENTER:
                return BOTTOM              # double-bottom (rare)
        return self.state

    def _on_transition(self, old: str, new: str, arr: np.ndarray, angle: float, t: float):
        # Rep starts when we first leave STANDING
        if old == STANDING and new == DESCENDING:
            self._buffer = [arr]
            self._angles = [angle]
            self._rep_start_t = t
            if self.on_rep_started:
                self.on_rep_started()

        # Rep completes when we return to STANDING via ASCENDING
        if old == ASCENDING and new == STANDING:
            self._finalize_rep(t)

    def _finalize_rep(self, t_now: float):
        if not self._buffer:
            return
        self.rep_count += 1
        traj = np.stack(self._buffer, axis=0)            # (n_frames, 17, 3)
        angles_arr = np.asarray(self._angles, dtype=float)

        if np.all(np.isnan(angles_arr)):
            bottom_idx, min_angle = 0, float("nan")
        else:
            bottom_idx = int(np.nanargmin(angles_arr))
            min_angle = float(angles_arr[bottom_idx])

        duration = (t_now - self._rep_start_t) if self._rep_start_t is not None else 0.0

        rep = RepData(
            rep_number=self.rep_count,
            trajectory=traj,
            bottom_frame_idx=bottom_idx,
            min_knee_angle=min_angle,
            duration_s=duration,
        )
        self._buffer.clear()
        self._angles.clear()
        self._rep_start_t = None
        self.on_rep_complete(rep)