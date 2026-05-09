"""
Phase 4 feature extraction.

Input  : 3D trajectory of one rep, shape (T, 17, 3) float32, units = cm.
         COCO keypoint order. NaN entries mark invalid joints.
         Coordinate frame: +x right, +y down, +z forward (camera frame).
Output : a 10-element feature dict (see FEATURE_NAMES) suitable for the
         classifier and for rule-based feedback construction.

Design notes
------------
* Bottom frame = argmax of mean(L_HIP.y, R_HIP.y) — y points down, so the
  deepest position has the largest hip y.
* Trunk lean is unsigned (angle from world-up); captures forward and side
  lean as a single magnitude. Cleaner than projecting onto a sagittal plane
  whose orientation depends on the athlete's facing direction.
* Knee-over-toe is the |knee - ankle| distance in the x-z (horizontal)
  plane. Direction-agnostic, so it works whether the camera is dead-side
  or 3/4-angle.
* Heels-lifting proxy: COCO-17 has no heel keypoint, only ankle. We track
  the rise of mean ankle.y vs. the standing baseline (median over the
  first 10% of frames). If the foot rolls onto the ball, the lateral
  malleolus marker rises.
* All metrics are NaN-safe; downstream code should still check for NaNs
  in the returned vector before feeding the model.
"""
from __future__ import annotations

import numpy as np

# COCO 17 keypoint indices we actually use
NOSE = 0
L_SHO, R_SHO = 5, 6
L_HIP, R_HIP = 11, 12
L_KNE, R_KNE = 13, 14
L_ANK, R_ANK = 15, 16

FEATURE_NAMES = [
    "min_knee_angle_avg",                # deg, min over rep of (L+R)/2
    "knee_angle_at_bottom",              # deg, at bottom frame, avg L+R
    "hip_minus_knee_y_at_bottom",        # cm, +ve = hip below knee = parallel
    "trunk_angle_at_bottom",             # deg from vertical
    "trunk_angle_max",                   # deg, peak across rep
    "knee_to_ankle_horiz_at_bottom",     # cm, |knee-ankle| in xz plane, avg L+R
    "knee_to_hip_width_ratio_at_bottom", # unitless, valgus indicator
    "ankle_y_drift_max",                 # cm, peak rise of ankle y above baseline
    "lr_knee_angle_delta_at_bottom",     # deg, |L - R| knee angles at bottom
    "descent_to_ascent_ratio",           # unitless, descent_frames / ascent_frames
]
N_FEATURES = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# small NaN-safe geometry helpers
# ---------------------------------------------------------------------------

def _vec_angle_deg(v1, v2) -> float:
    """Unsigned angle in degrees between two 3D vectors. NaN-safe."""
    if v1 is None or v2 is None:
        return float("nan")
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if not np.isfinite(n1) or not np.isfinite(n2) or n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    cos = float(np.dot(v1, v2)) / (n1 * n2)
    cos = max(-1.0, min(1.0, cos))
    return float(np.degrees(np.arccos(cos)))


def _knee_angle(hip, knee, ankle) -> float:
    """Hip-knee-ankle interior angle. ~180° fully extended, smaller when bent."""
    if (np.isnan(hip).any() or np.isnan(knee).any() or np.isnan(ankle).any()):
        return float("nan")
    return _vec_angle_deg(hip - knee, ankle - knee)


def _midpoint(a, b):
    """Average two 3D points; if one is NaN, return the other."""
    a_bad = np.isnan(a).any()
    b_bad = np.isnan(b).any()
    if a_bad and b_bad:
        return np.full(3, np.nan, dtype=np.float32)
    if a_bad:
        return b.astype(np.float32)
    if b_bad:
        return a.astype(np.float32)
    return ((a + b) / 2.0).astype(np.float32)


def _horiz_dist_xz(a, b) -> float:
    """Distance in the x-z plane (horizontal, ignores vertical)."""
    if np.isnan(a).any() or np.isnan(b).any():
        return float("nan")
    return float(np.hypot(a[0] - b[0], a[2] - b[2]))


def _euclid3d(a, b) -> float:
    if np.isnan(a).any() or np.isnan(b).any():
        return float("nan")
    return float(np.linalg.norm(a - b))


# ---------------------------------------------------------------------------
# bottom-frame detection + per-frame trajectories
# ---------------------------------------------------------------------------

def find_bottom_idx(traj: np.ndarray) -> int:
    """Bottom = max mean hip y. Robust to NaN; falls back to mid-rep."""
    hip_y = np.nanmean(traj[:, [L_HIP, R_HIP], 1], axis=1)
    if np.isnan(hip_y).all():
        return len(traj) // 2
    return int(np.nanargmax(hip_y))


def _knee_angles_per_frame(traj: np.ndarray) -> np.ndarray:
    """(T, 2) — left, right knee angle at each frame."""
    T = traj.shape[0]
    out = np.full((T, 2), np.nan, dtype=np.float32)
    for t in range(T):
        out[t, 0] = _knee_angle(traj[t, L_HIP], traj[t, L_KNE], traj[t, L_ANK])
        out[t, 1] = _knee_angle(traj[t, R_HIP], traj[t, R_KNE], traj[t, R_ANK])
    return out


def _trunk_angles_per_frame(traj: np.ndarray) -> np.ndarray:
    """(T,) — trunk-from-vertical angle at each frame, deg."""
    T = traj.shape[0]
    world_up = np.array([0.0, -1.0, 0.0], dtype=np.float32)  # camera +y is down
    out = np.full(T, np.nan, dtype=np.float32)
    for t in range(T):
        sho_mid = _midpoint(traj[t, L_SHO], traj[t, R_SHO])
        hip_mid = _midpoint(traj[t, L_HIP], traj[t, R_HIP])
        if np.isnan(sho_mid).any() or np.isnan(hip_mid).any():
            continue
        trunk = sho_mid - hip_mid  # points up-ish (negative y)
        out[t] = _vec_angle_deg(trunk, world_up)
    return out


# ---------------------------------------------------------------------------
# main public entry point
# ---------------------------------------------------------------------------

def extract_features(traj: np.ndarray, bottom_idx: int | None = None) -> dict:
    """
    traj: (T, 17, 3) float, cm. NaN-safe.
    Returns a dict with every key in FEATURE_NAMES.
    """
    assert traj.ndim == 3 and traj.shape[1:] == (17, 3), \
        f"expected (T, 17, 3), got {traj.shape}"
    T = traj.shape[0]
    if bottom_idx is None:
        bottom_idx = find_bottom_idx(traj)
    b = max(0, min(bottom_idx, T - 1))

    knee_angles = _knee_angles_per_frame(traj)         # (T, 2)
    knee_angle_avg = np.nanmean(knee_angles, axis=1)   # (T,)
    trunk_angles = _trunk_angles_per_frame(traj)       # (T,)

    # 1. min_knee_angle_avg
    if np.any(np.isfinite(knee_angle_avg)):
        min_ka_avg = float(np.nanmin(knee_angle_avg))
    else:
        min_ka_avg = float("nan")

    # 2. knee_angle_at_bottom
    ka_bottom = float(knee_angle_avg[b]) if np.isfinite(knee_angle_avg[b]) else min_ka_avg

    # 3. hip_minus_knee_y_at_bottom (cm). +ve = hip lower than knee = parallel achieved.
    #    NB the COCO hip marker sits above the joint center, so good reps give ~ -3 to -6
    #    on this metric; thresholds must be calibrated against the trainer's good distribution.
    hip_mid_b = _midpoint(traj[b, L_HIP], traj[b, R_HIP])
    knee_y_b = np.nanmean([traj[b, L_KNE, 1], traj[b, R_KNE, 1]])
    if np.isnan(hip_mid_b).any() or not np.isfinite(knee_y_b):
        hip_minus_knee_y = float("nan")
    else:
        hip_minus_knee_y = float(hip_mid_b[1] - knee_y_b)

    # 4 & 5. trunk angles
    trunk_at_b = float(trunk_angles[b]) if np.isfinite(trunk_angles[b]) else float("nan")
    if np.any(np.isfinite(trunk_angles)):
        trunk_max = float(np.nanmax(trunk_angles))
    else:
        trunk_max = float("nan")

    # 6. knee-to-ankle horizontal distance at bottom (xz plane), avg L+R
    kta_l = _horiz_dist_xz(traj[b, L_KNE], traj[b, L_ANK])
    kta_r = _horiz_dist_xz(traj[b, R_KNE], traj[b, R_ANK])
    pair = [v for v in (kta_l, kta_r) if np.isfinite(v)]
    knee_to_ankle = float(np.mean(pair)) if pair else float("nan")

    # 7. knee-separation : hip-width ratio at bottom (valgus indicator, <1 = caving).
    #    Camera-x only: z on the partially-occluded far-side joint is noisy and
    #    poisons 3D distances. In any reasonable camera placement (side-on or
    #    3/4-angle), the athlete's left-right axis has a meaningful x-component.
    def _abs_dx(p1, p2):
        if np.isnan(p1[0]) or np.isnan(p2[0]):
            return float("nan")
        return float(abs(p1[0] - p2[0]))
    knee_sep_x = _abs_dx(traj[b, L_KNE], traj[b, R_KNE])
    hip_sep_x  = _abs_dx(traj[b, L_HIP], traj[b, R_HIP])
    if np.isfinite(knee_sep_x) and np.isfinite(hip_sep_x) and hip_sep_x > 1e-3:
        knee_hip_ratio = float(knee_sep_x / hip_sep_x)
    else:
        knee_hip_ratio = float("nan")

    # 8. ankle_y_drift_max — peak rise of mean ankle y above the standing baseline.
    #    +ve cm = ankle has lifted (foot left the ground / heel lifted).
    ank_y = np.nanmean(traj[:, [L_ANK, R_ANK], 1], axis=1)
    valid_ank = np.isfinite(ank_y)
    if valid_ank.sum() >= 5:
        n_baseline = max(3, int(0.10 * valid_ank.sum()))
        baseline = float(np.median(ank_y[valid_ank][:n_baseline]))
        # smaller y = higher in space (y points down); rise = baseline - ank_y
        rise = baseline - ank_y
        ankle_drift = float(np.nanmax(rise))
    else:
        ankle_drift = float("nan")

    # 9. left/right knee angle delta at bottom
    kl, kr = knee_angles[b, 0], knee_angles[b, 1]
    if np.isfinite(kl) and np.isfinite(kr):
        lr_delta = float(abs(kl - kr))
    else:
        lr_delta = float("nan")

    # 10. descent vs ascent timing
    descent_n = b
    ascent_n = T - b - 1
    if ascent_n > 0:
        ratio = float(descent_n / ascent_n)
    else:
        ratio = float("nan")

    return {
        "min_knee_angle_avg":              min_ka_avg,
        "knee_angle_at_bottom":            ka_bottom,
        "hip_minus_knee_y_at_bottom":      hip_minus_knee_y,
        "trunk_angle_at_bottom":           trunk_at_b,
        "trunk_angle_max":                 trunk_max,
        "knee_to_ankle_horiz_at_bottom":   knee_to_ankle,
        "knee_to_hip_width_ratio_at_bottom": knee_hip_ratio,
        "ankle_y_drift_max":               ankle_drift,
        "lr_knee_angle_delta_at_bottom":   lr_delta,
        "descent_to_ascent_ratio":         ratio,
    }


def features_to_vector(features: dict) -> np.ndarray:
    """dict -> ndarray in FEATURE_NAMES order, dtype float32."""
    return np.array([features[k] for k in FEATURE_NAMES], dtype=np.float32)