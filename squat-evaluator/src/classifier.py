"""
FormClassifier: model-predicted verdict + rule-based specific feedback.

The pitch is "specificity" — the Core2 line shouldn't read 'shallow', it
should read 'Hip stopped 4 cm above parallel'. So we split the work:

  * The trained classifier answers WHICH class the rep belongs to
    (good / shallow / forward_lean / heels_lifting).
  * A small set of rule-based metrics inspects the same feature dict and
    builds a concrete cm/deg feedback string. Thresholds are loaded from
    the model bundle (auto-calibrated from the trainer's good reps at
    training time).

Top-level convenience that matches the spec exactly:

    from src.classifier import classify_and_explain
    verdict, msg = classify_and_explain(rep_features)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from .features import FEATURE_NAMES, features_to_vector

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "form_model.joblib"


# Last-resort thresholds if the bundle doesn't carry any.
STATIC_FEEDBACK_PHRASES = [
    "Solid rep — depth and posture clean",
    "Solid rep — chest stayed up",
    "Solid rep — clean form",
    "Go deeper — hip below knee",
    "Chest up — keep trunk vertical",
    "Drive knees out",
    "Knees collapsed inward — push them out",
    "Sit back into your heels",
]

DEFAULT_THRESHOLDS = {
    "hip_minus_knee_y_warning_cm":   -8.0,
    "trunk_angle_warning_deg":        45.0,
    "knee_to_hip_ratio_low_warning":   0.70,
    "ankle_drift_warning_cm":          5.0,
    "lr_knee_delta_warning_deg":      15.0,
}


def _isnan(x) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


class FormClassifier:
    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.classes: list[str] = list(bundle.get("classes", self.model.classes_))
        self.thresholds: dict = {**DEFAULT_THRESHOLDS, **bundle.get("thresholds", {})}

    # ----------------------------------------------------------- prediction

    def predict(self, features: dict) -> str:
        x = features_to_vector(features).reshape(1, -1)
        # RandomForest tolerates 0s but not NaNs. Replace NaN with 0; this is
        # rare since features.py is NaN-safe, but defensive.
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return str(self.model.predict(x)[0])

    def predict_proba(self, features: dict) -> dict[str, float]:
        x = features_to_vector(features).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        probs = self.model.predict_proba(x)[0]
        return {str(c): float(p) for c, p in zip(self.model.classes_, probs)}

    # ---------------------------------------------------- main entry point

    def classify_and_explain(self, features: dict) -> tuple[str, str]:
        """
        Returns (verdict, feedback_string) suitable for:
            coach.push(reps=N, verdict=verdict, feedback=msg, knee_angle=...)

        verdict          one of self.classes  ('good' | 'shallow' | ...)
        feedback_string  short, specific, fits on a Core2 line
        """
        verdict = self.predict(features)
        msg = self._build_feedback(verdict, features)
        return verdict, msg

    # --------------------------------------------- rule-based feedback layer

    def _build_feedback(self, verdict: str, f: dict) -> str:
        deviations = self._collect_deviations(f)

        # Verdict says "good" — usually no deviations, but if any borderline
        # rule still fired, mention it as a coaching cue.
        if verdict == "good":
            if not deviations:
                return self._praise(f)
            kind, msg = deviations[0]
            return f"Solid rep — watch: {msg.lower()}"

        # Verdict says something's off. Prefer a deviation matching the
        # predicted class so the cm/deg number echoes the classifier's call.
        # If no rule fired for the predicted class, use a class-specific
        # generic line — never a non-matching specific (it would read as
        # "shallow ... 'Trunk leaned 38° forward'").
        for kind, msg in deviations:
            if kind == verdict:
                return msg
        return self._generic_for_verdict(verdict)

    def _collect_deviations(self, f: dict) -> list[tuple[str, str]]:
        """List of (verdict_kind, message), strongest first."""
        ranked: list[tuple[str, float, str]] = []
        t = self.thresholds

        # SHALLOW — hip didn't drop enough
        diff = f.get("hip_minus_knee_y_at_bottom", float("nan"))
        if not _isnan(diff) and diff < t["hip_minus_knee_y_warning_cm"]:
            severity = t["hip_minus_knee_y_warning_cm"] - diff   # positive
            cm_above = max(0.0, -diff)  # how far hip stayed above knee
            ranked.append((
                "shallow", severity,
                f"Hip stopped {cm_above:.0f} cm above parallel",
            ))

        # FORWARD LEAN — trunk too far from vertical at bottom
        trunk = f.get("trunk_angle_at_bottom", float("nan"))
        if not _isnan(trunk) and trunk > t["trunk_angle_warning_deg"]:
            severity = trunk - t["trunk_angle_warning_deg"]
            ranked.append((
                "forward_lean", severity,
                f"Trunk leaned {trunk:.0f}° forward",
            ))

        # HEELS LIFTING — ankle marker rose
        drift = f.get("ankle_y_drift_max", float("nan"))
        if not _isnan(drift) and drift > t["ankle_drift_warning_cm"]:
            severity = drift - t["ankle_drift_warning_cm"]
            ranked.append((
                "heels_lifting", severity,
                f"Heels lifted ~{drift:.0f} cm — drive through midfoot",
            ))

        # KNEES CAVING — knee separation collapsed vs hip width
        ratio = f.get("knee_to_hip_width_ratio_at_bottom", float("nan"))
        if not _isnan(ratio) and ratio < t["knee_to_hip_ratio_low_warning"]:
            severity = (t["knee_to_hip_ratio_low_warning"] - ratio) * 100.0
            ranked.append((
                "knees_caving", severity,
                "Knees collapsed inward — push them out",
            ))

        # ASYMMETRIC — left/right knee angle mismatch at bottom
        lr = f.get("lr_knee_angle_delta_at_bottom", float("nan"))
        if not _isnan(lr) and lr > t["lr_knee_delta_warning_deg"]:
            severity = lr - t["lr_knee_delta_warning_deg"]
            ranked.append((
                "asymmetric", severity,
                f"L/R knees differ by {lr:.0f}° — even the load",
            ))

        ranked.sort(key=lambda x: -x[1])
        return [(kind, msg) for kind, _sev, msg in ranked]

    def _praise(self, f: dict) -> str:
        """Pick a positive line keyed on which aspect of the rep was strongest."""
        diff = f.get("hip_minus_knee_y_at_bottom", float("nan"))
        trunk = f.get("trunk_angle_at_bottom", float("nan"))
        # If both depth and trunk are clean, lead with depth (most common cue).
        if not _isnan(diff) and diff > -3.0:
            return "Solid rep — depth and posture clean"
        if not _isnan(trunk) and trunk < 25.0:
            return "Solid rep — chest stayed up"
        return "Solid rep — clean form"

    def _generic_for_verdict(self, verdict: str) -> str:
        return {
            "shallow":       "Go deeper — hip below knee",
            "forward_lean":  "Chest up — keep trunk vertical",
            "heels_lifting": "",
            "knees_caving":  "Drive knees out",
        }.get(verdict, f"Form needs work: {verdict.replace('_', ' ')}")


# ---------------------------------------------------------------------------
# Top-level convenience matching the spec API exactly.
# ---------------------------------------------------------------------------

_SINGLETON: Optional[FormClassifier] = None


def classify_and_explain(features: dict) -> tuple[str, str]:
    """
    Module-level entry point. Lazily instantiates a FormClassifier on first
    call, then re-uses it. Mirrors the spec:

        verdict, msg = classify_and_explain(rep_features)
    """
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = FormClassifier()
    return _SINGLETON.classify_and_explain(features)