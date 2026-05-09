"""
Train the form classifier from data/manifest.csv.

  * Loads every rep referenced in the manifest, extracts the 10-element
    feature vector with src.features.extract_features.
  * Fits a RandomForestClassifier (sklearn). At ~65 reps and 10 features
    this is overkill-ish, which is the point — robust, no scaler, no
    tuning, gives feature importances we can show on the slide.
  * Reports stratified 5-fold cross-validation, then refits on all reps
    and saves the production model.
  * Computes per-class feature percentiles AND a small set of
    rule-based-feedback thresholds derived from the 'good' class
    distribution (so they auto-calibrate to the trainer's body and
    movement style instead of textbook biomechanics numbers).

Run from project root:
    python scripts/train_classifier.py

Outputs:
    data/form_model.joblib     # the model bundle
    data/feature_stats.json    # per-class percentiles + thresholds
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import FEATURE_NAMES, extract_features, features_to_vector

DATA_DIR = ROOT / "data"
MANIFEST = DATA_DIR / "manifest.csv"
MODEL_PATH = DATA_DIR / "form_model.joblib"
STATS_PATH = DATA_DIR / "feature_stats.json"


def load_manifest_features():
    df = pd.read_csv(MANIFEST)
    print(f"[train] {len(df)} reps in manifest")
    print(f"[train] label counts:")
    for lbl, n in df["label"].value_counts().items():
        print(f"           {lbl:14s} {n}")
    print()

    X, y, ids, skipped = [], [], [], []
    for _, row in df.iterrows():
        npy_path = DATA_DIR / row["npy"]
        if not npy_path.exists():
            skipped.append((row["rep_id"], "npy missing"))
            continue
        traj = np.load(npy_path)
        if traj.ndim != 3 or traj.shape[1:] != (17, 3):
            skipped.append((row["rep_id"], f"bad shape {traj.shape}"))
            continue
        feats = extract_features(traj)
        vec = features_to_vector(feats)
        if not np.isfinite(vec).all():
            bad = [FEATURE_NAMES[i] for i in np.where(~np.isfinite(vec))[0]]
            skipped.append((row["rep_id"], f"non-finite features: {bad}"))
            continue
        X.append(vec)
        y.append(row["label"])
        ids.append(row["rep_id"])

    if skipped:
        print(f"[train] skipped {len(skipped)} reps:")
        for rid, reason in skipped:
            print(f"           {rid}: {reason}")
        print()

    return np.stack(X), np.array(y), ids


def cross_validate(X: np.ndarray, y: np.ndarray) -> None:
    classes = sorted(set(y))
    n_classes = len(classes)
    # fewest-class bound: can't have more folds than the smallest class
    min_class_n = int(min(np.sum(y == c) for c in classes))
    n_splits = min(5, min_class_n)
    if n_splits < 2:
        print(f"[train] only {min_class_n} samples in smallest class — skipping CV")
        return

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf = _make_clf()
    y_pred = cross_val_predict(clf, X, y, cv=skf)

    print(f"[train] stratified {n_splits}-fold CV report:")
    print(classification_report(y, y_pred, digits=3, zero_division=0))
    print(f"        confusion (rows=true, cols=pred):")
    cm = confusion_matrix(y, y_pred, labels=classes)
    header = "             " + "  ".join(f"{c[:10]:>10s}" for c in classes)
    print(header)
    for i, c in enumerate(classes):
        row = "  ".join(f"{n:>10d}" for n in cm[i])
        print(f"  {c[:10]:>10s} {row}")
    print()


def _make_clf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def per_class_stats(X: np.ndarray, y: np.ndarray) -> dict:
    classes = sorted(set(y))
    out = {}
    for cls in classes:
        Xc = X[y == cls]
        out[cls] = {
            "n": int(len(Xc)),
            "mean": {n: float(Xc[:, i].mean()) for i, n in enumerate(FEATURE_NAMES)},
            "std":  {n: float(Xc[:, i].std())  for i, n in enumerate(FEATURE_NAMES)},
            "p10":  {n: float(np.percentile(Xc[:, i], 10)) for i, n in enumerate(FEATURE_NAMES)},
            "p50":  {n: float(np.percentile(Xc[:, i], 50)) for i, n in enumerate(FEATURE_NAMES)},
            "p90":  {n: float(np.percentile(Xc[:, i], 90)) for i, n in enumerate(FEATURE_NAMES)},
        }
    return out


def derive_good_thresholds(X: np.ndarray, y: np.ndarray, fallback: dict) -> dict:
    """
    Derive feedback-trigger thresholds from the 'good' class distribution.
    A rule fires when a rep is outside the trainer's good envelope.
    If 'good' is absent, fall back to the hard-coded defaults.
    """
    if "good" not in y:
        return dict(fallback)
    Xg = X[y == "good"]
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    p10 = lambda f: float(np.percentile(Xg[:, idx[f]], 10))
    p90 = lambda f: float(np.percentile(Xg[:, idx[f]], 90))
    # A bit of headroom past the 10/90 percentile so borderline-good reps
    # don't flag. ~1 std worth of slack.
    sigma = lambda f: float(Xg[:, idx[f]].std())
    return {
        # below this many cm the hip is "above parallel"
        "hip_minus_knee_y_warning_cm": p10("hip_minus_knee_y_at_bottom") - sigma("hip_minus_knee_y_at_bottom"),
        # above this many degrees the trunk is "leaning forward"
        "trunk_angle_warning_deg": p90("trunk_angle_at_bottom") + sigma("trunk_angle_at_bottom"),
        # below this ratio the knees are "caving"
        "knee_to_hip_ratio_low_warning": p10("knee_to_hip_width_ratio_at_bottom") - 0.5 * sigma("knee_to_hip_width_ratio_at_bottom"),
        # above this many cm the heels are "lifting"
        "ankle_drift_warning_cm": p90("ankle_y_drift_max") + sigma("ankle_y_drift_max"),
        # above this many degrees of L/R asymmetry the rep is uneven
        "lr_knee_delta_warning_deg": p90("lr_knee_angle_delta_at_bottom") + sigma("lr_knee_angle_delta_at_bottom"),
    }


# Conservative fallbacks if 'good' class is missing / tiny.
FALLBACK_THRESHOLDS = {
    "hip_minus_knee_y_warning_cm":   -8.0,
    "trunk_angle_warning_deg":        45.0,
    "knee_to_hip_ratio_low_warning":   0.70,
    "ankle_drift_warning_cm":          5.0,
    "lr_knee_delta_warning_deg":      15.0,
}


def main() -> None:
    if not MANIFEST.exists():
        print(f"ERROR: {MANIFEST} not found")
        sys.exit(1)

    X, y, ids = load_manifest_features()
    if len(X) == 0:
        print("ERROR: no usable reps after feature extraction")
        sys.exit(1)
    print(f"[train] using {len(X)} reps, {X.shape[1]} features\n")

    cross_validate(X, y)

    clf = _make_clf()
    clf.fit(X, y)

    print("[train] feature importances (final model, fit on all reps):")
    for name, imp in sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1]):
        bar = "#" * int(round(imp * 40))
        print(f"  {imp:.3f}  {name:42s} {bar}")
    print()

    thresholds = derive_good_thresholds(X, y, fallback=FALLBACK_THRESHOLDS)
    print("[train] feedback thresholds (auto-calibrated from 'good' class):")
    for k, v in thresholds.items():
        print(f"  {k:40s} {v:+.2f}")
    print()

    stats = {
        "feature_names": FEATURE_NAMES,
        "classes": sorted(set(y)),
        "n_train": int(len(X)),
        "per_class": per_class_stats(X, y),
        "good_baseline_thresholds": thresholds,
    }
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[train] wrote {STATS_PATH}")

    bundle = {
        "model": clf,
        "feature_names": FEATURE_NAMES,
        "classes": list(clf.classes_),
        "thresholds": thresholds,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"[train] wrote {MODEL_PATH}")


if __name__ == "__main__":
    main()