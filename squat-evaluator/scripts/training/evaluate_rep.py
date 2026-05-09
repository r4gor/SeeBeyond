"""
Sanity check: feature extraction + classifier on saved reps.

Examples:
    python scripts/evaluate_rep.py rep_0001
    python scripts/evaluate_rep.py rep_0030 --features
    python scripts/evaluate_rep.py --all

NOTE: --all prints accuracy on the *full* training set, which is the
in-sample fit. For an honest accuracy number look at the cross-validation
report from train_classifier.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.classifier import FormClassifier
from src.features import FEATURE_NAMES, extract_features, find_bottom_idx

DATA_DIR = ROOT / "data"
MANIFEST = DATA_DIR / "manifest.csv"


def _eval_one(clf, df_row, *, show_features: bool) -> tuple[str, str, str]:
    """Returns (true_label, pred, msg)."""
    npy = DATA_DIR / df_row["npy"]
    traj = np.load(npy)
    feats = extract_features(traj)
    if show_features:
        bot = find_bottom_idx(traj)
        print(f"  bottom_idx = {bot} of {traj.shape[0]}")
        print(f"  features:")
        for n in FEATURE_NAMES:
            print(f"    {n:42s} {feats[n]:+8.2f}")
    verdict, msg = clf.classify_and_explain(feats)
    return df_row["label"], verdict, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rep_id", nargs="?", help="e.g. rep_0001")
    ap.add_argument("--all", action="store_true", help="scan full manifest")
    ap.add_argument("--features", action="store_true", help="print feature vector")
    args = ap.parse_args()

    if not MANIFEST.exists():
        print(f"ERROR: {MANIFEST} not found")
        sys.exit(1)

    df = pd.read_csv(MANIFEST)
    clf = FormClassifier()

    if args.all:
        right = wrong = 0
        confused = []
        for _, row in df.iterrows():
            if not (DATA_DIR / row["npy"]).exists():
                continue
            true, pred, msg = _eval_one(clf, row, show_features=False)
            ok = "OK " if pred == true else " X "
            print(f"  {ok} {row['rep_id']}  true={true:14s} pred={pred:14s} | {msg}")
            if pred == true:
                right += 1
            else:
                wrong += 1
                confused.append((row['rep_id'], true, pred))
        total = right + wrong
        print(f"\n  in-sample accuracy: {right}/{total} ({100*right/max(total,1):.1f}%)")
        print(f"  (use train_classifier.py CV report for the honest number)")
        if confused:
            print(f"\n  misclassifications:")
            for rid, t, p in confused:
                print(f"    {rid}: {t} -> {p}")
        return

    if not args.rep_id:
        ap.error("provide a rep_id, or use --all")
    matches = df[df["rep_id"] == args.rep_id]
    if len(matches) == 0:
        print(f"rep_id {args.rep_id} not in manifest")
        sys.exit(1)
    row = matches.iloc[0]

    print(f"\n{args.rep_id}  label={row['label']}  frames={row['n_frames']}")
    true, pred, msg = _eval_one(clf, row, show_features=True)
    print(f"\n  verdict:  {pred}  (true: {true})")
    print(f"  feedback: {msg}")


if __name__ == "__main__":
    main()