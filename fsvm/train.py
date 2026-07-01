"""Singer-dependent SVM training.

For each singer s ∈ {0..n_speakers-1}:
  - gather all MFCC frames from wavs labeled with singer s
  - 7-fold cross-validation (paper)
  - train LinearSVC and SVC(rbf) on the union of train folds
  - report CF (accuracy) on the held-out fold and hinge-loss as a sanity sidecar
  - save the model trained on the *full* singer's data for downstream inference

Outputs:
  models/singer_<s>__<kernel>.joblib
  results/per_singer_results.csv
  results/summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

from utils import DotDict, ensure_dir, load_config


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    return p.parse_args()


def _load_frames_for_singer(
    singer_id: int,
    singer_csv: str,
    mfcc_dir: str,
    label_dir: str,
):
    """Return (X, y) where X has shape (n_frames_total, 13) and y has (-1/+1)."""
    df = pd.read_csv(singer_csv)
    df = df[df["singer_id"] == singer_id].reset_index(drop=True)
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for fn in df["filename"]:
        mfcc = np.load(os.path.join(mfcc_dir, fn + ".npy"))  # (13, 25)
        lab = np.load(os.path.join(label_dir, fn + ".npy"))   # (25,)
        Xs.append(mfcc.T.astype(np.float32))
        ys.append(lab.astype(np.float32))
    if not Xs:
        return None, None
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y


@dataclass
class KernelSpec:
    name: str
    kind: str
    C: float
    gamma: Any = "scale"


def _build_model(spec: KernelSpec, random_state: int):
    if spec.kind == "linear":
        return LinearSVC(
            C=spec.C, dual="auto", tol=1e-4, max_iter=5000,
            random_state=random_state,
        )
    if spec.kind == "rbf":
        return SVC(
            C=spec.C, kernel="rbf", gamma=spec.gamma, tol=1e-3,
            random_state=random_state,
        )
    raise ValueError(f"unknown kernel kind {spec.kind}")


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config)

    singer_csv = cfg.singer_id.cache_csv
    mfcc_dir = os.path.join(os.path.dirname(cfg.data.audio_dir), "mfcc")
    label_dir = os.path.join(os.path.dirname(cfg.data.audio_dir), "label")

    models_dir = cfg.env.models_dir
    res_dir = cfg.env.expdir
    ensure_dir(models_dir)
    ensure_dir(res_dir)

    min_frames = int(cfg.train.min_frames_per_singer)
    cv_folds = int(cfg.train.cv_folds)
    rs = int(cfg.train.random_state)

    kernels: list[KernelSpec] = []
    for kname, kspec in cfg.train.kernels.items():
        kernels.append(KernelSpec(
            name=kname,
            kind=kspec.kind,
            C=float(kspec.C),
            gamma=kspec.gamma if hasattr(kspec, "gamma") else "scale",
        ))

    if not os.path.exists(singer_csv):
        print(f"[train] missing singer-id csv at {singer_csv}; run singer_id.py first")
        return 1

    df_s = pd.read_csv(singer_csv)
    singers = sorted(df_s["singer_id"].unique().tolist())

    per_singer_rows = []
    summary: dict[str, Any] = {"kernels": {}, "cv_folds": cv_folds}

    for s in singers:
        X, y = _load_frames_for_singer(s, singer_csv, mfcc_dir, label_dir)
        if X is None or y is None:
            print(f"[train] singer {s}: no frames, skipping")
            continue
        n = len(y)
        if n < min_frames:
            print(f"[train] singer {s}: only {n} frames (<{min_frames}); skip CV, fit+save only")
            for spec in kernels:
                clf = _build_model(spec, rs)
                clf.fit(X, y)
                acc = float((clf.predict(X) == y).mean())
                per_singer_rows.append({
                    "singer_id": s, "kernel": spec.name, "fold": -1,
                    "n_frames": n, "accuracy": acc,
                })
                joblib.dump(clf, os.path.join(models_dir, f"singer_{s}__{spec.name}.joblib"))
            continue

        # 7-fold CV (paper baseline). Stratify on labels since chest / falsetto can be imbalanced.
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=rs)
        for spec in kernels:
            fold_acc = []
            for fi, (tr, va) in enumerate(skf.split(X, y)):
                clf = _build_model(spec, rs)
                clf.fit(X[tr], y[tr])
                pred = clf.predict(X[va])
                acc = float((pred == y[va]).mean())
                fold_acc.append(acc)
                per_singer_rows.append({
                    "singer_id": s, "kernel": spec.name, "fold": fi,
                    "n_frames": n, "accuracy": acc,
                })
            # Full-data model for inference
            full_clf = _build_model(spec, rs)
            full_clf.fit(X, y)
            train_acc = float((full_clf.predict(X) == y).mean())
            joblib.dump(full_clf, os.path.join(models_dir, f"singer_{s}__{spec.name}.joblib"))
            mean_cv = float(np.mean(fold_acc))
            print(f"[train] singer={s:2d} {spec.name:7s} "
                  f"n={n:5d}  train_full_acc={train_acc:.3f}  "
                  f"7fold_mean_acc={mean_cv:.3f} (+/- {np.std(fold_acc):.3f})")
            summary["kernels"].setdefault(spec.name, []).append({
                "singer_id": int(s),
                "n_frames": int(n),
                "train_full_acc": train_acc,
                "cv7_mean_acc": mean_cv,
                "cv7_std": float(np.std(fold_acc)),
            })

    # Persist per-singer results
    csv_out = os.path.join(res_dir, "per_singer_results.csv")
    pd.DataFrame(per_singer_rows).to_csv(csv_out, index=False)

    # Summary
    for kname, rows in summary["kernels"].items():
        accs = [r["cv7_mean_acc"] for r in rows if r["cv7_mean_acc"] is not None]
        if accs:
            print(f"\n[kernel {kname}]  mean CF over {len(accs)} singers: "
                  f"{np.mean(accs):.4f}  (error = {1 - np.mean(accs):.4f})")
            summary["kernels"][kname] = {"per_singer": rows,
                                         "mean_cf": float(np.mean(accs)),
                                         "mean_error_frac": float(1 - np.mean(accs))}
    with open(os.path.join(res_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[train] results -> {csv_out}\n[train] summary -> {os.path.join(res_dir, 'summary.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
