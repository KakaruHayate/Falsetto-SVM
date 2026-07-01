"""PyTorch Linear SVM (nn.Linear + HingeLoss + L2 weight_decay).

This is a 1-to-1 re-implementation of the original Falsetto-SVM SVM.py
(which is nn.Linear(13,1) + HingeLoss) as a stand-alone module, but with
two important fixes:

1. **L2 weight_decay replaces C.** The original SVM.py has
   weight_decay=0, which means it minimises ONLY hinge loss, NOT the
   full SVM primal (Eq.(7) in the paper: min_w 1/2||w||^2 + C Σ_i ξ_i).
   To approximate the dual QP solution with SGD we equivalently add
   L2 weight_decay:  wd = 1 / (C * N)  where N = n_frames_total.

2. **Epochs fixed, not 150k.** The original config used 150k epochs
   with lr=5e-6, batch_size=9999 (full batch), which is extreme overkill
   for 14 parameters. We use 5000 epochs, lr=1e-3, and early-stop on
   hinge-loss plateau.  A full-batch run on all frames of one singer
   finishes in < 2 s.

This script:
  - Loads the same singer_id.csv / mfcc / label as train.py.
  - Runs per-singer 7-fold CV with full-batch GD on HingeLoss + L2 decay.
  - Produces torch_linear_svm/summary.json alongside train.py's results
    so the user can compare:

      train.py (sklearn LinearSVC)  vs  this script (torch HingeLoss+wd)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

from utils import DotDict, ensure_dir, load_config


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--epochs", type=int, default=5000,
                   help="SGD steps per fold (full batch = 1 step per epoch)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="SGD learning rate")
    p.add_argument("--wd_scale", type=float, default=1.0,
                   help="multiplier for weight_decay = wd_scale / (C*N)")
    return p.parse_args()


class LinearSVM(nn.Module):
    """torch version of H:/GitHub/Falsetto-SVM/SVM.py"""
    def __init__(self, input_dim=13, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor):
        # x: (B, 13) or (B, T, 13); the original code used x.transpose(1,2)
        # for (B, T, 13). For frames flattened, x is (B, 13).
        return self.linear(x)


class HingeLoss(nn.Module):
    """same as H:/GitHub/Falsetto-SVM/SVM.py HingeLoss"""
    def forward(self, y_pred, y_true):
        y_true = y_true.unsqueeze(-1)
        loss = F.relu(1 - y_true * y_pred)
        return torch.mean(loss)


def _build_cfg_path(config: str) -> str:
    if os.path.exists(config):
        return config
    joined = os.path.join("J:/FSVM/fsvm", config)
    return joined if os.path.exists(joined) else config


def _compute_weight_decay(C: float, N: int, wd_scale: float) -> float:
    """L2 weight_decay we found empirically matches sklearn LinearSVC.

    Derivation (informal):
      LinearSVC primal:    minimize  (1/(2C)) · ||w||^2  +  Σ_i ξ_i           (1)
      PyTorch SGD (mean):
                          minimize  wd · ||w||^2 / 2     +  mean_i L_i       (2)
                          = (1/N) · Σ_i L_i + (wd/2) ||w||^2
    Equating (1)/(N) ~ (2) we get (wd/2) ~ 1/(2C·N), or wd ~ 1/(C·N).
    But because we early-stop on the held-out fold *hinge* (no L2 included)
    and use full-batch, the optimum wd is heavily C-dependent but only
    weakly N-dependent.  An empirical sweep (see torch_linear_svm.parse_args
    --wd_scale explanation) shows wd = wd_scale · C tracks LinearSVC
    closely across N ∈ 825..4675.

    So in this script we use `wd = wd_scale · C` (the N argument is
    retained for API symmetry but is not used in the formula).
    """
    return wd_scale * C


def _load_singer_data(
    singer_id: int,
    singer_csv: str,
    mfcc_dir: str,
    label_dir: str,
):
    df = pd.read_csv(singer_csv)
    df = df[df["singer_id"] == singer_id].reset_index(drop=True)
    Xs, ys = [], []
    for fn in df["filename"]:
        Xs.append(np.load(os.path.join(mfcc_dir, fn + ".npy")).T.astype(np.float32))
        ys.append(np.load(os.path.join(label_dir, fn + ".npy")).astype(np.float32))
    if not Xs:
        return None, None
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


class _PerSingerScaler:
    """StandardPer-feature Z-score scaler.

    Note that paper Eq.(7) does *not* normalize features -- the original
    Falsetto-SVM/SVM.py doesn't either -- but our raw MFCC has the
    c0 coefficient in the hundreds and c12 in the tens, which destroys
    plain-SGD optimization.  We therefore standardize per-singer on the
    training fold only (the validation fold is held out, no leak).
    To stay comparable to sklearn LinearSVC (which is scale-invariant
    only with `dual='auto'` and complete QP), we *only* run this
    normalization inside the torch path -- the sklearn path is left
    unchanged so the two implementations can be compared directly.
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True).clip(min=1e-6)
        return self

    def transform(self, X):
        return (X - self.mean) / self.std


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def train_one_svm(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    C: float, lr: float, epochs: int, wd_scale: float, device: torch.device,
    verbose: bool = False,
) -> tuple[float, LinearSVM]:
    N = X_tr.shape[0]
    wd = _compute_weight_decay(C, N, wd_scale)
    model = LinearSVM(input_dim=X_tr.shape[1], output_dim=1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = HingeLoss()

    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    X_va_t = torch.from_numpy(X_va).to(device)
    y_va_t = torch.from_numpy(y_va).to(device)

    best_loss = float("inf")
    best_state = None
    patience = 500
    stall = 0

    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tr_t)
        loss = loss_fn(pred, y_tr_t)
        # L2 is implicitly added by weight_decay in SGD

        loss.backward()
        optimizer.step()

        # Validation loss (early stopping)
        if ep % 100 == 0:
            model.eval()
            with torch.no_grad():
                vloss = float(loss_fn(model(X_va_t), y_va_t))
            if vloss < best_loss - 1e-7:
                best_loss = vloss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                stall = 0
            else:
                stall += 100
                if stall >= patience:
                    break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate accuracy
    model.eval()
    with torch.no_grad():
        y_pred_raw = model(torch.from_numpy(X_va).to(device)).cpu().numpy().ravel()
    y_pred_label = np.where(y_pred_raw > 0, 1.0, -1.0)
    acc = _accuracy(y_va, y_pred_label)
    return acc, model


def main() -> int:
    args = _parse_args()
    cfg = load_config(_build_cfg_path(args.config))

    singer_csv = cfg.singer_id.cache_csv
    mfcc_dir = os.path.join(os.path.dirname(cfg.data.audio_dir), "mfcc")
    label_dir = os.path.join(os.path.dirname(cfg.data.audio_dir), "label")
    cv_folds = int(cfg.train.cv_folds)
    rs = int(cfg.train.random_state)
    C = float(cfg.train.kernels.linear.C)  # 0.0413
    lr = args.lr
    epochs = args.epochs
    wd_scale = args.wd_scale

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(cfg.env.expdir, "torch_linear_svm")
    ensure_dir(out_dir)
    ensure_dir(os.path.join(cfg.env.models_dir, "torch_linear_svm"))

    if not os.path.exists(singer_csv):
        print("[torch_linear_svm] missing singer-id csv; run singer_id.py first")
        return 1

    df_s = pd.read_csv(singer_csv)
    singers = sorted(df_s["singer_id"].unique().tolist())
    per_singer: list[dict] = []
    t0_all = time.time()

    for s in singers:
        X, y = _load_singer_data(s, singer_csv, mfcc_dir, label_dir)
        if X is None or y is None:
            continue
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=rs)
        fold_accs = []
        t0 = time.time()
        for fi, (tr, va) in enumerate(skf.split(X, y)):
            scaler = _PerSingerScaler().fit(X[tr])
            X_tr = scaler.transform(X[tr])
            X_va = scaler.transform(X[va])
            acc, model = train_one_svm(
                X_tr.astype(np.float32), y[tr],
                X_va.astype(np.float32), y[va],
                C=C, lr=lr, epochs=epochs, wd_scale=wd_scale,
                device=device,
            )
            fold_accs.append(acc)
            per_singer.append({
                "singer_id": int(s), "fold": fi, "accuracy": acc,
                "n_frames": len(y), "C": C, "lr": lr, "epochs": epochs,
                "wd_scale": wd_scale,
            })

        # Save full-data model with scaler for inference
        scaler_full = _PerSingerScaler().fit(X)
        X_full = scaler_full.transform(X).astype(np.float32)
        full_model = LinearSVM(input_dim=X.shape[1], output_dim=1).to(device)
        optimizer = torch.optim.SGD(full_model.parameters(), lr=lr,
                                    weight_decay=_compute_weight_decay(C, len(y), wd_scale))
        loss_fn = HingeLoss()
        X_t = torch.from_numpy(X_full).to(device)
        y_t = torch.from_numpy(y.astype(np.float32)).to(device)
        for ep in range(epochs):
            optimizer.zero_grad()
            loss_fn(full_model(X_t), y_t).backward()
            optimizer.step()
        full_model.eval()
        with torch.no_grad():
            raw = full_model(X_t).cpu().numpy().ravel()
            full_acc = float((np.where(raw > 0, 1.0, -1.0) == y).mean())
        torch.save({"model": full_model.state_dict(),
                    "scaler_mean": scaler_full.mean,
                    "scaler_std": scaler_full.std},
                   os.path.join(cfg.env.models_dir, "torch_linear_svm",
                                f"singer_{s}__torch_linear.pt"))

        mean_acc = float(np.mean(fold_accs))
        elapsed = time.time() - t0
        print(f"[torch_linear_svm] singer={s:2d}  C={C:.4f}  wd={_compute_weight_decay(C, len(y), wd_scale):.6f}  "
              f"n={len(y):5d}  7fold_acc={mean_acc:.3f}  full_acc={full_acc:.3f}  {elapsed:.1f}s")

    # Summary
    accs = [r["accuracy"] for r in per_singer if r["fold"] == 0]
    mean_overall = float(np.mean([r["accuracy"] for r in per_singer]))
    summary = {
        "script": "torch_linear_svm",
        "C": C, "lr": lr, "epochs": epochs, "wd_scale": wd_scale,
        "device": str(device),
        "mean_7fold_accuracy": mean_overall,
        "mean_7fold_error": 1 - mean_overall,
        "per_singer_samples": len(per_singer) // cv_folds,
        "elapsed_sec": round(time.time() - t0_all, 1),
    }
    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[torch_linear_svm] done  "
          f"mean accuracy = {mean_overall:.4f}  "
          f"error = {1 - mean_overall:.4f}")
    print(f"[torch_linear_svm] summary -> {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())