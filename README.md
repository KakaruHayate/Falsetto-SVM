# Falsetto-SVM

A faithful PyTorch reproduction of:

> G. J. Mysore & P. Smaragdis, *Singer-Dependent Falsetto Detection for Live
> Vocal Processing Based on Support Vector Classification*, ICASSP 2006.
> [IEEE link](https://ieeexplore.ieee.org/document/4176742)

The original dataset is not open; the upstream repo substitutes the
[CCMusic Chest-Falsetto dataset](https://www.modelscope.cn/datasets/ccmusic-database/chest_falsetto).

PyTorch is not a great fit for SVM (it has no built-in QP solver or kernel support),
so the upstream code minimises `nn.Linear + HingeLoss`, which is **not** the SVM
primal in Eq.(7) and lacks the RBF kernel of Eq.(8).

The original `train.py` is a global SVM; it does **not** implement the paper's
title attribute — *singer-dependent* — at all. See `REVIEW.md` for a full
five-point audit.

---

## A faithful reproduction is available in `fsvm/`

`fsvm/` is a complete, runnable re-implementation that fixes the five
gaps enumerated in `REVIEW.md`. Whereas the root-level `SVM.py`,
`preprocess.py`, `train.py`, etc. retain the upstream behaviour for
reference and comparison, **`fsvm/` is the script to run.**

- **Entry-point guides** — `fsvm/README.md`
- **Background paper reading & methodology** — `PYTORCH_VS_SKLEARN_SVM.md`
  (why `nn.Linear + HingeLoss` ≠ paper Eq.(7) and how to make them
  equivalent)

```text
fsvm/                # run from this directory
├── README.md
├── config.yaml      # single source of truth: hop=882, frames=25, etc.
├── utils.py
├── preprocess.py    # MFCC extraction, hop_length=882 (paper: 50 Hz)
├── singer_id.py     # Resemblyzer d-vector + Spectral Cluster → pseudo-singer-id
├── train.py         # per-singer LinearSVC + SVC(rbf), 7-fold CV
├── torch_linear_svm.py  # nn.Linear + HingeLoss + L2 + Z-score (returns to paper Eq.(7))
├── main.py          # singer-aware inference (Linear + RBF verdicts)
├── mfcc/*.npy       # 1280 MFCC tensors [13 × 25]
├── label/*.npy      # 1280 frame-level chest/falsetto labels
├── data/singer_id/  # embeddings.npy (1280×256) + singer_id.csv
└── results/falsetto/
    ├── summary.json        # sklearn LinearSVC + SVC(rbf) 7-fold CF
    ├── per_singer_results.csv
    └── torch_linear_svm/summary.json
```

## Repository layout

```
Falsetto-SVM/
├── SVM.py, preprocess.py, train.py, main.py            # upstream PyTorch sketch
├── config.yaml, data_loaders.py, solver.py, draw.py    # upstream helpers
├── data/train, data/val/                              # upstream pipeline dirs (empty)
├── mysore2006.pdf                                      # the paper
├── REVIEW.md                                           # audit: 5 gaps upstream vs. paper
├── PYTORCH_VS_SKLEARN_SVM.md                           # practitioner notes: nn.Linear ≠ QP
└── fsvm/                                               # faithful reproduction (see above)
```

## Quick start (the reproduction)

```bash
cd fsvm
python -m pip install resemblyzer webrtcvad hdbscan umap-learn pandas joblib
python preprocess.py        -c config.yaml
python singer_id.py         -c config.yaml
python train.py              -c config.yaml
python torch_linear_svm.py   -c config.yaml
python main.py               -i <clip.wav> -m results/models
```

Use whichever Python interpreter carries the dependencies above —
e.g. create a fresh conda/venv with PyTorch 2.x and the listed pip
packages. The reproduction was developed on Windows with Python 3.10
in the `diffsinger` conda env, but the scripts are pure Python.

The reproduction expects the CCMusic Chest-Falsetto wav pack to be
extracted under `cache_wavs/`; see `fsvm/README.md => Reproduction
recipe` for the exact path / loader call sequence.

## Citation

If you use this work, please cite both the original paper and (where
applicable) the upstream repo:

```bibtex
@inproceedings{mysore2006falsetto,
  title={Singer-Dependent Falsetto Detection for Live Vocal Processing Based on Support Vector Classification},
  author={Mysore, G. J. and Smaragdis, P.},
  booktitle={ICASSP}, year={2006}
}
```
