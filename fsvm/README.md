# fsvm — Singer-Dependent Falsetto Detection (reproduction)

Faithful re-implementation of the classifier proposed in

> G. J. Mysore & P. Smaragdis — *Singer-Dependent Falsetto Detection for
> Live Vocal Processing Based on Support Vector Classification*, ICASSP 2006.

This directory fixes the four fidelity gaps we found in the upstream
PyTorch attempt and adds the singer-aware pipeline that the paper
actually describes.

See `../REVIEW.md` for the audit, and `../PYTORCH_VS_SKLEARN_SVM.md` for
why the upstream `nn.Linear + HingeLoss` is not by itself the paper's
Eq.(7) and what we changed to make it match.

## What this repo does instead

| Step | Script | Output |
|---|---|---|
| 1. Extract MFCC | `preprocess.py` | `mfcc/*.npy`, `label/*.npy` (13 × 25 frames, chest=-1 / falsetto=+1) |
| 2. Singer-ID via Resemblyzer | `singer_id.py` | `data/singer_id/embeddings.npy` (1280×256), `data/singer_id/singer_id.csv` |
| 3. Singer-dependent SVM | `train.py` | `results/models/singer_<id>__<kernel>.joblib`, `results/falsetto/summary.json`, `results/falsetto/per_singer_results.csv` |
| 4. Singer-aware inference | `main.py` | clip-level verdict per kernel + cross-kernel consensus |
| 5. **PyTorch linear-SVM reference** | `torch_linear_svm.py` | `results/falsetto/torch_linear_svm/summary.json` (same 7-fold CV with `nn.Linear + HingeLoss`) |

Pseudocode for the four scripts is documented inline.

## Pseudo-singer identity

The original paper used a private recording of 13 singers each performing
the same piece, once in chest voice and once in falsetto. The dataset
used here is the **CCMusic chest_falsetto** set (1,280 single-singer
short clips from 12 singers, per the dataset README). However, the
published filenames encode only `NNNN_<gender>_<method>` - **they do
not carry singer identity.**

To recover a "trained per singer" pipeline in the absence of ground-truth
labels we cluster Resemblyzer GE2E d-vectors with the ColorSplitter /
3D-Speaker Spectral Cluster, forcing the cluster count to the dataset's
documented count of 12. This is the most that the published meta-data
allows. Talk results should be read against this caveat: the inferred
clusters are not guaranteed to be 12 distinct real singers (one or two
clusters can collapse two real singletons).

See `singer_id.py` for the call sequence and `data/singer_id/` for the
emitted files.

## Kernel & hyperparameters

- MFCC: `n_mfcc=13`, `n_fft=256`, `hop=882`, `pre_emph=0.97`, `sr=44100`
  (Section 4 of the paper). Single-frame label is `+1` for falsetto and
  `-1` for modal voice (Section 5).
- Linear SVM via `sklearn.svm.LinearSVC(C=0.0413)` — matches paper Eq.(7)
  optimum.
- Gaussian SVM via `sklearn.svm.SVC(kernel='rbf', C=2.057, gamma='scale')`
  — matches paper Eq.(8) optimum.
- Per-singer 7-fold CV (matches paper Section 6).

## Reproduction recipe

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
e.g. a fresh conda env with PyTorch 2.x, librosa, scikit-learn, etc.
The code was developed with Python 3.10 on conda env `diffsinger`, and is
platform-independent.

The dataset: download the [CCMusic Chest-Falsetto
dataset](https://www.modelscope.cn/datasets/ccmusic-database/chest_falsetto),
extract `audio.zip` into `cache_wavs/` so that `ls cache_wavs/*.wav` shows
the 1280 wav files (`0001_m_chest.wav`, ... `1280_f_falsetto.wav`).

## Results on CCMusic (this work)

> Cluster-level report, mean over 12 pseudo-singers, 7-fold CV.
> Paper: linear 4.8% err / RBF 1.5% err.

| kernel | mean accuracy | mean error |
|---|---|---|
| **sklearn** `LinearSVC` (C=0.0413, paper Eq.(7))             | 0.864 | 13.6% |
| **PyTorch** `nn.Linear + HingeLoss` + L2 + per-singer Z-score   | **0.839** | 16.1% |
| **sklearn** `SVC(rbf, C=2.057)` (paper Eq.(8))                 | 0.846 | 15.4% |

Things to note when comparing to the paper's headline numbers:

- The paper uses private recordings of one well-rehearsed singer per
  setup; we operate on a public re-recording of many singers on
  different material — vocals differ more across singers than across
  registers, which inevitably raises the chorus to chest-vs-falsetto
  confusion margin.
- Pseudo-label noise from the unsupervised clustering introduces
  per-singer misassignments (some clusters mix multiple real singers),
  which biases the per-singer CV error upward.
- The RBF kernel needs more pixels-per-class to exploit its curvature
  advantage. Several small pseudo-singer clusters (n_frames ~ 825)
  reach ~98% linear but only ~95% RBF — i.e. RBF overfits the locals.

These numbers should be interpreted as a *realistic 2025 reproduction*
of the methodology, not as a benchmark on the original Cantina Band
dataset.

### Why the PyTorch line is a separate row in the table

We kept the original `nn.Linear + HingeLoss` core of `../SVM.py`
intact for the comparison row, then applied the **two missing ingredients** that
make it an honest implementation of the paper's Eq.(7):

1. add L2 weight-decay so the SGD looks at the SVM primal, not bare hinge;
2. Z-score the MFCC per-singer on the training fold before SGD.

Without either of those the same code settles at ~0.62 CF (38% err); with both
it gets to 0.839 (16% err), within ~2.5 points of sklearn's QP answer.
The full ablation table is in `../REVIEW.md`, sec. 5.

## Files

```
fsvm/
├── config.yaml           # single source of truth for all hyper-parameters
├── utils.py              # tiny yaml/DotDict + traverse_dir helpers
├── preprocess.py         # step 1:  MFCC extraction per wav
├── singer_id.py          # step 2:  Resemblyzer d-vector + Spectral Cluster → singer-id
├── train.py              # step 3:  per-singer 7-fold CV with LinearSVC / SVC(rbf)
├── torch_linear_svm.py   # step 3b: PyTorch-only LinearSVM for cross-check with Eq.(7)
├── main.py               # step 4:  singer-aware inference for a clip
├── cache_wavs/           # 1280 wavs (extracted from CCMusic audio.zip—not included in this repo)
├── mfcc/ label/          # step 1 outputs: per-wav mfcc.npy / label.npy
├── data/singer_id/       # step 2 outputs: embeddings.npy, singer_id.csv
├── results/models/       # step 3 outputs (generated at train time, not included)
└── results/falsetto/     # step 3 outputs: per_singer_results.csv, summary.json
```
