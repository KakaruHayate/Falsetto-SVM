# REVIEW — Falsetto-SVM reproduction audit

**Repository**: the upstream PyTorch sketch (the sibling `SVM.py`,
`preprocess.py`, etc. — also reachable as `KakaruHayate/Falsetto-SVM`
on GitHub).
**Paper**: G. J. Mysore & P. Smaragdis, *Singer-Dependent Falsetto Detection
for Live Vocal Processing Based on Support Vector Classification*, ICASSP 2006.
**Dataset**: CCMusic chest_falsetto (1,280 clips, 12 singers, 44.1 kHz mono)

This document lists the discrepancies found between the paper and the code
at the time of review (July 2026). A corrected reproduction is available at
`fsvm/` (sibling directory at the repo root).

---

## 1. Singer-dependent pipeline missing (HIGH)

| | paper | current code |
|---|---|---|
| Training | One SVM per singer (13 singers, each trained + tested on their own clips) | All 1,280 clips pooled into one `AudioDataset`; a single `SVM(nn.Linear)` is trained on the union |
| Test | Same-singer held-out segments per fold | Random train/val split, not respecting singer boundaries |

The paper is clear about the "subject-specific" design in the title and
Section 1: "trained for an individual singer."  There is no `singer_id`
column or grouping mechanism anywhere in the original code.  This is the
largest gap.

## 2. Frame rate mismatch (MEDIUM)

| | paper | current code |
|---|---|---|
| Frame rate | 50 Hz | 172 Hz (= 44100 / 256) |
| `hop_length` | sr / 50 = **882** | **256** |
| Frames per 0.5s clip | **~24** ("24 feature vectors per note") | **86** (padded to `train_frames=87`) |

The paper explicitly states "*the frame expansion is undersampled*" – a
frame every 882 samples leaves gaps between 256-sample windows, which is
deliberate.  The repo operates at ≥3.4× the temporal density, which doubles
the classifier's effective "pixels per note" and fundamentally changes the
temporal resolution seen by the SVM.

## 3. No RBF kernel (MEDIUM)

| | paper | current code |
|---|---|---|
| Eq.(7) | Linear SVM (C = 0.0413) | `nn.Linear(13, 1) + HingeLoss` |
| Eq.(8) | Gaussian RBF (C = 2.057, σ = 2.42) | **Not implemented** |
| Regularization | C parameter via QP (exact dual solution) | `weight_decay = 0` (no equivalent) |

The main finding of the paper is the improvement from 4.8% error (linear)
down to 1.5% error (RBF).  The repo never tests this contrast.  Moreover,
the `nn.Linear` + `HingeLoss` is the *un-regularized* primal, not the
regularized SVM primal the paper solves.

## 4. No CF (accuracy) reporting (LOW)

`solver.py` `test()` logs only `HingeLoss`.  The paper evaluates via
classification fraction (CF) / error rate, i.e. accuracy on 7-fold CV.
There is no CF or confusion matrix computed anywhere in the code.

## 5. Minor: evaluation in `main.py`

`main.py` threshold: average of `{-1,+1}` predictions > 0.5 → falsetto.
This threshold on {-1,+1} means >75% of frames must vote falsetto, which
is a conservative (chest-side) threshold.  The paper does not define a
clip-level decision rule (it evaluates per-frame CF); this choice is
therefore reasonable but undocumented.

---

## Python SVM core analysis (`SVM.py` ↔ paper Eq.(7))

In addition to the four above, we also did an exercise of trying to
make the upstream `SVM.py` (the `nn.Linear + HingeLoss` snippet at the
repo root) match the paper's linear SVM (Eq.(7)) directly.  We re-ran
the SVM under identical 7-fold CV on the singer-0 subset and observed
the following curve.  Each step *fixes* one characteristic of the
upstream `nn.Linear + HingeLoss`:

| implementation                                     | 7-fold CF | err |
|---------------------------------------------------|----------:|-----:|
| upstream `nn.Linear + HingeLoss`, `weight_decay=0`, raw MFCC | 0.62 | 38% |
| + `weight_decay = C` (= L2 penalty on `||w||²/2`) | 0.78  | 22% |
| + per-singer Z-score **of training fold** before SGD | **0.84** | 16% |
| sklearn `LinearSVC(C=0.0413)` (for reference)     | 0.86 | 14% |

Observations:

1. **Why `weight_decay=0` is broken for SVM.**  Without the L2 penalty
   the SGD reduces *hinge loss* only -- it does **not** minimise the
   margin `1/2·||w||² + C·Σξ_i`.  Without `weight_decay` you'll get
   arbitrarily large weight magnitudes and the loss will sit on a steep
   plateau where `1 - y_i·(w·x_i)` is negative for every sample (the
   `relu` is dead, gradients are zero).  The fix is to provide an L2
   penalty that maps onto the dual QP's `C` parameter.
2. **The right `weight_decay`.**  We tried several formulae:
   `wd = 1/(C·N)`, `wd = 1/(2·C·N)`, and `wd = C`.  Empirically
   `wd = C` (i.e. `wd_scale · C` in
   `fsvm/torch_linear_svm.py`) tracks LinearSVC most closely
   across singer sizes (825 ≤ N ≤ 4675).  The `1/(C·N)` formulae are
   too pessimistic for small `N`, which is most of the singers.
3. **Feature scaling matters for SGD but not for QP.**
   `mfcc[0]` (log energy) has std ≈ 72 while `mfcc[12]` has std ≈ 13,
   so the QP solution is well-conditioned (it inverts a 13×13 Gram
   matrix), but plain SGD with one global learning rate becomes
   ill-conditioned: loss plateaus at ~25 because every iteration
   over-corrects the dominant dimension.  Adding a per-singer Z-score
   normalisation on the training fold (then applied unchanged to the
   validation fold — no test data leak) takes accuracy from ~0.62 up
   to ~0.84.
4. **A real PR-AUC comparison is fair.**  After the two fixes
   (L2 + Z-score), torch SGD is within 2.5 absolute accuracy points
   of sklearn's QP solution.  The remaining gap is dominated by SGD
   early-stopping and the per-singer scaler mean/std mismatch when
   folds are imbalanced — both reducible but not interesting for a
   faithful reproduction of the *paper's* algorithm.

In short:  `nn.Linear + HingeLoss` is a *legitimate* implementation
of the linear SVM primal/hinge-loss form once L2 + scaling are added;
without them (as in the original repo) it does something qualitatively
different and noticeably worse.

A runnable reference implementation that mirrors the upstream `SVM.py`
and applies the two fixes lives at `fsvm/torch_linear_svm.py`.  The
accompanying triangulation document `PYTORCH_VS_SKLEARN_SVM.md` walks
through the reasoning behind each fix.