# PyTorch vs. scikit-learn SVM — differences & gotchas

This is a practitioner-oriented write-up that explains why the original
`H:/GitHub/Falsetto-SVM/SVM.py` — a faithful snippet of
`nn.Linear(13, 1) + HingeLoss` — does *not* match paper Eq.(7) even
though both sketches "look the same", and what you'd need to do to
make them match.

The companion reproduction that lives at `J:/FSVM/fsvm/` exercises
both paths in identical 7-fold cross-validation loops on the same
singer-frames. The numbers there are the empirical evidence behind
everything written below.

---

## TL;DR

| Aspect | sklearn `LinearSVC` / `SVC` | PyTorch `nn.Linear + HingeLoss` |
|---|---|---|
| What gets minimised? | Dual quadratic program (QP) over `α_i`, closed-form | Iterative gradient descent on a primal loss with optional L2 |
| Regularisation | `C` parameter (high C = low margin penalty) | `weight_decay` (high wd = low margin penalty) — *inverse direction!* |
| Solver | liblinear / SMO (C) — direct, exact | SGD / Adam — approximate |
| Kernel | Built-in (`linear`, `rbf`, `poly`, ...) | Needs a custom kernel layer or manual expansion |
| Stopping criterion | `tol` on dual objective gap | Early-stop on validation loss / plateau |
| Reproducibility | Deterministic (single numerical solver) | Stochastic (depends on `manual_seed`, batch order, lr schedule) |
| Calibrated output | `decision_function` & `predict_proba` (platt-scaled) | Raw logit only; you decide on a threshold |
| Class imbalance | `class_weight='balanced'` supported | Must weight `pos_weight` of `BCELoss` or scale labels yourself |

For 13-D features and 800..5k frames these distinctions seem academic;
the empirical deltas are ~2.5 percentage points on this dataset — but
they matter in general.

---

## 1. How the two losses relate (and don't)

The paper's linear SVM target (Eq.(7)):

$$
\min_{\mathbf{w},b}\;\tfrac{1}{2}\| \mathbf{w}\|^2 + C \sum_i (1 - y_i (\mathbf{w}\cdot\mathbf{x}_i + b))_+ \tag{1}
$$

sklearn's `LinearSVC` solves the **dual** of (1):

$$
\max_{\boldsymbol{\alpha}} \;\sum_i \alpha_i  - \tfrac{1}{2} \sum_{i,j} y_i y_j \alpha_i\alpha_j\, K(\mathbf{x}_i,\mathbf{x}_j)\quad \text{s.t.}\; 0 \le \alpha_i \le C \tag{2}
$$

Liblinear computes the exact optimum of (2) in a finite number of
coordinate-descent steps; the resulting `w` equals the primal optimum
of (1) to ~`tol` precision.

PyTorch's `nn.Linear + HingeLoss` directly minimises (1) **via SGD**:

```python
loss = F.relu(1 - y_true * y_pred).mean()    # mean hinge
wd   = 1/(C · N)                              # see § 2
optimizer = SGD(model.parameters(), lr=lr, weight_decay=wd)
```

There are *three* structural differences vs. sklearn:

1. **Optimisation target.** Sklearn finds the **global** optimum of the
   primal-dual pair (1)/(2); SGD finds a *local* optimum of the primal
   (1) — usually the same point because (1) is convex, but at finite
   iterations you may still drift.
2. **C and `weight_decay` move in opposite directions.** Higher `C`
   makes the model *less* regularised (bigger margins are penalised
   less). Higher `weight_decay` makes the model *more* regularised.
   They don't translate linearly; you must invert.
3. **Per-sample vs. mean loss.** `HingeLoss.mean()` divides by `N`;
   `LinearSVC`'s C multiplies the *sum*.  This shifts the L2
   coefficient by a factor of `1/N`.

The empirical collapse we'll see in § 3 is a consequence of these.

---

## 2. Choosing `weight_decay` to approximate sklearn's C

Pick from this menu — don't trust intuition:

| `wd` formula | Behaviour with N | Use case |
|---|---|---|
| `0` (no L2)                | useless for SVM — only minimizes hinge; `W` keeps growing           | bad |
| `1/N`                      | scales with N (Primal SGD mean-loss = sum/N)                       | none, missing the √N term |
| `1/(C · N)`                | average scaling, OK for one fixed N                                | small-medium fixed N |
| `C`                        | only depends on `C`, adapts by N mainly via early-stop plateau   | medium-large N, robust across N |
| `1/(2 · C · N²)`           | shrink more for big N                                              | tight to dual only asymptotically |

Empirically on our 825..4675-frame singers, **`wd = C`** tracks
`LinearSVC(C=0.0413)` within ~3 absolute CF points with zero per-N
tuning. This is what `J:/FSVM/fsvm/torch_linear_svm.py` defaults to.
If you want a tighter fit, use a held-out validation grid — but then
the simplicity is gone.

> **Rule of thumb**: for any `nn.Linear + HingeLoss + weight_decay`
> that you intend to compare against `LinearSVC(C=...)`, **start with
> `weight_decay = C` (or `weight_decay = C / N`) and tune from
> there.** Don't jump into SGD without a known L2 anchor.

---

## 3. Why the upstream `SVM.py` collapses (and the fix)

```python
# H:/GitHub/Falsetto-SVM/SVM.py
class SVM(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fully_connected = nn.Linear(input_dim, output_dim)
class HingeLoss(nn.Module):
    def forward(self, y_pred, y_true):
        return torch.mean(F.relu(1 - y_true * y_pred))
```

The accompanying `config.yaml` had `weight_decay: 0` and `lr: 5e-6`.
Two failure modes conspire:

- **L2 = 0 + unbounded W growth.**  Without weight_decay, the network
  is free to drive `||w|| → ∞` as long as hinge stays at 0.  Even an
  isolated SGD trail will settle on a steep plateau where every
  hinge term is negative and the gradient is zero (no update rule
  pulls `w` back).
- **Raw MFCC has 13× scale mismatch (`std[0] ≈ 72` vs `std[12] ≈ 13`).**
  Plain SGD uses one global learning rate; the dominant coefficient
  controls its effective magnitude self-consistently, but on
  validation-time data the search direction is wrong.

Quantitative result on the singer-0 fold of the paper's 7-fold CV
(absolute):

| configuration | CF | err |
|--------------|---:|----:|
| upstream `SVM.py`, `wd=0`, raw MFCC | 0.62 | 38% |
| + `weight_decay = C = 0.0413` | 0.78 | 22% |
| + `weight_decay = C` + per-singer train-fold Z-score | **0.84** | 16% |
| `sklearn LinearSVC(C=0.0413)` (for reference) | 0.86 | 14% |

The two missing ingredients (L2, standardisation) recover ~22 points
of absolute accuracy.

---

## 4. The two kernels, side-by-side

### Linear (`LinearSVC` vs `nn.Linear + HingeLoss`)

| | sklearn | PyTorch |
|---|---|---|
| Solver | liblinear QP | SGD |
| Convergence | "happens", almost always <100ms | user must pick `epochs`, `lr`, early-stop |
| Calibration | exact `decision_function` & sparse `support_` | raw logit |
| Strong points | fast & reproducible on small N (<10⁵) | flexible: warm-start, custom losses, GPU batches, easy to integrate with deep nets |
| Weak points | no GPU, no custom losses | requires careful hyper-tuning (see § 3) |

Below ~50k samples and a simple linear model, sklearn is *strictly*
preferred unless you want rich output, deep integration, or feature
interaction with another PyTorch module.

### RBF / Gaussian (`SVC(kernel='rbf')` vs the same `nn.Linear`)

| | sklearn | naive PyTorch |
|---|---|---|
| Implementation | sliced kernel via SMO on subset | physically `RBF` requires a **kernel matrix** K ∈ ℝ^{N×N} |
| Memory | O(N²) double, acceptable at 5k samples | same — but `nn.Linear` is **not** a kernel; it is a linear map |
| How it works | replaces inner product `xᵢ·xⱼ` with `exp(-γ‖xᵢ-xⱼ‖²)` | would need `K = rbf(Xtr); α = ...) - 1)` then `pred = α·K_te·y` |

You **cannot** emulate RBF using only `nn.Linear + HingeLoss` — that
combination only ever represents a linear separator in the input
space. Implementing RBF via PyTorch requires either of:

1. **Pre-compute the kernel matrix** to a tensor, then write a
   `nn.Module` whose `forward(x)` does `K(x, X_train)·α` (a
   "kernel ridge" approach). Loss is then computed against that
   output. Use `nn.L1Loss`/`MSE` on (1 - y·f(x))_+ as a *hinge-like*
   surrogate (or just square it).
2. **Random Fourier features.** Approximate the RBF kernel with
   `z(x) = sqrt(2/D)·cos(ω·x + b)`, ω ~ `Normal(0, 2γ·I)`, b ~ U(0, 2π).
   Then a linear model on `z(·)` is *asymptotically* a kernel model.
   This is the only way to keep just `nn.Linear`.
3. **Use sklearn under the hood.** Lift `joblib` outputs into
   PyTorch by `state_dict`-ing the support vectors and
   coefficients. Cleanest answer if precision matters; see
   `J:/FSVM/fsvm/main.py` for how we did this on the linear path.

For the paper's RBF claim of 1.5% error, we used option (3) — keep
`SVC(kernel='rbf', C=2.057, gamma='scale')` as the trainer and
export to `joblib`.

---

## 5. Reproducibility checklist

When you write PyTorch-side SVM code that *must* match a sklearn
result to 1 CF point, run through this checklist:

1. **Set the seed.** `torch.manual_seed(0); torch.cuda.manual_seed_all(0);
   np.random.seed(0); random.seed(0)`. CUDA non-determinism still
   exists; for a 13-d task CPU is fine.
2. **Pick the L2 anchor.** Match `weight_decay = C` (or `C / N`) and
   document which.
3. **Scale features** *on the train fold only*; transform val/test
   with the train statistics. We use simple mean/std; you can also
   try `sklearn.preprocessing.StandardScaler` then convert to
   torch tensors.
4. **Pick the optimizer & lr.** For the hinge primal SGD on small N
   use `lr ∈ [1e-3, 5e-3]`, full-batch, and plateau-based early
   stopping.
5. **Choose the number of epochs.** Open-ended in PyTorch; we found
   5000 epochs with early-stop on validation loss is reliable.
6. **Verify against sklearn on a fold.** Run `LinearSVC(C=...)` on
   the same `(X_train, y_train)` and compare `(pred == y).mean()`.
   They should agree to within ~2 percentage points after the above
   fixes.
7. **Don't mix old + new.** If you bump the L2 anchor or the
   scaler, re-run the sklearn comparator. They aren't
   auto-corrections; they're individual dialling's.

---

## 6. Practical pitfalls

1. **`F.relu(1 - y · y_pred)` collapses to all-zeros.** Common when
   `weight_decay = 0` or when labels are not `{-1, +1}` (the paper's
   convention) but `+1`-only or `0/1`. Always `print(loss)` once
   during dev — if it's `0.0` you have a degenerate gradient.
2. **Loss curves that don't decrease.** Usually a learning-rate
   issue, not a bug in the loss. Step size too large ⇒ bounces; too
   small ⇒ takes thousands of epochs to converge. Print and bracket.
3. **`torch.tensor` dtype mismatches:** `1 - y*y_pred` requires
   `y_pred` and `y` to match dtype; we cast to `float32` early in
   `torch_linear_svm.py`.
4. **GPU↔CPU gradients:** the original `SVM.py`'s `nn.Linear`
   forward expects `(B, T, 13)` because of the
   `x.transpose(1, 2)`. The shape contract is subtle and
   incompatible with a plain `(B, 13)` flat frame layout. We
   flattened to `(B, 13)` for SGD parity with `LinearSVC`.
5. **`imblearn`-style class weighting.** sklearn has
   `class_weight='balanced'`. In torch you must construct per-sample
   weights and apply them yourself before the `.mean()`.
6. **Decision threshold.** sklearn's `predict()` maps the decision
   function with sign; for inferring probability you must platt-scale
   manually (`sklearn.calibration.CalibratedClassifierCV` after the
   fact). The paper does not need this for CF.
7. **`HingeLoss` vs. `nn.HingeEmbeddingLoss`.** The latter is *not*
   a SVM loss; it's the metric learning loss `max(0, 1 - y · |x-y|)`.
   The original repo's HingeLoss is the right one — a comment in
   SVM.py even warns about this. Keep that warning in any rewrite.

---

## 7. Where to find the reference code

| Stage | File |
|---|---|
| Upstream clone (the code under audit)               | `H:/GitHub/Falsetto-SVM/SVM.py` |
| Reproduction suite (PyTorch *and* sklearn paths)   | `J:/FSVM/fsvm/torch_linear_svm.py`, `J:/FSVM/fsvm/train.py` |
| Inference main pairing both back-ends              | `J:/FSVM/fsvm/main.py` |
| Full triangulation: paper ↔ upstream ↔ repro     | `H:/GitHub/Falsetto-SVM/REVIEW.md` |

A clean ablation table (what we measured vs. what we expected) is in
`REVIEW.md` § 5.
