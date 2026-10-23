# Methodology

This document describes what the code actually does, in the order in which data
are transformed. Every number quoted in [`RESULTS.md`](RESULTS.md) corresponds to
an artifact produced by these steps.

---

## 1. Data model and preprocessing

### 1.1 Source representation

`data/kato.py` reads MATLAB v7.3/HDF5 with `h5py` (not `scipy.io.loadmat`, which
rejects v7.3). For each worm it exposes:

- `traces`: `float32` array `T × N` of ΔF/F traces, already bleaching-corrected
  by the original authors;
- `raw_traces`: the same before NeuroBranch smoothing/detrending;
- `neurons`: cell names where the original archive provides them, and opaque
  numeric labels otherwise;
- `states`: one of the eight Kato labels per frame;
- `fps ≈ 2.90–2.91`, `time_seconds`;
- source path and SHA256.

The archive contains **more segmented traces than named cells** (109–135 traces,
31–47 named). Unnamed traces are retained as predictors because they are real
recorded sources; they cannot be targeted by name in an intervention and are
never assigned a biological identity.

### 1.2 Behaviour labels and circularity control

Kato's labels are derived from calcium in command/interneuron populations. The
following are excluded from model inputs:

```
AVAL, AVAR, SMDDR, SMDDL, SMDVR, SMDVL, RIBR, RIBL
```

This removes the most direct label-defining channels but **does not** make labels
independent: the remaining neurons are co-recorded in the same animal, and the
annotators used the worm's behaviour. All decoding metrics are therefore
*observational consistency* checks, not independent validation.

Macro map:

| Fine label | Macro |
|---|---|
| `fwd`, `slow` | FWD |
| `rev1`, `rev2`, `revsus` | REV |
| `dt`, `vt` | TURN |
| `nostate` | NOSTATE |

### 1.3 Optional filters (off by default for the primary suite)

- `causal_moving_average(x, width)`: trailing mean, no future samples.
- `causal_highpass_segment(x, width)`: `x - trailing_mean(x)`, computed
  independently within each contiguous split segment, with warm-up samples
  discarded. This is a causal approximation to the zero-phase filter used
  upstream, chosen so that no future frame influences a training input.
- No global standardisation, no PCA, and no imputation is fitted on held-out
  frames. The scaler is fitted on the training mask only.

Primary config uses **no smoothing and no detrending** (`smoothing_sigma_frames:
0`, `detrend_window_frames: 0`). The reason is empirical: aggressive causal
high-pass filtering removed slow common-mode drift but also removed the
behaviourally relevant slow component and did not improve held-out prediction.
This choice is recorded in [`DECISIONS.md`](DECISIONS.md).

---

## 2. Splits and leakage control

Two regimes are reported; they answer different questions.

### 2.1 Primary: blocked interpolation (`blocked`)

Each recording is divided into five contiguous blocks. For the primary suite:

- train = blocks 0, 1, 3
- validation = block 4
- test = block 2 (the middle block)
- purge = 12 frames at every boundary

This holds out an interior time span and trains on both earlier and later data.
It is **interpolation in time**, useful for model-class comparison under a
stationary-ish assumption, and **not a forecasting benchmark**. The label
"blocked" is used everywhere to prevent the common misreading that this is
prospective prediction.

### 2.2 Negative control: strict forward (`temporal`)

Train on the first 70%, validate on the next 15%, test on the final 15%, with a
12-frame purge. This is the honest forecast regime and exposes non-stationarity:
late recordings are behaviourally degenerate in several worms (for example worm 1
is almost entirely FWD in the final block, worm 2 almost entirely REV).

### 2.3 Window validity

A window of history `h` ending at `t` and target `t+Δ` is admitted only if every
frame from `t-h+1` to `t+Δ` lies inside the same split. Starts are computed after
masking, so a window cannot silently cross a purge gap. This is enforced in
`preprocessing.split_starts` and tested.

### 2.4 What is fitted where

- Scaler: training frames only.
- PCA (latent models, branching projection): training frames only.
- Behaviour probe: training windows only.
- Early stopping: validation block only.
- Test frames are touched once, for the final metrics.

No test-based model selection, threshold tuning, or feature selection is
performed.

---

## 3. Models

All models consume a history of standardised neural activity and predict the
next standardised state. Behaviour probabilities come from a multinomial logistic
probe on the same history.

| Kind | Description | Parameters / notes |
|---|---|---|
| `persistence` | `x̂_{t+1} = x_t` | zero parameters; the benchmark to beat |
| `var` | ridge VAR(history) | history 1 in primary suite; closed form |
| `mlp` | feed-forward MLP, direct next-state prediction | 2 hidden layers, dropout, early stopping |
| `mlp_residual` | MLP predicts a residual added to `x_t` | same backbone, residual output |
| `gru` | GRU over history (12 frames) | sequence model |
| `latent_mlp` | PCA (16-D, train-only) + MLP latent dynamics, decoded | linear decode |
| `latent_knn` | PCA + stochastic k-nearest-neighbour transition in latent space | non-parametric; support/OOD aware |

Design notes:

- The residual model was introduced after diagnosing a double-residual bug in an
  earlier version; the bug is fixed and the earlier result is not used.
- The latent kNN transition uses a softmax over negative squared distances. When
  weights underflow, it falls back to a uniform distribution rather than
  producing `NaN`.
- GRU uses a 12-frame history rather than 15 so that the blocked split has enough
  valid training windows after purging.
- Hyperparameters were fixed before the final suite and are recorded in
  `configs/experiments/*.yaml`; no per-worm tuning was performed.

---

## 4. Evaluation metrics

Let `x_t` be the standardised true state and `x̂_t` the prediction.

- **One-step R²**: pooled over test frames and neurons,
  `1 - SS_res / SS_tot`, where `SS_tot` uses the test-set mean per neuron.
  Negative values mean worse than predicting the test mean.
- **Rollout MSE at horizons 1, 3, 5, 10**: open-loop rollout from held-out test
  starts, compared with the observed future. Noise is zero for the deterministic
  metric; intervention simulations use explicit noise.
- **Behaviour accuracy**: argmax of the probe's macro-state distribution against
  the observed macro label at `t+1`.
- **Behaviour persistence accuracy**: accuracy of predicting the current macro
  label for the next frame. This is the relevant trivial baseline for decoding,
  because behaviour is highly autocorrelated.
- **Transition total variation (TV)**: distance between the predicted and
  observed distribution of macro-state transitions at a 20-frame horizon,
  averaged over held-out reversal starts. Persistence's TV is computed against
  the same observed transitions with a "stay in REV" prediction.
- **Bootstrap intervals**: cluster bootstrap over held-out starting frames, so
  temporally adjacent windows are not treated as independent.
- **Paired comparisons**: each model is compared with persistence on the same
  worm and seed. The number of wins out of 15 pairs is reported alongside the
  mean paired difference.

---

## 5. Branching analysis

The word "branching" is used operationally, not mechanistically.

For a candidate frame `t` (a held-out REV frame with at least 20 future frames):

1. Project all neural states to a 3-D PCA space fitted on training frames.
2. Find the `k=40` nearest neighbours of `z_t`, excluding frames within ±12
   frames of `t` (to prevent trivially overlapping windows from dominating).
3. For each neighbour, look up its macro state 20 frames later.
4. Compute the empirical distribution over the four macro outcomes and its
   Shannon entropy in bits.

A high entropy means that *similar current states in this projection had
different futures*. It does **not** establish a dynamical bifurcation, an
unobserved discrete state, or a causal branch point.

Two controls are run:

- **Shuffled-outcome null**: future outcomes are permuted across candidate
  frames, preserving the marginal outcome distribution and the candidate
  geometry, destroying the state→future association. 100 permutations.
- **Projection-dimension stability**: the same analysis in 8-, 16-, and 32-D PCA,
  compared by Spearman correlation of per-frame entropy with the 3-D result.

A model-based branching measure applies the same outcome logic to rollouts from
the fitted model, so observed and simulated branching can be compared without
claiming they are the same object.

---

## 6. Virtual perturbation

### 6.1 Operation

An `InterventionSpec` selects named neurons, a mode, an amplitude in
standardised SD units, an onset frame, a duration, a horizon, a number of initial
states, and simulation seeds. Supported modes:

| Mode | Effect at frame `s` in `[onset, onset+duration)` |
|---|---|
| `additive` | `x_i ← x_i + a` |
| `clamp` | `x_i ← a` |
| `zero` | `x_i ← 0` |
| `scale` | `x_i ← a · x_i` |

Behaviour-label neurons are rejected at spec-construction time. Unnamed traces
cannot be targeted.

### 6.2 Paired design

For each held-out initial frame:

1. Draw a baseline rollout and an intervention rollout with the **same random
   seed** (common random numbers).
2. Apply the perturbation only to the intervention branch, during the specified
   window.
3. Record the distribution of macro behaviour over the horizon, the trajectory
   displacement, and the fraction of states outside the model's training support.

Reported effect: `ΔP(target behaviour) = P_intervention − P_baseline`, with a
cluster bootstrap over initial frames and variability over simulation seeds.
Random-neuron controls use the identical amplitude, timing, horizon, and seeds.

### 6.3 Out-of-support warning

For neural models, OOD is flagged when the model's own `ood_score` exceeds the
99th percentile of training scores. For latent kNN, it is the distance to the
k-th training neighbour. The reported OOD fraction is the proportion of simulated
states beyond that threshold. A large effect with a large OOD fraction is not
evidence of a biological effect; it is evidence that the model was extrapolating.

### 6.4 Causal status

`InterventionSpec` carries `EmpiricalEvidence`, which must document a real
intervention dataset, its hash, matched controls, species, negative control, and
pre-registration before `validated` can be true. Without it, every result is
labelled:

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

The gate is deliberately conservative and can reject even a real dataset if the
manifest is incomplete. It is a guard against rhetoric, not a substitute for
peer review.

---

## 7. Reproduction of upstream work

### 7.1 BunDLe-Net

- Official 3-D embeddings (`bundlenet_consistent_embedding_worm_*.npz`) are
  loaded and evaluated with a kNN classifier and silhouette score.
- **These embeddings were fitted on all windows by the upstream code.** Our kNN
  classifier holds out training-block labels, but the test embeddings were seen
  by the encoder, so the evaluation has **embedding-level leakage** and cannot be
  compared with a held-out number. The silhouette is fully in-sample. This is
  stated in every table and figure.
- A PyTorch reimplementation (`BunDLeLite`) reproduces the architecture
  (50-30-25-10-3 MLP, residual transition, 8-class head, Gaussian noise 0.05,
  `γ=0.9`, 15-frame windows) but uses a blocked train/test split and a causal
  low-pass filter. Deviations are listed in
  `artifacts/reproduction_bundlenet_lite.json`.
- The first reimplementation used upstream's **zero-phase** `filtfilt` on the
  full recording before splitting, which is acausal and leaked future test
  frames. It scored kNN 0.624. With the causal filter the held-out kNN is 0.407
  and the silhouette is −0.064. The higher number is treated as a leakage
  artifact, not as evidence of representation quality.

### 7.2 NC-MCM

- Distributed `.npz` artifacts are treated as read-only inputs.
- The lag-2 surrogate Markovianity test is reimplemented in Python and compared
  with the stored p-values at `K=3` for all five worms.
- A 5-fold blocked logistic decoder reproduces the behaviour-decoding analysis.
- The artifact's stored state-name array is misaligned with its numeric codes;
  NeuroBranch uses `STATE_ORDER` rather than the stored names.

---

## 8. Software and determinism

- Python 3.13.5, NumPy 2.3.1, SciPy 1.16.0, scikit-learn 1.9.1, PyTorch 2.7.1.
- Seeds are set for NumPy and PyTorch at each training run. Full bitwise
  determinism is not claimed for CPU threading or PyTorch kernels.
- Every run writes `metrics.json`, `train_history.json`, `test_predictions.npz`,
  `model.pt`, and `resolved_config.yaml`.
- Every suite writes `suite_metrics.csv`, `suite_summary.json`, and
  `run_manifest.json` (Git SHA, dirty flag, dataset hash, config hash, split
  manifest, seeds, environment, command, output paths).
- Environment and provenance are recorded; see
  [`EXPERIMENT_REGISTRY.md`](EXPERIMENT_REGISTRY.md).
