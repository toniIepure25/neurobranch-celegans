# Master Review: Code and Mathematical Audit

**Scope:** the exact code path that produces the Kato common-state result.  
**Branch/HEAD:** `research/kato-final-reproducibility-audit` / `96a8dd333643de0564247e2c01c98a85511dac82`.  
**Evidence:** source line citations, independent reference calculation, frozen artifacts, and executable bounds tests.  
**Verdict:** arithmetic passes; the inferential design has three material confounds. No future-frame leakage was found.

---

## 1. What was audited

The audit followed the data path:

```text
WT_NoStim.mat
  -> Kato loader (deltaFOverF_bc, NeuronNames, States, fps, tv)
  -> predictor exclusion (eight behaviour-label names)
  -> train-only standardization
  -> train-only PCA(10)
  -> per-recording temporal split
  -> test landmarks (history 2, stride 10, H100)
  -> rolling causal fits at each landmark
  -> B3 / B5 / B4m / B1m / B0 ensembles
  -> energy, MSE, coverage, covariance, autocorrelation
  -> paired differences and moving-block intervals
```

The audit did not refit the full 175-landmark analysis. It:

- read the exact model and metric code;
- recomputed three landmarks with an independent reference implementation;
- recomputed all split and landmark bounds from HDF5;
- recomputed the paired effect sizes and coverage from the frozen per-landmark table;
- checked the bootstrap construction;
- ran focused tests.

The independent reference check passes with maximum absolute error `7.46e-7`, well inside the declared `2e-5` PCA/float tolerance. The arithmetic is not the problem.

---

## 2. Equations as implemented

### 2.1 Standardization

`src/neurobranch/preprocessing/__init__.py:174-177`:

```text
mean_j = mean_{t in train} x_{t,j}
sd_j   = std_{t in train} x_{t,j}
z_{t,j} = (x_{t,j} - mean_j) / sd_j
```

Only training frames are used. There is no test mean or test variance. The primary configuration uses no smoothing and no detrending. This is correct.

### 2.2 B3: full population VAR

`src/neurobranch/adaptive/var.py:42-100`, `src/neurobranch/kato_interactions/models.py:91-130`.

For standardized `z_t in R^n`:

```text
z_{t+1} = A z_t + b + eps_t
```

The ridge objective, after centering, is:

```text
min_A  ||Y - Z A^T||_F^2 + lambda ||A||_F^2,   lambda = 1.0
```

with an unpenalized intercept. The code solves:

```text
A^T = (Z^T Z + lambda I)^{-1} Z^T Y
b   = mean(Y) - mean(Z) A^T
```

Then the operator is scaled if its spectral radius exceeds `0.995`:

```text
if rho(A) > 0.995: A <- A * 0.995 / rho(A)
```

The parameter count is `n(n+1)`: `n^2` coefficients plus `n` intercepts.

**Verified:** the independent reference reproduced B3 energies on sessions 0, 2 and 4; across all cases the maximum absolute difference was `5.78e-7` for energy and `7.46e-7` for MSE.

### 2.3 B5: own history plus ten common factors

`src/neurobranch/kato_common_state/models.py:224-292`.

A 10-component PCA is fitted on training frames only:

```text
f_t = P^T (x_t - mean_train),   P in R^{n x 10}
```

The factor scores follow a fitted VAR:

```text
f_{t+1} = A_f f_t + b_f + eta_t
```

Each neuron is predicted from its own current value and the current factors:

```text
x_{t+1,j} = b_j + a_j x_{t,j} + beta_j^T f_t + eps_{t,j}
```

This is a diagonal-plus-rank-10 model. It has `12n + 110` parameters for `k=10`:

- `n` own-lag coefficients;
- `10n` factor loadings;
- `n` intercepts;
- `10*11 = 110` factor-VAR coefficients.

**Verified:** the independent reference reproduced B5 energies on three sessions; maximum absolute difference `5.8e-7`.

### 2.4 Shared innovations

B3, B5, B4m, B1m and B0 use the same J5 innovation stream at each landmark. This is the right way to compare conditional means and distributions: the models see the same random shocks. The independent check reuses only the J5 sampler, not the model fits, PCA or rollouts.

The shared-innovation design does **not** equalize the effective stochastic process. B3 and B5 propagate the same shocks through different operators, so their ensemble spread can differ even with identical innovations. That is exactly what the coverage difference shows.

### 2.5 Energy score

`src/neurobranch/probabilistic/metrics.py:24-42`:

```text
ES(P, y) = E_{X~P} ||X - y|| - 0.5 E_{X,X'~P} ||X - X'||
```

The second term uses distinct pairs only (`triu_indices(m, k=1)`). This is the fair finite-ensemble estimator. Lower is better. The implementation is correct.

The score is proper for the ensemble distribution. It rewards both calibration and sharpness. It is not a mean-accuracy metric.

### 2.6 MSE

`src/neurobranch/kato_common_state/evaluation.py:69-75`:

```text
MSE_H = mean_j (mean_m ensemble_{m,H,j} - truth_{H,j})^2
```

This is the squared error of the ensemble mean, i.e. a conditional-mean metric. It is computed in the 10-D PCA space for `pca10_mse_*`, and in neuron space for `neuron_mse_*`. The PCA transform is orthonormal, but the PCA MSE is still only over the ten retained components; it is not full neuron-space MSE.

### 2.7 Coverage

`src/neurobranch/probabilistic/metrics.py:109-120`:

```text
coverage_95 = mean_j 1[truth_j in central 95% marginal interval of ensemble_j]
```

This is **marginal per-dimension coverage**, averaged over dimensions. It is not joint 95% coverage of a 10-D ellipsoid. A model can have 0.94 marginal coverage and much lower joint coverage.

This matters for the B3/B5 contrast: B5’s `0.725` is already poor marginally; its joint coverage would be worse. B3’s `0.938` is close to nominal marginally but does not establish joint calibration.

### 2.8 Moving-block interval

`src/neurobranch/kato_interactions/metrics.py:36-76`:

- split the paired differences by session;
- resample contiguous blocks within each session;
- concatenate the sampled blocks;
- truncate to the original session length;
- average over all resampled values;
- repeat 1,000 or 5,000 times;
- take the 2.5% and 97.5% quantiles.

The docstring is explicit: “The resulting interval is a repeated-session interval, not an animal-level interval.”

The construction is correct **for its target**. It is not correct for animal-level generalization, because it never resamples the five recordings. A separate cluster bootstrap that resamples recordings gives wider intervals (see §5).

---

## 3. Causal flow: no future leakage, but test-past adaptation

### 3.1 The split guard

`src/neurobranch/preprocessing/__init__.py:219-241`:

For a start `t`, history `h`, horizon `H`:

```text
lo = t - h + 1
window = [lo, t + H]
admit iff every frame in window is in the allowed split
```

This is the key leakage guard. It prevents a window from crossing a purge gap or using a target outside the split. The audit independently verified the stronger H1–H100 conditions.

### 3.2 The rolling fit

`src/neurobranch/kato_common_state/models.py:338-396`:

At landmark `t`:

```text
starts = arange(max(1, t - window + 1), t)
```

The fit uses transitions `s -> s+1` for `s < t`. The target at `t` is observed before forecasting. This is causal: no future frame is used.

However, at a test landmark, `starts` includes **validation and earlier test frames**. The model is refit at every landmark on all observed past data, including test-past. This is online adaptation, not a frozen train-only model.

Is this leakage? **No future leakage.** At forecast time `t`, all frames `< t` are legitimately observed. But it is not the same estimand as “fit once on train, predict test.” The performance includes adaptation to the test session.

This matters because:

- B3 can adapt its full `n x n` operator to the test-past;
- B5 can adapt its own/factor coefficients but not its PCA basis;
- B3’s larger capacity can exploit test-past more;
- the claim “test frames are touched once” in `docs/METHODOLOGY.md` is false for this rolling analysis.

The correct description is:

> Both B3 and B5 are refit causally at each landmark using all observed frames before that landmark, including earlier test frames. There is no future leakage, but the evaluation is an online-refit forecast, not a frozen train-only forecast.

This is a **protocol mismatch**, not a computational leak. It should be reported explicitly.

### 3.3 The PCA basis

The 10-D PCA is fitted on training frames only. Factor scores at test time use the training mean and components. This is correct and avoids a transductive representation.

The **evaluation** PCA and the **common-factor** PCA are the same transform (per the final audit). B5 therefore retains all ten evaluation dimensions; it is not losing evaluation-space information by compression. Its restriction is in the residual operator, not the metric space.

### 3.4 Landmark generation

`src/neurobranch/adaptive/evaluation.py:43-47`:

```text
valid = split_starts(test, history=2, horizon=100)
landmarks = valid[::10]
```

The first landmark is one frame after the test start because the two-frame history must be inside the test mask. This explains why session 0 starts at `2690`, not `2689`.

---

## 4. Findings

### C01 — The primary interval is conditional on the five recordings (high)

**Location:** `src/neurobranch/kato_interactions/metrics.py:36-76`.

**Finding:** the moving-block bootstrap resamples within sessions, then pools. It never resamples sessions. The interval is conditional on these five recordings.

**Impact:** the reported `[+0.075, +0.403]` cannot be read as an animal-level confidence interval. A recording-level bootstrap gives `[-0.041, +0.479]` at block 5 and `[-0.131, +0.440]` at block 20. The t interval across the five session means is `[-0.064, +0.587]`; the sign test is `p=0.1875`.

**Severity:** high, because the claim language often treats the interval as robustness across animals.

### C02 — B3 and B5 are not capacity- or adaptivity-matched (high)

**Location:** `src/neurobranch/kato_common_state/models.py:224-292`; `src/neurobranch/adaptive/var.py:42-100`.

**Finding:** B3 has `n(n+1)` parameters; B5-10 has `12n+110`. The ratio is `7.96–9.95×`. B3 has arbitrary residual cross-neuron rows; B5 is restricted to a diagonal-plus-rank-10 operator. B3 also has a local full operator; B5 has a globally fixed factor basis.

**Impact:** a positive B3-over-B5 score is a valid restricted-model contrast, but it does not isolate “cross-neuron information” from “more parameters” or “more adaptivity.” The ordered 51/49 split is arithmetic, not a structural partition.

### C03 — The energy advantage is partly a spread/calibration contrast (high)

**Location:** `src/neurobranch/probabilistic/metrics.py:24-42`, `109-120`; frozen `model_per_landmark.csv`.

**Finding:** B3’s marginal 95% coverage is `0.938`; B5’s is `0.725`. The PCA-10 MSE difference is `+0.0545` with a block-5 interval `[-0.157, +0.241]`; the standardized mean MSE effect is `dz = 0.04`. The energy effect is `dz = 0.27`.

**Impact:** the energy score rewards calibration and sharpness, not just the mean. B5 is severely undercovered; its energy deficit can arise from too-narrow ensembles. The result should not be described as a mean-forecast improvement.

This is not a claim that B5’s undercoverage **fully** explains the gap. It is a claim that the current analysis does not separate the two. A calibration decomposition or a spread-matched ensemble comparison is required.

### C04 — The model is refit on test-past, contrary to “test touched once” (medium-high)

**Location:** `src/neurobranch/kato_common_state/models.py:338-396`.

**Finding:** rolling fits use all transitions before the landmark, including validation and earlier test frames.

**Impact:** no future leakage, but the estimand is online-refit forecasting. The generic `METHODOLOGY.md` sentence “Test frames are touched once, for final metrics” is false for this analysis. The final audit’s “causal window” language is accurate; the generic documentation is not.

### C05 — B6 is a coefficient ablation, not a refitted restriction (low-medium)

**Location:** `src/neurobranch/kato_common_state/models.py:399-403`; `src/neurobranch/kato_interactions/models.py:161-178`.

**Finding:** B6 zeros B3’s off-diagonals but inherits the fitted coefficients and recomputes the intercept from the local window. It is not a refitted diagonal model.

**Impact:** B6 is diagnostic only. B4m and B5 are the clean restrictions. The final audit states this correctly.

### C06 — Numeric trace labels are not cell identities (low)

**Location:** `src/neurobranch/data/kato.py:184-252`.

**Finding:** unnamed traces are retained as predictors with numeric labels. They are not aligned across recordings.

**Impact:** all analyses are within-recording. No cross-animal neuron alignment is possible for the unnamed traces. Named-neuron analyses cover only the identified subset (31–47 per recording).

### C07 — `stateParams` is not row-aligned (low)

**Location:** HDF5 `stateParams`.

**Finding:** session 1 has 135 traces but 107 `stateParams` rows; session 3 has 125 traces but 126 rows. The field is unused.

**Impact:** no effect on current scores; a future user must not assume a row-wise join.

### C08 — Coverage is marginal, not joint (medium)

**Location:** `src/neurobranch/probabilistic/metrics.py:109-120`.

**Finding:** `coverage_95` averages per-dimension marginal coverage. It is not joint 10-D coverage.

**Impact:** both models are less calibrated than the marginal number suggests. The B3/B5 comparison is still valid as a marginal diagnostic, but “95% coverage” must be qualified.

---

## 5. The two bootstrap targets

It is easy to read the final audit table and the master-review table as contradictory. They answer different questions.

| Target | Construction | Block 5 | Block 20 |
|---|---|---:|---:|
| Conditional on these five recordings | moving-block within session, equal pooling, no session resampling | `[+0.075, +0.403]` | `[+0.092, +0.294]` |
| Generalization across recordings | resample the five recordings, then moving blocks within each | `[-0.041, +0.479]` | `[-0.131, +0.440]` |
| Equal-weight session means | t interval over five session means | `[-0.064, +0.587]` | — |

The first is the final audit’s primary interval. The second is the one relevant to “would this replicate in another five animals?” The third is a simple check that does not model within-session precision.

The correct reporting is to give both, label the target, and not choose the one that looks better. The animal-level interval includes zero. That is the central inferential fact.

---

## 6. Independent numerical check

`reports/master_review/independent_model_check.json` reimplements:

- ridge VAR;
- factor PCA and factor VAR;
- B3 and B5 rollouts;
- energy and MSE;
- three landmarks from sessions 0, 2 and 4.

Results:

| Case | B3 energy abs. diff | B5 energy abs. diff | B3 MSE abs. diff | B5 MSE abs. diff |
|---|---:|---:|---:|---:|
| session 0, landmark 2690 | `4.17e-7` | `3.21e-7` | `4.22e-7` | `3.42e-7` |
| session 2, landmark 2624 | `1.14e-7` | `5.78e-7` | `2.80e-7` | `7.46e-7` |
| session 4, landmark 2592 | `6.56e-8` | `1.69e-8` | `1.03e-7` | `1.16e-8` |

Maximum absolute difference: `7.46e-7`. The tolerance is `2e-5`, justified by PCA sign/rounding and float32-to-float64 paths. This is an arithmetic reproduction, not an independent scientific replication.

At session 2, landmark 2624, B3 is much worse than B5:

```text
B3 energy 7.065
B5 energy 4.454
```

This is the session that reverses the average. The independent check reproduces it, so the reversal is a real feature of the fitted models, not a transcription error.

---

## 7. What the code does not do

- It does not select hyperparameters on test.
- It does not fit the scaler or PCA on test.
- It does not use future frames.
- It does not interpolate missing frames.
- It does not align numeric traces across recordings.
- It does not identify direct interactions from the VAR.
- It does not validate a biological perturbation.
- It does not produce a calibrated joint predictive distribution.
- It does not provide an animal-level generalization interval.

These absences are not all defects. Some are deliberate scope limits. The problem is when a document describes a narrower object than the claim requires.

---

## 8. Tests and reproduction

Focused tests:

```powershell
python -m pytest -q tests/master_review/test_kato_bounds.py
```

Six tests pass:

1. H100 targets are inside each recording.
2. Landmark counts are not 35 per session.
3. Frozen landmarks match the independent set.
4. No target crosses a split boundary at any reported horizon.
5. Documentation defects are recorded, not hidden.
6. The animal-level interval is not hidden by the landmark bootstrap.

Lint:

```powershell
python -m ruff check scripts/master_review tests/master_review
```

The tests are in [`tests/master_review/test_kato_bounds.py`](../tests/master_review/test_kato_bounds.py). They recompute bounds from HDF5 and do not import the project split helper for expected values.

---

## 9. Code-audit verdict

The implementation is careful, causal and numerically reproducible. The important defects are not coding errors in the narrow sense:

1. the primary interval is conditional, not animal-level;
2. B3 and B5 are not capacity- or adaptivity-matched;
3. the energy contrast is confounded with calibration/spread;
4. the rolling fit uses test-past, contradicting the generic “test touched once” statement;
5. the coverage number is marginal, not joint.

A code review that stopped at “tests pass” would miss all five. A scientific review that stopped at “the code is causal” would miss the difference between no future leakage and a valid animal-level claim.
