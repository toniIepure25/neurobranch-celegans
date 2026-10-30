# Phase 13 measurement-model methods

This note documents the Phase 13 candidates and their calibration. It is a
measurement-model methods record, not a causal or generative claim. The frozen
protocol is `research/phase13_protocol.md`; the results are in
`docs/PHASE13_RESULTS.md`.

## 1. The two defects

For target `t`, animal `a`, responder `r`:

```
y_ar = mu_r + u_ar + e_ar
mu_r ~ N(0, sigma^2)
u_ar ~ N(0, tau^2)
e_ar ~ N(0, v_ar)
```

The Phase 12R fit centers each animal profile,

```
z_ar = y_ar - mean_r(y_ar),
```

but leaves `v_ar` unchanged. For equal cell variances, `Var(z_ar) = v * (1 -
1/R_a)`; unequal variances also induce off-diagonal covariance. The likelihood
treats the centered noise as `diag(v)`, which is too large by the factor
`1/(1 - 1/R_a)`.

Separately, 32,427 of 41,086 observed cells (79%) have one event and zero sample
variance. The Phase 12R fit floors those cells at `1e-8`. That is not a measured
variance and makes one-event cells effectively noiseless.

## 2. Candidate models

### M0: Phase 12R reference

Center first, standardize by the pooled SD of centered profiles, fit the
unchanged diagonal likelihood with the `1e-8` floor. M0 reproduces
`0.3035306473018537`. It is a historical control and is never adoptable.

### M1: projection-scaled known `v`

Center `y`, and for each animal with `R_a` observed responders set

```
f_a = 1 - 1 / R_a
v'_ar = v_ar * f_a
nu' = mean_ar(v'_ar).
```

The fit uses `v'` and `nu'`. This is a first-order diagonal correction; it does
not model the off-diagonal covariance induced by unequal variances. M1 is a
diagnostic, not the primary model.

### M2: event-level marginal variance

For cells with `n_ar >= 2` and sample variance `s^2_ar > 0`:

```
(n_ar - 1) * s^2_ar / omega_e^2 ~ ChiSquare(n_ar - 1).
```

A HalfNormal(1) prior is placed on `omega_e` on the standardized scale, and the
posterior is evaluated on a 200-point log grid over `omega_e`. The posterior
median `omega_e^2` is the plug-in. Then

```
v_ar = omega_e^2 / n_ar
```

for every observed cell, including `n_ar = 1`. No floor is used. The M1
projection factor is applied after constructing `v`.

This is a **marginal, homogeneous** event-variance model. It does not model
cell-level variance heterogeneity or detection. Protocol deviation D2 records
the narrowing; no event-variance draws are propagated through the target-level
fit.

## 3. Reliability estimand

After centering and variance transformation, the fitted model produces
`(sigma^2, tau^2)` on the centered scale. For `n` animals:

```
R(n) = sigma^2 / (sigma^2 + (tau^2 + nu)/n),   nu = mean_ar(v_ar).
```

The pooled estimate is the median of `R(22)` over eligible targets; the ceiling
is `sqrt(R(22))`. This is conditional on the observed coverage pattern and is not
a causal effect.

## 4. Posterior computation

The target-level posterior is a deterministic 80 x 80 grid over
`(log sigma, log tau)` from `-6` to `log 20`, with 2,500 draws obtained by
sampling grid cells proportional to posterior mass and adding within-cell
jitter. The fit reports boundary mass (`grid_edge_mass`) and effective grid size
(`grid_n_eff`). Target seeds are derived from the frozen seed 12 and the target
name. Grid discretization is a limitation, not a convergence guarantee.

## 5. Calibration

- **SBC:** M1 uses 1,000 replicates x 3 real-target templates; M2 uses 600 x 3.
  Parameters are drawn from the fitting prior; real masks, event counts, and
  known variances are preserved; simulation and fit use the same projection.
  Ranks and 95% coverage are computed at `n = 1, 22`, with a KS uniformity test
  and Wilson interval. M2 additionally reports coverage of the true
  `omega_e^2`.
- **Negative control:** within-animal responder-label permutation; 100 targets,
  1,200 draws, 80 x 80 grid, seed 43. Event counts and cell variances are
  preserved.
- **Sign-flip control:** each animal's profile is multiplied by `-1` with
  probability 0.5 before fitting; same summaries, seed 44.
- **Prior sensitivity:** HalfNormal scales 0.5, 1.0, 2.0; 1,200 draws on a
  50 x 50 grid, seed 47.
- **Held-out PPC:** not run. After both null controls failed, no valid M2
  animal-holdout check was implemented. It is reported as `not_run` and counted
  as a gate failure (deviation D4).

## 6. What the calibration can and cannot show

A pass would show that the interval procedure is calibrated **under the
simulation model**; it would not prove the model is biologically true. A failure
can be caused by the likelihood, the prior, the centering approximation, the
event-variance model, or the null construction. Phase 13's negative control is
the most interpretable failure: under a null that preserves marginal values and
coverage, M2 still reports median `R(22) = 0.1485`, so nonzero between-target
variance is being assigned to sampling noise.

## 7. Code map

| Purpose | Path |
|---|---|
| M0/M1/M2 fits and event-variance posterior | `src/neurobranch/benchmark/measurement.py` |
| SBC, null controls, prior sensitivity | `src/neurobranch/benchmark/measurement_calibration.py` |
| Event-keyed paired inference and BCa intervals | `src/neurobranch/benchmark/inference.py` |
| Full-run driver | `scripts/phase13_measurement.py` |
| Manifest and validator | `scripts/phase13_manifest.py`, `scripts/phase13_validate.py` |
| Tests | `tests/benchmark/test_measurement.py`, `tests/benchmark/test_inference.py` |

## 8. Methods anchors

The reliability-as-variance-ratio framing follows Shrout & Fleiss (1979; DOI
`10.1037/0033-2909.86.2.420`). The SBC design follows Modrak et al. (2025; DOI
`10.1214/23-ba1404`, PMID 41049725). These are methods anchors, not evidence
that this model is calibrated. No new systematic literature search was performed
for Phase 13; the negative result is internal to this atlas and this model.
