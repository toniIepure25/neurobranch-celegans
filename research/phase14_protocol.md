# Phase 14 protocol: assumption-robust reference and zero-inflated reliability

**Status:** preregistered analysis plan, frozen before the Phase 14 implementation
is run on real data. This is not a confirmation study, a causal analysis, or a
systematic review.

**Frozen on:** 2026-10-31
**Branch:** `research/phase14-zinb-reference`
**Base commit:** `eedd882` (`v1.3.1` plus CI hardening)
**Supersedes:** the Phase 13 comparison reference only. Phase 13 remains a
negative result; its SBC, null-control, and PPC failures are not reinterpreted.

## 1. Reason for Phase 14

Phase 13 separated two defects in the Phase 12R model and rejected both repairs.
It also exposed a defect in the reference used to judge them:

- `cell_summaries` fills the sample variance of a one-event cell with `0.0`.
  Phase 11's `profile_components_by_target(..., min_events=1)` therefore treats
  79% of observed cells as noiseless when computing `0.314 / 0.560`.
- Phase 13 M2 replaces that value with `omega^2 / n`. The frozen Phase 13 G6 gate
  compared an M2 fit at `min_events=1` with a Phase 11 reference that uses a
  different variance treatment. G6 was therefore not a like-for-like truth test.

Phase 13 still fails independently: SBC coverage/rank, both null controls, and
the missing PPC. Phase 14 asks two separate questions:

1. Is the descriptive reference `0.560` robust to the one-event variance
   treatment?
2. Does a zero-inflated (spike-and-slab) model make `sigma = 0` credible under
   the permutation null and pass every calibration gate?

No target ranking, animal-budget guarantee, calibrated interval, causal effect,
or digital-twin claim is published unless every gate below passes.

## 2. Fixed input and reference

| Item | Value |
|---|---|
| Input | `reports/phase3/randi_trial_table_phase3.csv.gz` |
| Response | `response_mean` |
| Animal unit | `dataset_id` + animal identifier used by Phase 11 |
| Coverage | 0.80 |
| `fold_size` | 22 |
| `min_animals` | 6 |
| Minimum responders | 3 |
| Phase 11 reference targets | 126 |
| Phase 11 `R(22)` | `0.3136077301426884` |
| Phase 11 ceiling | `0.5600069018705827` |
| Standardization | pooled finite-value SD, as Phase 11 |

The input SHA-256 is recorded in the manifest. Raw data are not redistributed.

## 3. Phase 14A: reference audit

### Estimators

All estimators use `cell_summaries` and `profile_components_by_target`; defaults
are unchanged.

| Treatment | `n = 1` cell variance | `n >= 2` cell variance |
|---|---|---|
| V0 historical | `0.0` | sample variance |
| V1 primary audit | `omega^2 / n` | sample variance |
| V1h sensitivity | `omega^2 / n` | `omega^2 / n` |
| V2 filter | cell dropped (`min_events=2`) | sample variance |

`omega^2` is the **median** of the observed within-cell sample variances over
cells with `n >= 2` and `s^2 > 0`. This is a robust central estimate: the raw
cell-variance distribution is extremely heavy-tailed (99th percentile > 80,
maximum > 10^6 on the response scale), so a likelihood plug-in on the raw scale
is dominated by a handful of outlier cells and is not a valid summary of the
typical event variance. The median is frozen before the full run. A
HalfNormal(1) scaled-chi-square posterior on the raw scale is reported only as a
failed numerical diagnostic, not as a candidate reference.

The primary comparison is V1 versus V0. V1h tests whether the result depends on
retaining observed `n >= 2` sample variances; V2 tests the historical
`min_events=2` filter. A 99% winsorized-mean variant is reported as an
outlier-sensitivity diagnostic; it is not used for the A2 decision because it is
not a robust central estimator of the event variance.

### Summaries

- pooled `R(22)` and ceiling for V0/V1/V1h/V2;
- animal-cluster bootstrap 95% percentile intervals, 2,000 resamples, seed 61;
- common-target-set and all-eligible-target results;
- target-level Spearman correlation and top-20 overlap between V0 and V1;
- coverage sensitivity at 0.50/0.80/1.00;
- `min_events=2` target count and estimate.

### Decision rule

- **A1:** V0 must reproduce `0.3136077301426884` and `0.5600069018705827` within
  `1e-6`. If not, the audit implementation is wrong and the model run does not
  start.
- **A2:** if `|V1 - V0| > 0.05`, `0.560` is declared **not assumption-robust**.
  The V1 descriptive estimate and its bootstrap interval replace it in the
  headline, claims, and manuscript.
- If `|V1 - V0| <= 0.05`, `0.560` is retained as a descriptive bound.
- The audit does not create a calibrated interval, target ranking, or adoption.

## 4. Phase 14B: model M4

### Estimand

For target `t`, animal `a`, responder `r`:

```
z_t ~ Bernoulli(pi0), fixed pi0 = 0.5
mu_r | z_t = 1 ~ N(0, sigma^2)
mu_r | z_t = 0 = 0
u_ar ~ N(0, tau^2)
e_ar ~ N(0, v_ar)
v_ar = omega^2 / n_ar
```

The model is fit after the Phase 13 projection-consistent centering:

```
v'_ar = v_ar * (1 - 1 / R_a)
nu' = mean_ar(v'_ar)
```

Reliability of a profile pooled over `n` animals is

```
R(n) = sigma^2 / (sigma^2 + (tau^2 + nu') / n)
```

so `R(n) = 0` exactly when `sigma = 0`.

### Priors and computation

| Parameter | Value |
|---|---|
| `pi0` | 0.5 fixed |
| `sigma` slab | HalfNormal(1) |
| `tau` | HalfNormal(1) |
| `omega` | HalfNormal(1), Phase 13 event-variance posterior |
| `(sigma, tau)` grid | 80 x 80, log bounds `[-6, log 20]` |
| Spike | exact point mass at `sigma = 0` |
| Event grid | 200 points, `[1e-4, 10]` |
| Posterior draws | 2,500 |
| Fit seed | 12 |
| Event posterior draws | 5, propagated as sensitivity |

The discrete spike and the continuous slab are normalized with a log-sum-exp
evidence calculation. `pi0` is not estimated. A pilot may only detect runtime or
numerical bugs; it cannot select priors or thresholds.

### Calibration design

- **SBC:** 1,000 replicates x 3 real-target templates, 3,000 fits. Truth draws
  `z ~ Bernoulli(0.5)`, `sigma ~ HalfNormal(1)` when `z=1`, `sigma=0` otherwise,
  `tau ~ HalfNormal(1)`, and `omega` from its prior. Masks, event counts, and
  sampling variances are preserved. Simulation, fit, and truth use the same
  centering and standardization.
- **Rank statistic:** randomized tie-breaking for the discrete `R = 0` mass;
  KS test with `p > 0.01` required. Report the tie fraction and a Cramér-von
  Mises statistic descriptively.
- **Held-out PPC:** 20 animal-level splits, 20% held out, up to 40 targets per
  split, 1,200 draws, 60 x 60 grid, seed 41. Observed and predicted profiles are
  centered and standardized identically. Coverage and median absolute error are
  computed at the target level.
- **Negative control:** within-animal responder-label permutation, 100 requested
  targets, 1,200 draws, 50 x 50 grid, seed 43. Preserve event counts and sampling
  variances. Report requested, eligible, excluded, and exclusion reasons.
- **Sign-flip control:** flip each animal's centered profile sign with probability
  0.5; same summaries.
- **Prior sensitivity:** slab scales 0.5/1.0/2.0 and `pi0` 0.25/0.5/0.75;
  1,200 draws, 50 x 50 grid, seed 47.
- **Event-variance recovery:** posterior coverage of the true `omega^2` in
  `[0.90, 0.98]`.
- **One-event stratum:** coverage in `[0.90, 0.98]` when the `n=1` fraction is at
  least 0.5.

### Gates

| Gate | Requirement |
|---|---|
| G1a SBC coverage | 22-animal coverage in `[0.93, 0.97]` |
| G1b SBC rank | randomized-rank KS `p > 0.01` |
| G1c grid edge | continuous-component maximum edge mass `< 0.05` |
| G1d fit integrity | failed or edge-dominated fits `<= 5%` of attempted SBC fits |
| G1e one-event stratum | coverage in `[0.90, 0.98]` when `n=1` fraction `>= 0.5` |
| G2 held-out PPC | coverage `>= 0.90`; median absolute error `<= 0.10`; must be run |
| G3 negative control | median `R(22) <= 0.05`; q95 `<= 0.15` |
| G3b sign-flip control | median `R(22) <= 0.05`; q95 `<= 0.15` |
| G4 prior sensitivity | range of pooled `R(22)` across prior scales `<= 0.05` |
| G5 event-variance recovery | coverage of `omega^2` in `[0.90, 0.98]` |
| G6 reference agreement | `|R(22) - R_ref(V1)| <= 0.05` |
| G7 spike calibration | for true `z=0`, median posterior `P(z=0) >= 0.80` |

G6 uses the Phase 14A V1 reference. V0 is reported descriptively. All gates are
conjunctive. A borderline G1a is not relabelled. A missing PPC is a failure, not
an omission.

## 5. Decision rule

Adopt M4 only if every gate passes.

If adopted:

1. report a calibrated pooled `R(22)` interval and the reference audit;
2. retain the non-causal claim boundary;
3. do not publish target rankings, because no ranking-stability gate exists;
4. release `v1.4.0`.

If any gate fails:

1. decision `not_adopted_gate_failures`;
2. publish the failed gate, model version, PPC result, and diagnosis;
3. retain `0.560` or replace it only with the Phase 14A V1 descriptive estimate;
4. publish no calibrated interval, target ranking, or animal-budget guarantee;
5. release `v1.3.2`;
6. do not tune the prior, grid, filter, seed, or sample size to recover a pass.

No second tuning cycle on the same data without a new protocol.

## 6. Deviations and exclusions

Every deviation is recorded in `research/phase14_deviations.md` before the full
run and before interpretation. A pilot may detect runtime, numerical, or
implementation bugs only. If a pre-registered computation is infeasible, the
protocol is amended and re-frozen before the full run with the exact change.

Excluded targets are reported with machine-readable reasons. Missing cells are
not imputed. Invalid variances fail closed. No target is selected because its
estimate is convenient.

## 7. Artifacts

Authoritative outputs are written under `reports/phase14/`:

- `reference_audit.json`, `reference_audit_by_target.csv`,
  `reference_audit_bootstrap.csv`;
- `summary.json` — estimates, gates, decision;
- `sbc.json` — ranks, coverage, strata, spike calibration;
- `heldout_ppc.json` — predictive check;
- `negative_control.json` — permutation null and exclusions;
- `signflip_control.json` — sign-flip null;
- `prior_sensitivity.json` — prior-scale audit;
- `variance_recovery.json` — event-variance coverage;
- `hierarchical_by_target.csv` — per-target summaries;
- `min_events_sensitivity.csv`, `coverage_sensitivity.csv`;
- `transport_status.json` — `not_evaluable` or a separate frozen protocol;
- `manifest.json`, `manifest.sha256`, `validation.json`.

Text artifact hashes are LF-normalized. The input table is referenced by SHA-256
and is not redistributed.

## 8. What Phase 14 cannot establish

- It cannot establish a causal effect of any neuron or stimulation.
- It cannot validate a digital twin, foundation model, or virtual perturbation.
- It cannot make a target ranking or animal-budget guarantee.
- It cannot generalise beyond the atlas, response window, preparation, and
  responder-coverage population.
- A passing calibration gate means the interval procedure is calibrated under
  the simulation model; it does not prove the model is biologically true.
- Behavioural transport is not measurement reliability transport. If no
  independent repeated-stimulation dataset exists, transport is `not_evaluable`.
