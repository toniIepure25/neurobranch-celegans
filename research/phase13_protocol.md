# Phase 13 protocol: projection-consistent reliability and event-level variance

**Status:** preregistered analysis plan, frozen before the Phase 13 implementation
is run on real data. This is not a new confirmation study, not a causal analysis,
and not a systematic review.

**Frozen on:** 2026-10-30
**Branch:** `research/phase13-measurement`
**Base commit:** `5dc4407` (`v1.3.0`)
**Supersedes:** the Phase 12R calibration evidence only. The Phase 11 descriptive
ceiling of 0.560 remains the reference bound.

## 1. Reason for Phase 13

Phase 12R found that the hierarchical reliability model failed its calibration
gates. Two defects remain that were not part of the Phase 12R correction:

1. **Centering changes the noise model.** `transform_target(center_animals=True)`
   subtracts each animal's responder mean from `y` but leaves `v` unchanged.
   For equal cell variances, the centered noise variance is
   `v * (1 - 1/R)`; unequal variances also induce non-zero covariances. The
   likelihood continues to treat the post-centering noise as `diag(v)`. The SBC
   truth applied the centering factor to `sigma^2` and `tau^2` but not to `nu`.
   This is an algebra-to-model mismatch, not a biological finding.

2. **The variance floor is not a measurement model.** 32,427 of 41,086 observed
   cells (79%) have zero within-cell variance because they contain one event and
   use the `1e-8` floor. The `min_events=2` sensitivity changes the estimate from
   0.304 to 0.199 and keeps only 36 targets.

Phase 13 asks whether these two defects explain the failed gates. It does not
assume that they do. If the corrected model still fails, that is the result.

## 2. Question

After applying a projection-consistent variance transformation and replacing the
variance floor with an event-level variance model, does the reliability model:

1. reproduce the 22-animal pooled reliability at 80% responder coverage;
2. pass SBC rank and coverage, held-out predictive checks, negative controls,
   prior sensitivity, and event-variance recovery; and
3. remain within 0.05 of the Phase 11 reference 0.314?

No target ranking, animal-budget guarantee, calibrated interval, causal effect,
or digital-twin claim is published unless every gate passes.

## 3. Estimand

For target `t`, animal `a`, and responder `r`:

```
y_ar = mu_r + u_ar + e_ar
mu_r ~ N(0, sigma^2)
u_ar ~ N(0, tau^2)
e_ar ~ N(0, v_ar)
```

`v_ar` is the sampling variance of the cell mean. Reliability of a profile
pooled over `n` animals is

```
R(n) = sigma^2 / (sigma^2 + (tau^2 + nu) / n)
nu = mean_ar(v_ar)
```

The pooled estimate is the median of `R(22)` over eligible targets. The ceiling
is `sqrt(R(22))`. This is a measurement model conditional on the observed
coverage pattern. It is not causal.

## 4. Fixed primary configuration

| Parameter | Value |
|---|---|
| Responder coverage | 0.80 |
| `min_events` | 1 (primary, comparable with Phase 11); 2 is a sensitivity |
| `min_animals` | 6 |
| Minimum responders | 3 |
| `n_eval` | 1, 22 |
| Standardization | pooled finite-value SD |
| Prior | `sigma, tau ~ HalfNormal(1)` on the standardized scale |
| Posterior | deterministic 2-D grid, 80 x 80, log bounds `[-6, log 20]` |
| Posterior draws | 2,500 |
| Fit seed | 12 |

`min_events=1` is retained because Phase 11 and the Phase 12 real fit used it.
`min_events=2` is reported as a sensitivity, never as a replacement.

## 5. Candidate models

All candidates use the same 126 eligible targets and the same coverage filter.
They are compared as paired analyses, not as independent discoveries.

### M0 - Phase 12R reference

Current `fit_hierarchical_target` with animal centering, unchanged `v`, and the
`1e-8` floor. M0 is a historical control and is never eligible for adoption.

### M1 - projection-scaled known variance

Center `y` as before. For each animal with `R_a` observed responders, transform
the sampling variance by the first-order centering factor

```
f_a = 1 - 1 / R_a
v'_ar = v_ar * f_a
nu' = mean_ar(v'_ar)
```

The fit uses `v'` and `nu'`; the truth simulation applies the same factor. This
corrects the omitted noise scaling for equal variances. It is explicitly a
diagonal approximation; unequal variances also induce off-diagonal covariance
that M1 does not model. M1 is therefore a diagnostic, not the final model.

### M2 - event-level variance model

Replace the floor with a marginal event-variance model. For each cell with
`n_ar >= 2` and positive sample variance `s^2_ar`, use the scaled chi-square
likelihood

```
(n_ar - 1) * s^2_ar / omega_e^2 ~ ChiSquare(n_ar - 1)
```

Cells with `n_ar = 1` contribute no sample-variance likelihood but are retained
in the mean likelihood with `v_ar = omega_e^2 / n_ar`. A weakly informative
prior is placed on `omega_e`; the prior is `HalfNormal(1)` on the standardized
scale unless the pilot shows a numerical problem, in which case the deviation is
recorded before the full run. The posterior median of `omega_e^2` is used as a
plug-in for the cell means, and its 2.5--97.5% interval is reported. Five
posterior draws of `omega_e^2` are propagated through the fit and pooled with
equal weights; if that propagation is numerically unstable, the median-only fit
is reported with the limitation stated.

`v_ar` is then `omega_e^2 / n_ar`, with no floor. The M1 centering factor is
applied after constructing `v`.

M2 is the primary candidate. It tests both defects together. The homogeneous
`omega_e` assumes a common event-level variance across cells; cell-level
heterogeneity is a stated limitation and a sensitivity, not a hidden assumption.

### M3 - animal-intercept sensitivity (optional)

If time permits before the full run, fit an uncentered animal-intercept model

```
y_ar = c_a + mu_r + u_ar + e_ar,  c_a ~ N(0, omega_c^2)
```

with `omega_c` fixed at a method-of-moments estimate and integrated as a
sensitivity. This model avoids the centering projection entirely. It is not part
of the adoption gate unless the protocol is amended before the full run.

## 6. Simulation-based calibration

- 1,000 replicates x 3 real-target templates (3,000 fits) for M0 and M1.
- M2 uses 600 replicates x 3 templates (1,800 fits) because each fit includes
  the event-variance grid; the reduced count is frozen before the run.
- Parameters drawn from the fitting prior: `sigma, tau ~ HalfNormal(1)`;
  M2 additionally draws `omega_e ~ HalfNormal(1)`.
- Real target masks, event counts, and known sampling variances are preserved.
- Simulation, fit, and truth use the same centering and standardization.
- Truth for M1 is computed with the scaled `nu'`; for M2 with the simulated
  `omega_e` and the observed event counts.
- 2,000 posterior draws on an 80 x 80 grid per fit.
- Rank statistic uses uniform tie-breaking. Coverage is reported at `n=22` with
  a Wilson 95% interval.
- Report failed fits, grid-edge mass, effective grid size, and the `n=1` stratum.

## 7. Held-out predictive check and adversarial controls

- **PPC:** 20 animal-level splits, 20% held out, up to 40 targets per split, 1,200
  draws, 60 x 60 grid, seed 41. Observed and predicted profiles are centered and
  standardized identically.
- **Negative control:** within-animal responder-label permutation, 100 requested
  targets, 1,200 draws, 50 x 50 grid, seed 43. Preserve event counts and sampling
  variances. Report requested, eligible, excluded, and exclusion reasons.
- **Sign-flip control:** flip the sign of each animal's centered profile with
  probability 0.5 before fitting; report the same null summaries.
- **Prior sensitivity:** HalfNormal scales 0.5, 1.0, 2.0; 1,200 draws, 50 x 50
  grid, seed 47. Report the range of pooled `R(22)`.
- **Event-variance recovery:** for M2, report posterior coverage of the true
  `omega_e^2` and the fraction of cells with `n=1`.

## 8. Gates

A single failure blocks adoption. Thresholds are unchanged from Phase 12; G1e,
G5, and the sign-flip control are additions, not relaxations.

| Gate | Requirement |
|---|---|
| G1a SBC coverage | 22-animal coverage in [0.93, 0.97] |
| G1b SBC rank | KS test of rank uniformity `p > 0.01` |
| G1c grid edge | maximum edge posterior mass `< 0.05` |
| G1d fit integrity | failed/edge-dominated fits `<= 5%` of attempted SBC fits |
| G1e one-event stratum | coverage in [0.90, 0.98] for targets with `n=1` fraction >= 0.5 |
| G2 held-out PPC | coverage `>= 0.90`; median absolute error `<= 0.10` |
| G3 negative control | median `R(22) <= 0.05`; q95 `<= 0.15` |
| G3b sign-flip control | median `R(22) <= 0.05`; q95 `<= 0.15` |
| G4 prior sensitivity | range of pooled `R(22)` across prior scales `<= 0.05` |
| G5 event-variance recovery | posterior coverage of `omega_e^2` in [0.90, 0.98] |

## 9. Decision rule

Adopt M2 as the primary inferential model only if:

1. G1a--G1e, G2, G3, G3b, G4, and G5 all pass; and
2. the pooled `R(22)` is within 0.05 of the Phase 11 reference 0.314.

If M2 fails but M1 passes every gate, M1 may be reported as a diagnostic model
with the diagonal-covariance limitation; it cannot support a calibrated interval
or a target ranking. If both fail, the Phase 12R negative result stands.

If any condition fails:

1. retain 0.560 as the only descriptive coverage-aware ceiling;
2. publish no calibrated interval, target ranking, or animal-budget statement;
3. report the failed gate, the model version, and the diagnosis;
4. do not tune the prior, grid, filter, seed, or sample size to recover a pass;
5. do not run a second tuning cycle without a new protocol.

## 10. Deviations and exclusions

Every deviation is recorded in `research/phase13_deviations.md` before the full
run and before interpretation. A pilot run may be used only to detect runtime,
numerical, or implementation bugs; pilot results are not used to choose
thresholds. If a pilot reveals that a pre-registered computation is infeasible,
the protocol is amended and re-frozen before the full run, with the reason and
the exact change recorded.

Excluded targets are reported with machine-readable reasons. Missing cells are
not imputed. Invalid sampling variances fail closed. No target is selected
because its estimate is convenient.

## 11. Artifacts

Authoritative outputs are written under `reports/phase13/`:

- `summary.json` - estimates, gates, decision;
- `sbc.json` - calibration ranks, coverage, strata;
- `heldout_ppc.json` - predictive check;
- `negative_control.json` - permutation null and exclusions;
- `signflip_control.json` - sign-flip null;
- `prior_sensitivity.json` - prior-scale audit;
- `variance_recovery.json` - event-variance posterior coverage;
- `hierarchical_by_target.csv` - per-target summaries;
- `min_events_sensitivity.csv` - filter sensitivity;
- `coverage_sensitivity.csv` - coverage curve;
- `manifest.json` and `manifest.sha256` - provenance.

The input table is referenced by SHA-256 and is not redistributed.

## 12. What Phase 13 cannot establish

- It cannot establish a causal effect of any neuron or stimulation.
- It cannot validate a digital twin or a foundation model.
- It cannot make a target ranking or animal-budget guarantee unless a separate
  ranking-stability gate is added in a future protocol.
- It cannot generalise beyond the atlas, response window, preparation, and
  responder-coverage population.
- A passing calibration gate means the interval procedure is calibrated under
  the simulation model; it does not prove the model is biologically true.
