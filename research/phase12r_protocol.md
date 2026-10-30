# Phase 12R protocol: calibration-harness repair and reanalysis

**Status:** preregistered correction, frozen before the Phase 12R code changes and
before the corrected run. This is not a new confirmation study and not a
systematic review. It corrects a code-to-model mismatch found after Phase 12 was
reported.

**Frozen on:** 2026-10-30
**Branch:** `research/phase12r-calibration-repair`
**Base commit:** `232d236`
**Supersedes:** Phase 12 calibration evidence, not the Phase 11 descriptive bound.

## 1. Reason for the correction

Phase 12 reported that the hierarchical interval failed its calibration gates.
A post-report audit initially suspected that the simulation path discarded the
known cell-level sampling variance. That suspicion was tested directly before
any Phase 12R result was produced and **was not reproduced**:
`_cells_from_template()` returns per-cell summaries, and
`fit_hierarchical_cells()` passes their `var / n` into the likelihood without
recomputing them through `cell_summaries()`. The known-variance regression test
is retained as a guard, but Phase 12 is not invalidated by a variance-loss bug.

The audit did find a real alignment defect. `prepare_templates()` returns
targets that are already animal-centered and standardized; the SBC then
simulates from those values, and `fit_hierarchical_cells()` centers the
simulated rows again. The SBC truth formula, however, is the uncentered
variance-component expression. Simulation and truth therefore describe
different transformations. Phase 12R must align the simulation, the fit, and
the truth before interpreting any gate.

The audit also found that the Phase 12 prose said `min_events=2`,
`min_animals=2`, and two responders, while the executed code used
`min_events=1`, `min_animals=6`, and three responders. Phase 12R fixes the
parameter record and treats `min_events=2` as a sensitivity, not a silent
replacement.

Phase 12's real-data point estimate is not changed by any of these issues. The
question is whether its calibration conclusion survives an aligned reanalysis;
it is not assumed either way.

## 2. Question

After passing the known sampling variance and applying the same centering and
standardization to simulation and fit, does the hierarchical measurement model
reproduce the coverage-aware pooled reliability and ceiling at 80% responder
coverage, and does its uncertainty satisfy the preregistered calibration gates?

The target-level question remains unanswered unless every gate passes. No target
ranking, animal-budget guarantee, or calibrated interval is published otherwise.

## 3. Estimand and model

For target $r$, animal $a$, and responder $c$:

$$
y_{ar} = \mu_r + u_{ar} + e_{ar},
\quad \mu_r \sim N(0,\sigma_r^2),
\quad u_{ar} \sim N(0,\tau_r^2),
\quad e_{ar} \sim N(0,v_{ar}).
$$

$v_{ar}$ is the known sampling variance of the cell mean, computed as the
within-cell variance divided by the event count. The reliability of a profile
averaged over $n$ animals is

$$
R_r(n) = \frac{\sigma_r^2}{\sigma_r^2 + (\tau_r^2 + \nu_r)/n},
\qquad
\nu_r = \frac{1}{|\text{observed cells}|}\sum_{a,c} v_{ar}.
$$

The pooled estimate is the median of $R_r(22)$ across eligible targets; the
pooled ceiling is the median of $\sqrt{R_r(22)}$. The model is conditional on
the observed coverage pattern. It is not causal and not a digital twin.

## 4. Fixed primary configuration

| Parameter | Value |
|---|---|
| Responder coverage | 0.80 |
| `min_events` | 1 (comparability with Phase 11; 2 is a sensitivity) |
| `min_animals` | 6 |
| Minimum responders | 3 |
| `n_eval` | 1, 22 |
| Animal centering | yes, as part of the fitted model |
| Standardization | pooled finite-value SD |
| Prior | $\sigma,\tau \sim \mathrm{HalfNormal}(1)$ on the standardized scale |
| Posterior | deterministic 2-D grid, 80 x 80, log bounds $[-6,\log 20]$ |
| Posterior draws | 2500 |
| Fit seed | 12 |
| Resampling unit | animal |

`min_events=1` is retained because Phase 11 and the Phase 12 real fit used it;
the analysis-plan text that said 2 was wrong. The `min_events=2` variant is a
reported sensitivity, not a replacement chosen after seeing the result.

## 5. Simulation-based calibration

- 1000 replicates, 3 real-target templates per replicate (3000 fits).
- Parameters drawn from the fitting prior: $\sigma,\tau \sim \mathrm{HalfNormal}(1)$.
- Real target masks and known per-cell sampling variances are preserved.
- The simulated data are passed through the same centering and standardization
  as the fitted model. The truth is computed by Monte Carlo from that same
  transformed generative process; the uncentered closed-form formula is not
  used as truth for a centered fit.
- 2000 posterior draws on an 80 x 80 grid per fit.
- Rank statistic uses a uniform tie-breaking rule. Coverage is reported at
  $n=22$ with a Wilson 95% interval.
- Report the number of failed fits, grid-edge mass, and effective grid size.

## 6. Held-out predictive check and adversarial controls

- PPC: 20 animal-level splits, 20% held out, up to 40 targets per split, 1200
  draws, 60 x 60 grid, seed 41. Observed and predicted profiles are centered and
  standardized identically.
- Negative control: within-animal responder-label permutation, 100 requested
  targets, 1200 draws, 50 x 50 grid, seed 43. Preserve each cell's known
  variance. Report requested, eligible, excluded, and the exclusion reasons.
- Prior sensitivity: HalfNormal scales 0.5, 1.0, 2.0; 1200 draws, 50 x 50 grid,
  seed 47. Report the range of pooled $R(22)$.

## 7. Gates

A single failure blocks adoption. The thresholds are unchanged from Phase 12;
G1d is an additional fail-closed diagnostic, not a relaxation.

| Gate | Requirement |
|---|---|
| G1a SBC coverage | 22-animal coverage in [0.93, 0.97] |
| G1b SBC rank | KS test of rank uniformity p > 0.01 |
| G1c grid edge | maximum edge posterior mass < 0.05 |
| G1d fit integrity | failed/edge-dominated fits <= 5% of attempted SBC fits |
| G2 held-out PPC | coverage >= 0.90 and median absolute error <= 0.10 |
| G3 negative control | median $R(22)$ <= 0.05 and q95 <= 0.15 |
| G4 prior sensitivity | range of pooled $R(22)$ across prior scales <= 0.05 |

## 8. Decision rule

Adopt the hierarchical posterior as the primary inferential model only if:

1. G1a--G1d, G2, G3, and G4 all pass; and
2. the hierarchical pooled estimate is within 0.05 of the Phase 11 reference
   0.314.

If any condition fails:

1. retain 0.560 as the only descriptive coverage-aware ceiling;
2. do not publish a calibrated reliability interval;
3. do not rank targets or recommend an animal budget from the posterior;
4. report the failed gates and the negative-control result as a model
   limitation;
5. do not tune the prior, grid, filter, seed, or sample size to recover a pass.

If the gates pass, 0.560 remains reported as the Phase 11 descriptive reference;
the hierarchical interval may be reported as calibrated under this protocol.

## 9. Deviations and exclusions

Every deviation from this protocol is recorded before interpretation in
`research/phase12r_deviations.md`. Targets excluded by the declared eligibility
rules are reported with reasons. No target is excluded because its estimate is
inconvenient.

The input trial table is referenced by SHA-256 and is not redistributed. No new
data are acquired for Phase 12R. This protocol does not support causal,
whole-brain-mechanism, foundation-model, or validated-digital-twin claims.
