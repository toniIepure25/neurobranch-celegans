# Phase 10 analysis plan: metric-aligned ceilings

**Status:** internal analysis plan, written after the Phase 9 audit and before
the Phase 10 recomputation. It is **not** a registered report and it is not a
systematic review. Its purpose is to fix the estimand for each metric before
looking at the corrected numbers, because the Phase 9 correction was itself
prompted by a discrepancy between the metric and the ceiling.

## 1. Why a new phase

Phase 9 paired every Phase 6-8 metric with a single-animal ceiling of 0.473.
Code inspection showed that Phase 6-8 pair Spearman is computed on pair means
**pooled over the animals in a test fold** (about 22 animals), not on individual
animals. The correct ceiling for a pooled mean is higher, and it depends on the
number of animals pooled. Phase 9 therefore overgeneralised a valid
individual-level statement to pooled metrics.

A second defect was found in the same audit: the pooled sampling variance in
`_profile_from_cells` divided by the total animal count squared instead of the
per-responder count squared. Responders measured in fewer animals were
under-corrected, inflating the signal component. Phase 10 fixes both.

## 2. Estimands

| Facet | Estimand | Formula | Used for |
| --- | --- | --- | --- |
| Repeat | Same-animal repeated stimulation | variance-corrected ratio across stimulations | individual repeatability |
| Single animal | One animal's profile | `signal / (signal + noise + interaction)` | per-animal claims |
| Pooled fold | Mean profile over `n` animals | `signal / (signal + (noise + interaction)/n)` | Phase 6-8 pair metrics |
| Pooled all | Mean profile over all observed animals | same, `n = n_animals` | descriptive only |

`n = 22` is the Phase 6-8 fold size and is the primary pooled estimand. Values
for other `n` are reported as a curve, with `extrapolated = True` whenever
`n` exceeds the observed animal count for a target.

## 3. Hypotheses

- **H10.1 (descriptive).** The pooled reliability at 22 animals is higher than
  the single-animal reliability. This is an arithmetic consequence of the model,
  not an inferential claim.
- **H10.2 (stability).** Per-target single-animal reliability estimated on two
  random halves of the animals has a Spearman correlation above 0.3. This tests
  whether target-level reliability estimates are stable enough to rank targets.
- **H10.3 (summary robustness).** The fold-level pooled ceiling computed from
  `response_mean`, `response_integral` and `response_residual` has a
  target-level Spearman correlation above 0.8 against `response_mean`.
- **H10.4 (cluster uncertainty).** The animal jackknife 95% interval for the
  mean fold-level reliability has a width below 0.2. Wider intervals mean the
  aggregate is not precisely estimable from this design.
- **H10.5 (sample-size design).** The number of animals needed for a target
  reliability of 0.8 is finite for the majority of targets, and its median is
  reported as a design budget.

## 4. Decision rules

- H10.2 fails: target-level reliability rankings are not reproducible, so they
  must not be used to prioritise targets. Only the aggregate ceiling is
  reported as stable.
- H10.3 fails for a summary: that summary is not interchangeable with the
  primary one, and any conclusion that depends on it is flagged.
- H10.4 fails: report the point estimate with an explicit warning that the
  aggregate interval is wide; do not upgrade any claim.
- H10.5 is descriptive; the calculator is not used to justify a biological
  claim.

## 5. Deviations from the approved Phase 10 plan

- The EIG target-selection portfolio and the Bayesian SBC rank test were
  replaced by a ceiling-targeted design calculator and a **frequentist
  simulation coverage study**. The ceiling analysis showed that the binding
  design variable is the number of evaluation animals, not target selection.
  The calculator and coverage study are documented in `docs/DESIGN_ENGINE.md`.
- The jackknife interval is centred on the raw estimate with a normal
  approximation; the bias-corrected jackknife estimate was numerically unstable
  and is reported only as a diagnostic (`logit_bias`).
- `response_z` is not used as a primary sensitivity because its scale is
  dominated by extreme values; it is winsorised only if reported.

## 6. What would falsify the correction

If the pooled formula, recomputed from the per-target components, did not
reproduce the observed fold-level pair reliability in a held-out simulation,
the metric alignment would be wrong. The synthetic recovery tests in
`tests/benchmark/test_variance.py` and `tests/design/test_design.py` cover the
formula; they do not validate the biological model.
