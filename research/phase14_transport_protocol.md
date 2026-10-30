# Phase 14 transport protocol (not executed)

**Status:** `not_evaluable` in this checkout. No independent repeated-stimulation
atlas with the same estimand is available locally. This protocol defines what
would count as a transport test; it is not a result and does not support a
generalisation claim.

## Question

Does a measurement model selected on the Randi et al. atlas produce calibrated
reliability intervals on an independent preparation with repeated stimulation of
the same neuron pairs?

## Eligibility

- Independent animals and laboratory; no animal or stimulation event shared with
  the development atlas.
- Repeated stimulation of the same target with at least two events in a
  substantial fraction of animals; otherwise the one-event stratum dominates and
  the test is not informative about `R(22)`.
- Whole-brain or target-neuron calcium imaging with a documented response window,
  baseline and z-score procedure.
- Animal identifiers, target identifiers, responder identifiers, event counts and
  per-event responses available; not only aggregate means.
- A licence permitting the planned analysis.

## Frozen analysis

1. Apply the Phase 11 eligibility rule without refitting it: at least 6 animals,
   3 responders, 80% responder coverage, fold size 22.
2. Use the model and hyperparameters frozen before transport; no tuning on the
   transport set.
3. Primary metric: interval coverage for `R(22)` and rank uniformity by SBC on
   the transport masks.
4. Secondary: median absolute error of held-out profiles and the null q95.
5. Report the target population and event-count distribution before the estimate.

## Decision rules

| Outcome | Interpretation |
|---|---|
| Coverage in `[0.93, 0.97]`, rank KS p > 0.01, null q95 <= 0.15, PPC coverage >= 0.90 | Transport supported for this population only |
| Point estimate close but interval miscalibrated | No transport claim; the estimand may be similar but uncertainty is not |
| Coverage fails in the same direction as development | Systematic model misspecification, not sampling noise |
| Fewer than 100 eligible targets or mostly one-event cells | `not_evaluable`, not a negative transport result |

## What would not count

- Reusing Randi animals in a new split.
- Behavioural data from Atanas et al. (different estimand, no matched
  stimulation).
- A synthetic simulation using the same generative model as the fit.
- A point estimate within 0.05 without interval calibration.

## Current status

`not_evaluable`. No transport dataset was accessed, so no transport claim is
made in `docs/CLAIMS.md` or the manuscript draft.
