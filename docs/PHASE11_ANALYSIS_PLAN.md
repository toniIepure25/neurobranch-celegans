# Phase 11 analysis plan

Status: written before the Phase 11 sensitivity artifacts were generated.
This is an internal analysis plan, not a preregistration of a new experiment and
not a systematic review.

## 1. Question

Phase 10 estimated a pooled reliability ceiling of 0.833 at 22 evaluation
animals. That estimator used every responder in the Randi atlas, including
responders observed in only one animal. Does the ceiling survive (a) requiring
responders to be observed in most animals and (b) resampling animals and
responders together?

## 2. Estimands

| Symbol | Estimand |
|---|---|
| `R_fold` | mean over targets of `signal / (signal + (noise + interaction)/22)` |
| `R_single` | mean over targets of the same expression at `n_eval = 1` |
| `r_half` | Pearson correlation between target profiles estimated on disjoint animal halves, per target, averaged over splits |

`R_fold` and `R_single` are measurement ceilings, not causal effects and not
prediction accuracy.

## 3. Gates

### G1 — coverage sensitivity

Compare `R_fold` at minimum responder coverage 0.0, 0.3, 0.5, 0.8 and 1.0.

- **Pass:** the 0.8-coverage estimate is within 0.10 of the Phase 10 estimate.
- **Fail:** the difference exceeds 0.10. Then the Phase 10 pooled ceiling is
  reported as coverage-sensitive and is not used as a headline.

A failure is a result. The estimator is not re-tuned to recover 0.833.

### G2 — extrapolation

Fit components on animal half A, predict `sqrt(rel_A * rel_B)`, and compare with
the observed cross-half profile correlation at coverage 0.8.

- **Pass:** median absolute error <= 0.10 and |median bias| <= 0.05.
- **Fail:** the curve's extrapolated region is labelled "not validated".

### G3 — shrinkage

Empirical-Bayes logit shrinkage of per-target single-animal reliability.

- **Pass:** median half-split Spearman >= 0.30 with 95% interval lower bound
  > 0.10.
- **Fail:** ranking is withdrawn; only aggregate claims remain.

### G4 — two-way interval coverage

Synthetic coverage study with fixed target profiles and 100 replicates.

- **Pass:** coverage in [0.93, 0.97].
- **Conservative:** coverage > 0.97 with a reported width. This is not
  anti-conservative, but it is not called calibrated.
- **Fail:** coverage < 0.93; the interval is labelled anti-conservative and is
  not used for claims.

## 4. Multiplicity

Phase 11 adds no positive inferential claims. It audits existing claims and
reports sensitivity. No new p-value is interpreted without a family definition.

## 5. Decision rules

- If G1 fails, Phase 10's `phase10-fold-ceiling` claim remains a historical
  artifact but is marked coverage-sensitive in `CORRECTIONS.md`; the Phase 11
  card uses the coverage-aware value.
- If G2 fails, every figure that extends beyond the observed animal count is
  annotated.
- If G3 fails, no target ranking is published as a recommendation.
- If G4 is conservative, the interval is still reported with its width; no
  equivalence claim is derived from it.

## 6. Deviations

Any change after seeing results is appended to
[`PHASE11_DEVIATIONS.md`](PHASE11_DEVIATIONS.md) with the reason.
