# Phase 15 protocol: reliability without extrapolation

**Frozen:** 2026-11-01, before the Phase 15 audit run.
**Branch:** `research/phase15-reliability`.
**Input:** `reports/phase3/randi_trial_table_phase3.csv.gz`, SHA256
`8186c96bcb33cdce15dc9ae88691e09fc596e42c5513310b6d6a636e4d5dd515`.
**Status:** pre-registered analysis plan. No result in this file is a finding.

## 1. Question

For the Randi repeated-stimulation atlas at 80% responder coverage, what
reliability is identified by the observed animal counts, what is a
counterfactual extrapolation to 22 animals, and what uncertainty and null bias
do those quantities carry?

This is a measurement-audit question. It is not a causal, transport, ranking or
animal-budget question.

## 2. Estimands

All three are reported together; none is a sole headline.

1. `R_obs = mean_t R_t(n_t)` over the 126 eligible targets, where `n_t` is the
   target's observed animal count and
   `R_t(n) = signal_t / (signal_t + (noise_t + interaction_t) / n)`.
2. `R_22 = mean_t R_t(22)` over the same targets. Targets with `n_t < 22` are
   flagged `extrapolated`; this is a design counterfactual, not an identified
   observed-data quantity.
3. `R_id = mean_{t: n_t >= 22} R_t(n_t)` over the identified subset. This is a
   different target population and is not a replacement for `R_obs`.

`R_t` is clipped to `[0, 1]`; `signal_t` is the non-negative moment estimate
already used by the Phase 11 estimator. Ceilings are `sqrt(R)` and are reported
only as derived descriptive quantities.

## 3. Variance treatments

The primary treatment is V1: a common one-event variance
`omega^2 = median(s^2)` over cells with `n >= 2` and `s^2 > 0`. V0 (historical
zero one-event variance) is reported as a sensitivity. V2 (`min_events=2`) is
not used as a correction; if reported, its different target population is
labelled. No post hoc filter change is allowed.

## 4. Uncertainty

Primary: delete-one-animal jackknife, conditional on the 80%-coverage responder
set. The aggregate estimand for a deleted animal is recomputed for every target
containing that animal; targets not containing it contribute their full-sample
value. For `R_obs` and `R_id`, the target's animal count is reduced by one in
the deleted-animal replicate; for `R_22`, the evaluation count stays at 22.
The standard error uses the finite-sample factor `(A - 1) / A`; the interval is
normal at 95%. BCa is a secondary diagnostic.

Cross-checks:

- Bayesian bootstrap with `Dirichlet(1, ..., 1)` weights over animals, 2,000
  draws, weighted moment estimator, same fixed target populations.
- Animal-half replication and leave-animal-out prediction of held-out half
  correlations.
- The two-way bootstrap is a diagnostic only. It is `invalid` when the point
  estimate is outside its percentile interval and is never a headline interval.

No interval is called calibrated unless the simulation in section 5 passes.

## 5. Calibration simulation

Use the real target sizes, animal/responder missingness masks, and cell sampling
variances. For each target, draw `mu_r ~ N(0, signal_t)`,
`u_ar ~ N(0, interaction_t)` for observed cells, and
`e_ar ~ N(0, sampling_var_ar)`; form `y_ar = mu_r + u_ar + e_ar`. Truth is the
same functional at the fixed target counts.

- 1,000 replicates for the primary simulation.
- 500 replicates for a heavy-tailed/heteroskedastic misspecification check.
- Coverage, bias, width and tail error for all three estimands.
- Pass: coverage in `[0.93, 0.97]` for each estimand at nominal 95%; otherwise
  the interval is `not_calibrated`.

## 6. Null controls

For each estimand, 1,000 replicates of:

1. responder-label permutation within each animal and target, preserving the
   observed missingness and event counts;
2. animal-level sign flip of each target profile.

Pass: null median `<= 0.05` and null q95 `<= 0.15`. Failure means no ceiling
claim.

## 7. Extrapolation and ranking

- Animal-half replication at coverage 0.8, 15 splits, seed 11.
- Pass: median absolute error `<= 0.10` and absolute bias `<= 0.05`; otherwise
  `R_22` is `counterfactual_only`.
- Ranking stability: Spearman correlation of target ranks across animal halves
  and top-10 Jaccard overlap. Pass: Spearman lower 95% CI `> 0.50` and Jaccard
  `> 0.60`; otherwise `no_ranking`.

## 8. Gates and decision

| Gate | Requirement |
|---|---|
| G1 estimand completeness | all three estimands, counts, extrapolation flags |
| G2 jackknife integrity | >= 95% valid LOO replicates, no NaN, counts match |
| G3 calibration | coverage 0.93--0.97 for each estimand |
| G4 null control | median <= 0.05, q95 <= 0.15 for each control |
| G5 extrapolation | MAE <= 0.10, abs bias <= 0.05 |
| G6 ranking stability | Spearman lower CI > 0.50, Jaccard > 0.60 |
| G7 variance robustness | abs(V1 - V0) <= 0.05 |
| G8 reproducibility | manifest, hashes, validator, clean worktree |

Decision labels are fail-closed: `calibrated_descriptive`,
`not_calibrated`, `counterfactual_only`, `no_ranking`, `not_evaluable`.

## 9. Stop rules

- No threshold, filter, seed, variance treatment, simulator or estimator change
  after seeing results.
- No M4 retuning and no new measurement model in Phase 15.
- No causal, transport, digital-twin, foundation-model, animal-budget or target
  ranking claim.
- A failed gate is reported as a result, not rescued by a different interval.
- The two-way bootstrap is not called calibrated.

## 10. Provenance

Every run records the input hash, git commit, dirty state, Python and package
versions, seeds, commands, and artifact hashes in `reports/phase15/manifest.json`.
Deviations are recorded in `research/phase15_deviations.md`.
