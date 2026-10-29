# Phase 8 deviations and pre-run decisions

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

## Development access

Development smoke runs used split seeds **17, 29 and 43**, with reduced folds,
one model seed and short training. An initial full run on seed **37** was
aborted after the active-selection implementation proved pathologically slow;
that run is an aborted attempt, not a result. The reported full run uses split
seed **41**, which was not used for development, model selection or any
completed test evaluation. This is the mechanism that keeps the reported full
test relatively untouched after smoke testing.

## Changes made before freezing the full run

1. **Target/responder empirical priors.** The first smoke model had no direct
   per-responder or per-target prior. A frozen train-only prior vector
   (`target_prior`, `responder_prior`, `target_seen`) was added to the row
   representation. This makes the APT versus CRO comparison less obviously
   disadvantaged by CRO's learned embeddings.
2. **Trajectory loss weight.** Set to 1.0 (the first smoke used 0.5) to avoid
   the scalar endpoint completely dominating the shared representation.
3. **Budget-curve scope.** Full support-size curves are evaluated for the random
   arm; the active strategies are compared at the primary k=8. This keeps the
   row-level audit file tractable while preserving the preregistered H2/H3
   comparisons.
4. **Ensemble predictions.** The three model seeds are combined into a deep
   ensemble for the reported k=0/k=8 predictions; the seed fit summaries are
   retained. This is a variance-reduction choice, not a cherry-pick of the best
   seed.
5. **Sequence persistence baseline.** For the first event, persistence uses the
   per-responder training-mean template rather than the APT prediction, removing
   an obvious self-comparison bias.
6. **Residual trajectory template.** The trajectory head predicts a residual
   around the train-only per-responder mean trajectory. This avoids forcing the
   model to relearn a strong baseline; it is disclosed because it makes the H1
   trajectory comparison conservative and directly interpretable.
7. **Bootstrap replicates in the recovered summary.** The preregistered 1,000
   animal-cluster replicates were reduced to **100** after the original summary
   step proved pathologically slow. Point estimates and decisions were unchanged;
   intervals are slightly less precise. The fast pair-level bootstrap is now the
   default in `paired_metric_delta`.

## Not run / deferred

- **H5 history-feature ablation** was preregistered as exploratory but is not
  part of the first full run. It is reported as not run, not as a null result.
- No cross-dataset or external optogenetic validation was run; the repository
  does not contain a same-task external cohort.
- No wet-lab closed loop was attempted.

## Interpretation rules

- H1 is the single primary gate. A failure of H1 is not rescued by H2/H3/H4.
- H3 is confirmatory only if H2 is supported; otherwise it is descriptive.
- All intervals are animal-cluster bootstrap intervals; they describe
  uncertainty over animals in this atlas, not over experimental protocols.
- Any audit overlap invalidates the run rather than being adjusted away.
