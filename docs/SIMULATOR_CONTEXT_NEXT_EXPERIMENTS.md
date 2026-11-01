# Next experiments

Ordered by value, not by model complexity.

## P0 — fix the evidence base before more models

1. **Animal-disjoint evaluation.** Obtain recordings with confirmed animal identity and split by animal, not session. Report the number of animals, not the number of files.
2. **Independent oxygen schedules.** Randomize transition times/intervals across animals so time-since-transition and oxygen level are not collinear. Pre-register the number of independent transition events.
3. **Untouched confirmatory test.** Freeze the split and model class before looking at the final test block. Do not reuse a Phase 6 block and call it untouched.
4. **Intervention-compatible Gate 8.** Obtain targeted perturbation data with cell identity, dose, timing, and matched controls. Without it, keep Gate 7/8 NOT_EVALUABLE.

## P1 — model classes worth one controlled comparison

5. **Nonlinear but stable latent dynamics.** A small neural state-space model with an explicit stability/contractivity constraint, compared on exactly the same splits as the linear LDS. Do not run a broad architecture search.
6. **Better compositional inference.** If dLDS is revisited, use the authors' licensed code in an isolated environment and reproduce their synthetic benchmark first. Only then compare on these data. Do not call an alternating NNLS approximation a reproduction.
7. **Observation-model ablations.** Compare PCA, factor analysis, and a session-specific linear observation map with a shared latent transition. Report identifiability diagnostics; do not pool worm-specific numeric IDs.
8. **Calibration-aware transfer.** Evaluate zero-shot, calibration-adapted, and local models on the same calibration budget. Report the budget explicitly.

## P2 — uncertainty and branching

9. **Calibrated predictive intervals.** Replace the nearest-neighbour support diagnostic with a proper predictive log score and interval coverage on held-out animals.
10. **Branching only after dynamics pass.** Do not interpret branching statistics until autonomous rollout beats persistence on animal-disjoint data. Use process covariance, not injected observation noise, and pre-register the comparator set.

## P3 — perturbations

11. **Matched experimental validation.** Only after an intervention dataset exists, test whether model sensitivity ranks match measured effects. Use bounded amplitudes and report model dependence.
12. **No dose equivalence.** Keep model units separate from experimental dose. Do not optimize a large target set against an unvalidated model.

## Stop rules

- If autonomous rollout remains worse than persistence in a properly powered animal-disjoint test, stop the current latent/compositional family.
- If measured oxygen is indistinguishable from a time-since-transition control, do not claim a specific oxygen effect.
- If coefficient entropy collapses and the policy RMSE stays high, do not add operators; fix the inference or stop.
- If support coverage at the chosen quantile is materially below 1.0, report it and do not present the model as in-distribution.
