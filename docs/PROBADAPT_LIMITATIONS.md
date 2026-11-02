# Limitations

## Data and design

1. **Repeated sessions are not confirmed independent animals.** The five Kato
   WT_NoStim recordings are repeated sessions. Session-level variation is
   reported, but no animal-level confidence interval or generalization claim is
   possible.
2. **The test tail is reused.** The chronological split was fixed in earlier
   stages. This extension did not reselect the split or the primary rolling
   window, but the test landmarks have been seen in prior analyses. The result is
   exploratory, not a preregistered confirmatory test on fresh data.
3. **Landmarks overlap.** Consecutive H20 landmarks are ten frames apart, so
   their errors and ensemble scores are dependent. Moving-block bootstrap
   intervals account for some temporal dependence but do not create independent
   replicates.
4. **No interventional data.** The recordings are observational. Nothing in this
   report identifies a causal effect of a neuron or a circuit.
5. **One dataset and one preparation.** The conclusions are about Kato
   WT_NoStim, not C. elegans population dynamics in general.

## Modeling

6. **Lag-one linear VAR is a strong restriction.** A rolling VAR can absorb slow
   shared state through its intercept and local mean. That is exactly why the
   full model does not beat the local mean: the comparison cannot distinguish
   "cross-neuron coupling is absent" from "this linear lag-one representation
   does not capture it."
7. **Near-unit-root operators amplify innovation uncertainty.** The spectral cap
   `0.995` keeps rollouts finite but does not make the fitted operator
   statistically stable. Small changes in `A` or the innovation covariance can
   produce large changes in stationary variance.
8. **The innovation covariance is not stationary.** Residual scale depends on
   state and time. A single frozen covariance, even with shrinkage, is a
   misspecification.
9. **Block bootstrap preserves short-range dependence but not the full
   autocovariance.** A ten-frame block is a compromise. It does not reproduce
   long-memory or nonstationary variance, and it can add low-frequency power that
   inflates simulated variance.
10. **No arbitrary variance calibration was applied.** This is a scientific
    choice, but it means the reported failure is a property of the fitted model,
    not of a post-hoc rescaling. A future model may calibrate variance only with
    a prespecified, train-only procedure and proper scoring.

## Evaluation

11. **The primary energy score uses a 10-D PCA projection.** This makes the
    score tractable and fixed before test, but it is not the full 103–127
    dimensional neural state. A model can look better or worse in projection.
    Per-neuron CRPS and full-space variance are reported as checks.
12. **The projection is training-fitted but session-specific.** PCA components
    are not directly comparable across sessions; scores are aggregated within
    session and then summarized.
13. **Energy scores and coverage answer different questions.** A proper score
    rewards sharpness and calibration jointly; coverage exposes overdispersion.
    The post-hoc P7 variant illustrates the tension. The prespecified decision
    uses the primary model, not the best-looking variant.
14. **The 10-D temporal covariance error is near 1 for all models.** The truth
    covariance is nearly singular in this projection, so the relative error is
    not a precise discriminator. Variance ratio, coverage, width, OOD and energy
    are more informative.
15. **PIT is computed from per-neuron ensembles at H20.** It is a marginal
    diagnostic and does not test joint calibration.
16. **OOD is defined by training 1–99% ranges.** It is a useful support check,
    not a likelihood or a mechanistic statement.

## Baselines

17. **All baselines are linear and relatively simple.** They are strong in the
    sense that they adapt and are calibrated fairly, but they do not include
    nonlinear, switching or latent-state models. A nonlinear model could change
    the comparison.
18. **Baseline settings were selected on the last 20% of train.** This avoids
    test leakage but consumes training data and may not be optimal for every
    session. The selected settings are reported.
19. **Probabilistic baselines use block resampling with different block
    lengths.** This is defensible for each mean model but is not a fully
    hierarchical uncertainty model.

## Interpretation

20. **A negative result is not proof of absence.** Failure to demonstrate added
    cross-neuron information does not show that the population has no
    interactions. It bounds what this model, split and metric can show.
21. **A calibrated simulator would still not be a biological mechanism.** Even a
    model that passes distributional gates would be a model. Biological
    validation would require independent interventional data and out-of-sample
    animals.
22. **The trajectory example is one history.** It is illustrative, not a
    statistical result. Divergent paths are not biological bifurcations.
23. **The perturbation stage was not launched.** This is a deliberate
    consequence of failed gates, not an omitted analysis that would have been
    interpretable.
24. **Historical artifacts were not modified.** The earlier adaptive report
    remains as produced. This branch adds a new evidence layer and does not
    rewrite the prior percentage or its exploratory status.

## What would change the conclusion

- A genuinely held-out animal or independent recording with the same protocol.
- A stochastic model that matches H20 variance and coverage without widening
  intervals arbitrarily.
- A state-conditioned innovation model that predicts residual scale from
  pre-landmark variables.
- A nonlinear or latent-state model that beats local mean and independent AR on
  proper scores, not only MSE.
- Independent interventional data for any causal claim.
