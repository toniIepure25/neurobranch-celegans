# Supervisor brief — generative-failure diagnostics

## Bottom line

The existing NeuroBranch simulators do not fail because of a simple preprocessing leak or a misaligned target. They fail because the observed population is only weakly forecastable at 7–34 s, the latent representation is not stable across the recording, and free rollout amplifies model error. A calcium-aware observation model improves one-step prediction but is numerically unstable in autonomous rollout. Oxygen transitions show no reproducible response and are confounded with elapsed time. Virtual perturbations disagree across observationally similar models.

## Strongest evidence

- **Reproduction is exact.** Session-0 H20 latent LDS skill −1.9716 and stable VAR −1.6279 reproduce stored values to six decimals; eight headline metrics reproduce within 5×10⁻⁴.
- **Persistence is a hard baseline.** No model has positive mean skill at any horizon. Persistence R² falls from 0.395 at 1 frame to 0.071 at 20 and −0.754 at 100.
- **Population coupling helps only modestly.** Population VAR beats independent AR in 4/5 sessions at H20 by ~0.1 MSE, but remains worse than persistence.
- **Latent state is locally observable but time-unstable.** Observability rank 15/15; half-split subspace angle 56–62° versus 3–6° bootstrap.
- **Observation model does not rescue rollout.** Calcium LDS one-step mean skill −0.529, but H20 mean −7.05×10³ and H100 ~−1.8×10³⁷.
- **Oxygen is null and confounded.** Event-pair correlation ≈0; event-window skill −0.194 versus time-only −0.097; fixed 30 s schedule, no randomized/sham timing.
- **Perturbation is model-dependent.** Cross-model cosine ranges from −0.073 to 0.976; target-ranking median Spearman 0.371.

## What is publishable / discussable

A rigorous negative/diagnostic contribution:

1. a reproducible failure audit;
2. a forecastability map against persistence, mean, trend and independent-neuron baselines;
3. evidence that calcium observation is a limitation but not the sole cause;
4. a time-instability result for latent subspaces;
5. a controlled null oxygen analysis;
6. a cross-model perturbation-ambiguity warning.

This is not a new state-of-the-art simulator. It is a constraint on what the current data and models can support.

## What not to claim

- No biological causal effect of any named neuron.
- No oxygen-evoked response.
- No animal-independent generalisation.
- No anatomical meaning of latent axes.
- No “validated digital twin”.
- No claim that the system is unpredictable in principle.

## Recommended next step

The smallest decisive experiment is **randomized oxygen timing with a sham schedule in independent animals**, analysed with the same causal baseline, time-only and placebo controls. This directly tests whether the oxygen null is biological or design-limited.

The smallest cheap analysis is an **animal-disjoint re-split** of existing recordings, to test whether the forecast failure survives without reusing the same test tails. It cannot create new animals, but it tightens the uncertainty statement.

## Decision status

`FORECASTABILITY_CHARACTERIZED`, `OBSERVATION_MODEL_TESTED`, `IDENTIFIABILITY_ASSESSED`, `OXYGEN_RESPONSE_ASSESSED`, `PERTURBATION_UNCERTAINTY_CHARACTERIZED` are **PASS** as diagnostics. `BIOLOGICAL_INTERVENTION_VALIDATION` is **NOT_EVALUABLE**.
