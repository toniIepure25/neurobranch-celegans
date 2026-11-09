# Tier 1 benchmark — pre-run design record

This record was written before the final benchmark run and is kept with the result.

## Hypothesis

A behaviour-conditioned generator should forecast held-out-worm behaviour better than a global
quadratic map, while a kernel sampler over observed transitions should remain a strong descriptive
baseline. We expected the quadratic map to win one-step latent prediction and lose at longer
horizons.

## Experimental unit and target

- Unit: one worm (`worm_id` 0–4), not one frame. Splits are leave-one-worm-out.
- Target: future 3-D BunDLe-Net latent and future behaviour label.
- Test access: each held-out worm is used once per model; no hyperparameter tuning on the test worm.

## Data and version

- Five published consistent embeddings (`data/processed/bundlenet_sim/`), 3-D, 3121–3295 frames.
- Behaviour labels 0–7; canonical order in `dataset.BEHAVIOR_NAMES`.
- Models fit on transitions from the other four worms only. All scalers, read-outs and behaviour
  transition counts are fit inside that training fold.

## Primary metric

Behaviour accuracy over a **25-frame open-loop horizon**, averaged over starts and seeds. Starts
are every 75 frames, beginning at frame 0; **16 seeds** are used for stochastic models. A start is
used only if the full horizon exists.

## Secondary metrics (v2 amendment, before final run)

- Latent MSE at horizon 10 and its skill versus a constant-initial-state baseline.
- **Ensemble CRPS** at horizon 10 (proper score; lower is better).
- **Marginal 50% and 90% interval coverage** of the ensemble predictive distribution.
- Behaviour accuracy at horizons 1, 10 and 25.
- **Behaviour NLL and Brier score** from the ensemble categorical read-out.
- Jensen–Shannon divergence of generated versus true behaviour distribution and behaviour
  transition matrix at horizon 25.
- Quadratic one-step latent R², as a sanity check on the fitted map.
- Horizon curve of latent MSE at 1, 5, 10 and 25 frames.
- Persistence latent MSE, persistence behaviour accuracy, and marginal-behaviour NLL.

The v2 amendment was made after the v1 run exposed zero-width intervals for the deterministic
quadratic model. Quadratic and switching rollouts now receive **training-residual noise** (standard
deviation of the one-step residual on the four training worms), so coverage is defined. This is a
model-based observation-noise calibration, not a posterior.

## Baselines and comparators

- Constant-initial-state latent baseline (for MSE skill).
- Majority behaviour label (descriptive context; not a generative model).
- Quadratic ridge map (the existing Tier 1 model).
- Switching linear map with an empirical Markov behaviour chain.
- Manifold walk: k-nearest-neighbour sampler over observed transitions, temperature 0.6.

## Decision criterion

Report all three models with per-worm values and 95% normal intervals. Do **not** select a winner
from a difference smaller than the interval overlap. Any claim is limited to descriptive forecast
performance on these five worms.

## Leakage and confound checks

- No held-out worm contributes to model fitting, scaler statistics, behaviour transition counts or
  neighbour library.
- Starts come from the held-out worm only; the generated trajectory is never teacher-forced.
- Behaviour labels are observed at training time (switching model) and are not used to fit the
  held-out worm's transition matrix.
- Cluster IDs are never compared across worms in Tier 2.

## Seeds and uncertainty

16 seeds for switching and manifold walk; quadratic uses training-residual noise with 16 seeds.
Uncertainty is the between-worm standard deviation and a normal 95% interval over five folds.
Ensemble intervals are **not** calibrated posterior intervals; observed 90% coverage is reported
and is currently below nominal. This is a small-n interval and is reported as such, not as a
population confidence guarantee.

## Known limitations

- Only five worms.
- The manifold walk cannot generate transitions absent from the training library.
- The switching model assumes behaviour is observed and Markov; both are approximations.
- The benchmark does not test mechanism, intervention, or out-of-distribution animals.
