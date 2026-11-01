# Forecastability — what is predictable, and for how long?

Artifacts: `reports/simulator/failure_diagnostics/forecastability/`
Protocol: `docs/GENFAIL_RESEARCH_PROTOCOL.md`
Data: Kato 2015 WT_NoStim, five sessions, chronological 70/15/15 split, 12-frame purge.

## 1. Question and metric

For every model and horizon, predictions start from the same observed history at the same test frame. The primary metric is

`skill = 1 − MSE_model / MSE_persistence`,

where persistence predicts the last observed value for every future frame. Positive means better than persistence; zero means no improvement; negative means worse. `R²` is computed against the test target variance and is reported separately because a model can improve on persistence while still having negative absolute R².

Horizons were 1, 3, 5, 10, 20, 50 and 100 frames. At ~2.8–3.1 fps these are approximately 0.3, 1, 1.7, 3.4, 7, 17 and 34 s. Results are per session; aggregate intervals are moving-block descriptive intervals, not animal-level confidence intervals.

## 2. Models compared

| model | information used | role |
|---|---|---|
| persistence | last observed frame | reference |
| train mean | train-only mean | marginal reference |
| local causal trend | last two observations, causal extrapolation | naive trend control |
| per-neuron AR(5) | each neuron’s own past only | independent-neuron reference |
| population VAR(1) | all neurons’ past, ridge | population linear reference |
| latent LDS | train-only PCA + latent AR(1) | existing low-dimensional model |

All hyperparameters were fixed before the test evaluation. No model was selected on test.

## 3. Headline result

**No model beats persistence on average at any horizon.** The best mean skill is always negative; the least-negative model is population VAR at long horizons.

| horizon (frames) | persistence R² | best model | best mean skill | population VAR mean skill | latent LDS mean skill |
|---:|---:|---|---:|---:|---:|
| 1 | 0.395 | persistence | 0.000 | −1.954 | −2.487 |
| 3 | 0.371 | persistence | 0.000 | −2.009 | −2.357 |
| 5 | 0.340 | persistence | 0.000 | −1.957 | −2.221 |
| 10 | 0.254 | persistence | 0.000 | −1.782 | −1.968 |
| 20 | 0.071 | persistence | 0.000 | −1.492 | −1.646 |
| 50 | −0.365 | persistence | 0.000 | −0.993 | −1.121 |
| 100 | −0.754 | persistence | 0.000 | −0.624 | −0.733 |

The pattern is not “the models get better with horizon”. Absolute error still grows: population VAR MSE rises from 1.19 at step 1 to 2.02 at step 100, while persistence rises from 0.43 to 1.73. Skill improves only because persistence degrades faster. At every horizon the model’s absolute MSE remains above persistence.

The local trend control is catastrophic at long horizons (mean skill −31 at H50, −94 at H100): a causal linear extrapolation of the last two frames is not a stable forecasting rule.

## 4. Which components are predictable?

At H20, per-neuron skill versus persistence:

| model | fraction of neuron-session series with positive skill | median skill |
|---|---:|---:|
| per-neuron AR(5) | 0.199 | −0.725 |
| population VAR | 0.299 | −0.417 |
| latent LDS | 0.280 | −0.549 |

So ~20–30% of individual series are forecastable better than persistence at 7 s, but the typical neuron is not. The forecastable subset is concentrated in slow, high-autocorrelation neurons. Examples at H20 (worm 4) include unnamed trace 18 (skill 0.680, train autocorrelation 0.962), AVEL (0.648, 0.992), trace 126 (0.621), AVER (0.582), RIML (0.464). These are descriptive examples, not a neuron-class claim.

The fraction with positive skill rises at longer horizons because persistence becomes a weaker baseline:

| horizon | AR(5) positive fraction | population VAR positive fraction | latent LDS positive fraction |
|---:|---:|---:|---:|
| 1 | 0.095 | 0.174 | 0.206 |
| 20 | 0.199 | 0.299 | 0.280 |
| 100 | 0.382 | 0.498 | 0.443 |

Even at H100 half of population-VAR series are still worse than persistence. This is not evidence that “most neurons become predictable at 34 s”; it is evidence that persistence itself becomes a poor predictor there.

## 5. Does the population add information beyond independent neurons?

Yes, modestly, but not enough to rescue the forecast. Population VAR minus independent AR(5) MSE is negative on average at every horizon:

| horizon | mean MSE(population) − MSE(independent) | sessions where population is better |
|---:|---:|---:|
| 1 | −0.146 | 4/5 |
| 5 | −0.115 | 5/5 |
| 20 | −0.105 | 4/5 |
| 50 | −0.063 | 3/5 |
| 100 | −0.039 | 3/5 |

The advantage shrinks with horizon and disappears in two of five sessions by H100. Cross-neuron coupling contains some short-horizon information, but the population model is still far below persistence. The correct conclusion is **“population structure helps relative to independent neurons, but the dominant failure is not lack of coupling in a linear model”**, not “the population model is good”.

## 6. Where does predictability disappear?

Using absolute persistence R² as the descriptive criterion, the five-session mean crosses zero between H20 (mean R² 0.071) and H50 (mean R² −0.365). At H20 the mean is already only 0.07, and the per-session values range from negative to modest positive. A defensible statement is:

- **~7 s (20 frames):** little average absolute predictability remains; most neurons are at or below persistence.
- **~17 s (50 frames):** persistence has negative mean R²; no tested model has positive mean skill.
- **~34 s (100 frames):** all models have negative mean skill and large absolute error.

The exact crossover is session-dependent and should not be reported as a biological time constant.

## 7. Stability across test time

Splitting the test tail into three equal chronological thirds shows degradation in later thirds for every non-persistence model:

| model | test third 1 mean skill | third 2 | third 3 |
|---|---:|---:|---:|
| population VAR | −1.223 | −1.519 | −1.823 |
| latent LDS | −1.336 | −1.686 | −2.009 |
| AR(5) | −1.392 | −1.687 | −1.998 |

This is consistent with a distribution shift over the recording, not with a stationary dynamical model that simply needs more capacity. It also means a single test-tail average hides time-varying failure.

## 8. Observation-noise sensitivity

Adding independent Gaussian noise to the history (in train-standardized units) makes every model’s skill less negative, including persistence-relative models:

| noise SD | AR(5) mean skill H20 | population VAR mean skill H20 |
|---:|---:|---:|
| 0.00 | −1.661 | −1.492 |
| 0.10 | −1.627 | −1.451 |
| 0.25 | −1.459 | −1.255 |
| 0.50 | −1.084 | −0.805 |

This is not evidence of robustness. Noise drives model predictions toward the mean, which can reduce MSE when the target is itself mean-reverting, and it degrades the persistence reference too. The experiment shows that the ranking is not improved by adding noise; it does not identify the real calcium noise level.

## 9. Interpretation against the hypotheses

- **H1 (limited forecastability):** strongly supported. Persistence R² is near zero by 20 frames and negative by 50; no model has positive mean skill.
- **H2 (observation-model mismatch):** plausible but not sufficient by itself; tested in `GENFAIL_OBSERVATION_MODEL.md`.
- **H3 (missing state):** not ruled out; linear population coupling gives only a small gain.
- **H4 (inadequate identification):** supported; test-third degradation and cross-session variability show that the fitted operator is not stable over the recording.
- **H5 (data/evaluation mismatch):** supported as a contributor; later test thirds are worse for every model.
- **H6 (structural non-identifiability):** tested separately; subspace instability is severe.

## 10. What this does not show

- It does not show that the system is unpredictable in principle; it shows that these linear/low-dimensional models, with this observation, lose to persistence.
- It does not establish a biological forecasting horizon.
- It does not provide animal-level uncertainty; the five sessions are repeated recordings, and the test tails are not independent animals.
- It does not prove that nonlinear models cannot help; it sets the baseline such a model must beat.

Stage verdict: `FORECASTABILITY_CHARACTERIZED` — PASS for the descriptive characterisation, LIMITED for animal-level generalisation.
