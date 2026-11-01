# Observation model — does calcium filtering explain the failure?

Artifacts: `reports/simulator/failure_diagnostics/observation/`
Code: `src/neurobranch/diagnostics/observation.py`, `src/neurobranch/diagnostics/synthetic.py`
No biological intervention is claimed.

## 1. Question

The recorded traces are calcium imaging signals, not membrane voltage. Calcium kinetics, indicator noise, and indirect observation could make the true dynamical state unobservable or distort its transition operator. This stage asks:

1. Can the existing latent model recover a known synthetic dynamical system from calcium-like observations?
2. Does a small, causal, observation-aware model improve real held-out forecasts on identical starts?
3. Does an observation-aware model produce the correct **sign and scale** of synthetic perturbation responses?

## 2. Models tested

| model | observation handling | transition |
|---|---|---|
| AR(1) | none; standardized trace as state | first-order linear |
| AR(2) | none; two lags | second-order linear |
| second-order latent LDS | train-only PCA, two latent lags | latent VAR(2), causal decode |
| causal calcium LDS | causal high-pass + train-only PCA + first-order calcium filter | latent AR(1), alpha selected on train one-step MSE |

The calcium alpha was selected only on training frames per session. Alpha values tested were 0.1, 0.2, 0.3, 0.5, 0.7, 0.9. No future-dependent smoothing, no test-time refitting, no test-selected alpha.

Selected alpha values varied by session (0.3, 0.5, 0.7, 0.3, 0.7), which is itself a warning: a single fixed calcium time constant is not supported across recordings.

## 3. Synthetic known-truth controls

Synthetic systems: stable unfiltered, calcium-filtered stable, oscillatory unfiltered, oscillatory filtered. Each has four latent dimensions and calcium-like observation noise. H20, 180 starts. Skill is versus persistence on the observed traces.

| condition | model | skill H20 | R² H20 |
|---|---|---:|---:|
| stable unfiltered | AR(1) | 0.384 | 0.231 |
| stable unfiltered | AR(2) | 0.390 | 0.238 |
| stable unfiltered | calcium LDS | 0.188 | −0.015 |
| calcium filtered | AR(1) | 0.356 | 0.350 |
| calcium filtered | AR(2) | 0.350 | 0.345 |
| calcium filtered | calcium LDS | 0.287 | 0.281 |
| oscillatory unfiltered | AR(1) | 0.694 | 0.389 |
| oscillatory unfiltered | AR(2) | 0.702 | 0.404 |
| oscillatory unfiltered | calcium LDS | 0.514 | 0.030 |
| oscillatory filtered | AR(1) | 0.672 | 0.347 |
| oscillatory filtered | AR(2) | 0.677 | 0.358 |
| oscillatory filtered | calcium LDS | 0.346 | −0.301 |

Findings:

- The synthetic systems **are forecastable**. AR models beat persistence, so the evaluation itself is not impossible.
- Calcium filtering lowers attainable skill but does not make the system unpredictable.
- The observation-aware calcium model is consistently **worse** than the simpler AR models on these controls. It does not recover an advantage from modelling the filter.
- The synthetic latent recovery is good: R² between true and estimated latent 0.77–0.99, so the low-dimensional structure is identifiable in these clean controls.

## 4. Synthetic perturbations

A known latent perturbation of amplitude 0.5 was applied and the 20-frame response was predicted. The true response norm is small (0.014–0.069). All models over-predict its magnitude by factors of 2–7:

| condition | AR(1) cosine | AR(2) cosine | calcium cosine | relative response error (calcium) |
|---|---:|---:|---:|---:|
| stable unfiltered | 0.50 | 0.49 | 0.46 | 7.10 |
| calcium filtered | 0.55 | 0.59 | 0.52 | 6.97 |
| oscillatory unfiltered | −0.01 | 0.00 | −0.04 | 2.38 |
| oscillatory filtered | −0.10 | −0.07 | −0.11 | 3.46 |

A cosine near zero means the predicted response direction is unrelated to the true response direction. This is direct evidence that **observational similarity is not enough for perturbation validity**, even when the latent state is recoverable.

## 5. Real data

Five WT_NoStim sessions, identical test starts and horizons. Negative skill means worse than persistence.

| model | horizon | mean skill | median skill | range |
|---|---:|---:|---:|---|
| AR(1) | 1 | −2.612 | −2.851 | −5.424 to −0.314 |
| AR(2) | 1 | −2.596 | −2.842 | −5.402 to −0.299 |
| calcium LDS | 1 | **−0.529** | −0.166 | −1.307 to +0.061 |
| AR(1) | 20 | −1.679 | −1.014 | −4.826 to −0.128 |
| AR(2) | 20 | −1.561 | −0.915 | −4.551 to −0.063 |
| calcium LDS | 20 | **−7,054** | −466 | −22,390 to −6.28 |
| AR(1) | 100 | −0.751 | −0.212 | −2.832 to +0.242 |
| AR(2) | 100 | −0.659 | −0.108 | −2.672 to +0.257 |
| calcium LDS | 100 | **−1.8 × 10³⁷** | −1.3 × 10²⁸ | explode |

The observation-aware model is the **best one-step model** among these (mean skill −0.529; one session slightly positive) but it is numerically unstable in autonomous multi-step rollout. The instability is not a cosmetic overflow: the causal high-pass plus recursive first-order calcium feedback amplifies errors. We report the failure rather than clipping it and calling the model successful.

The AR(2) model is slightly better than AR(1) at every horizon, but both remain far below persistence. Adding a second lag does not fix the problem.

## 6. Interpretation

- **H2 is supported as a real limitation but not as the sole explanation.** Calcium filtering reduces achievable skill in synthetic controls, and the real observation-aware model changes one-step performance. But it does not produce a stable multi-step model that beats persistence.
- **Known dynamics are recoverable from calcium-like observations in synthetic data.** Therefore “calcium makes identification impossible” is too strong.
- **Perturbation responses are not recovered reliably**, even with known synthetic dynamics. This anticipates the perturbation-identifiability failure.
- The real-data one-step advantage of the calcium model is not a simulator success: one-step prediction is not autonomous rollout, and its H20/H100 behaviour is catastrophic.

## 7. Negative findings preserved

1. The calcium LDS explodes at H20 and H100; we do not hide or winsorize this.
2. Alpha is not stable across sessions.
3. AR(2) does not rescue the forecast.
4. Synthetic perturbation direction is wrong for oscillatory systems.
5. No observation-aware model was selected on test.

Stage verdict: `OBSERVATION_MODEL_TESTED` — PASS for the test being executed and reported; the causal observation-aware model does **not** improve real autonomous forecasting. This is a negative result, not a validated model.
