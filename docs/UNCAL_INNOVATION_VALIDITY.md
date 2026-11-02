# Prequential innovation validity

This document answers Stage B: were the innovations used by the ensemble actually valid one-step innovations? The evidence is in `innovation_validation.csv`, `innovation_summary.csv`, `lead_profile.csv`, and `lowrank_validation.csv`.

## 1. What counts as a valid innovation

For a one-step forecast, the innovation should be approximately:

- mean zero conditional on past activity;
- serially uncorrelated, or at least not strongly dependent;
- not predictable from the current state;
- stable in variance across the evaluation period;
- not dominated by a cross-neuron factor that a diagonal model ignores.

No test can prove all of these. The purpose here is to reject clearly invalid pools and to label the remaining assumptions.

## 2. Definitions

- **Lead 1**: residual `x_{t+1} - forecast_t(x_{t+1})` from a model fitted only on data before `t`.
- **Lead k**: residual `x_{t+k} - forecast_t(x_{t+k})` from the same fixed fit; this is a multi-step forecast error.
- **Whitened ACF**: autocorrelation after dividing by the fitted innovation scale; it removes the effect of changing variance.
- **State predictability**: out-of-sample R² from ridge regression of the residual on causal state features (absolute level, absolute change, recent standard deviation, recent absolute mean).
- **Cross-neuron split-half correlation**: correlation between the off-diagonal covariance estimated on the first and second halves of the residual series.

All residuals are prequential: the forecast at `t` uses no observation after `t`. The evaluation target is `t+lead`, but the fit is causal.

## 3. Serial dependence

Median values over sessions and landmarks:

| Residual pool | ACF(1) | ACF(5) | ACF(10) | whitened ACF(1) |
|---|---:|---:|---:|---:|
| independent AR, lead 1 | 0.071 | 0.195 | 0.147 | 0.015 |
| independent AR, block 50 | 0.104 | 0.256 | 0.217 | 0.024 |
| adaptive VAR, block 10, lead 1 | 0.288 | 0.178 | 0.054 | 0.070 |
| adaptive VAR, block 10 | 0.288 | 0.178 | 0.054 | 0.070 |
| adaptive VAR, block 50, all leads | **0.553** | **0.505** | **0.394** | 0.214 |

The independent AR lead-1 pool is close to white after whitening (ACF1 0.015). The raw ACF5 of 0.195 is non-zero, so the pool is not literally i.i.d.; this is a caveat, not a free pass.

The adaptive block-50 pool is strongly serially dependent. It is **not a valid innovation pool** for an independent-increment recursion. A block bootstrap of that pool preserves the dependence inside each sampled block, which is exactly one route to excess variance.

## 4. Mean predictability from state

Out-of-sample R² for predicting the residual mean from causal state features:

| Model | Mean R² |
|---|---:|
| adaptive VAR | -0.375 |
| independent AR | -0.439 |

Negative R² means the ridge map performed worse than predicting the residual mean. This is not evidence that the conditional mean is correct; it is evidence that this particular state map does not generalize. Possible reasons include distribution shift between fit and evaluation frames, finite-sample overfitting, or a genuinely state-independent mean. The experiment cannot distinguish these with the available data.

The negative result is retained as a negative result. It would be wrong to write "residuals are unpredictable" without the qualification.

## 5. Conditional variance

A state-conditioned variance model was fitted with four causal features and evaluated on held-out frames:

| Quantity | Value |
|---|---:|
| state-variance log-variance R² | approximately -0.006 (adaptive), -0.001 (independent) |
| state-variance variance R² | approximately -0.21 (both) |
| variance second-half / first-half ratio | 0.88 (adaptive), 0.89 (independent) |
| cross-neuron covariance split-half correlation, adaptive | 0.20 |
| cross-neuron covariance split-half correlation, independent | 0.60 |

The state model does not recover stable conditional variance. The variance is not constant either: the second-half ratio is 0.88, so there is drift. A constant diagonal innovation is therefore an approximation, not a verified description.

The independent model's cross-neuron covariance is more reproducible across halves than the adaptive model's (0.60 vs 0.20). This is consistent with the independent AR being the more stable statistical description, but it does not establish that cross-neuron dependence is absent.

## 6. Why the low-rank model was not built

Stage D allowed a low-rank-plus-diagonal innovation only if the residual audit justified it. A held-out Gaussian likelihood selected:

| Session | Selected rank |
|---|---:|
| 0 | 0 (diagonal) |
| 1 | 3 |
| 2 | 0 |
| 3 | 0 |
| 4 | 0 |

Diagonal won in four of five sessions; mean rank 0.6. The retained factor fraction was small even when rank 3 was selected. There was no stable, meaningful low-rank structure. `build_lowrank_model` is therefore **false**, and no low-rank model enters the final comparison.

This is a prespecified decision, not a post-hoc simplification.

## 7. Validation labels

| Pool | Label | Reason |
|---|---|---|
| independent AR, lead 1 | **conditionally acceptable** | whitened ACF1 0.015; raw ACF5 0.195 caveat; no positive state predictability |
| independent AR, block 50 | **weakly invalid** | stronger raw ACF; block resampling adds dependence |
| adaptive VAR, block 10, lead 1 | **weakly invalid** | ACF1 0.288, whitened 0.070; cross-neuron covariance unstable |
| adaptive VAR, block 50, all leads | **invalid** | ACF1 0.553; multi-step errors; serial dependence |
| state-conditioned diagonal | **not justified** | no stable conditional variance gain; worse calibration |
| low-rank plus diagonal | **not justified** | held-out likelihood selects diagonal in 4/5 sessions |

The primary model uses the least invalid pool available. That does not make it correct; it makes it the defensible prespecified choice and gives the calibration failure a clean interpretation.

## 8. What would change the verdict

- A residual ACF that remains large after whitening, or a state map with positive held-out R², would require a richer mean or variance model.
- A stable low-rank covariance across sessions would justify D3.
- An observation model with repeated measurements or simultaneous electrophysiology could separate process from measurement noise; this dataset cannot.

## Related documents

- [Variance propagation](UNCAL_VARIANCE_PROPAGATION.md)
- [Probabilistic models](UNCAL_PROBABILISTIC_MODELS.md)
- [Calibration](UNCAL_CALIBRATION.md)
- [Limitations](UNCAL_LIMITATIONS.md)
