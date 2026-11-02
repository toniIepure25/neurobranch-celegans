# JOINTDYN innovation structure

**Result: G3 PASS for the corrected lead-1 population pool. D3 was not justified.**

## 1. Construction rule

Only `lead == 1` residuals enter the innovation analysis. A residual is accepted only when the model was fitted at the same frame used to make the one-step prediction:

```
fit at t -> predict X[t] to X[t+1] -> residual X[t+1] - prediction
```

The block builder records later frames in a block as leads 2, 3, ... because their parameters are stale. Those multi-step errors are excluded. The historical model incorrectly admitted leads 1-50 from a fixed fit and block-resampled them.

## 2. Sample sizes and basic structure

| Model | Median n | ACF1 | ACF5 | ACF10 | ACF20 | Temporal cross-cov lag 1 | Effective rank | Top eigenvalue fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Independent AR, lead 1 | 294 | 0.136 | 0.037 | 0.009 | -0.006 | 0.122 | 50.9 | 0.050 |
| Population VAR, lead 1 | 264 | **0.022** | **0.008** | -0.004 | -0.012 | **0.007** | 50.3 | 0.052 |

The corrected population residuals are close to white at the median. The independent AR residuals retain a small lag-1 dependence (0.136), but below the prespecified 0.15 threshold for a new temporal-innovation model.

## 3. Per-session population ACF1

| Session | ACF1 | ACF5 |
|---:|---:|---:|
| 0 | 0.053 | 0.009 |
| 1 | 0.022 | -0.016 |
| 2 | 0.021 | 0.009 |
| 3 | 0.031 | 0.008 |
| 4 | 0.020 | -0.013 |

No session shows a strong residual autoregression. There is no basis for adding a residual AR state.

## 4. Cross-neuron structure

- Effective rank is about 50 of 103.
- The largest eigenvalue carries only about 5% of residual trace.
- Population covariance split-half correlation is **0.107**; independent AR is **0.153**.
- Temporal cross-covariance at lag 1 is 0.007 for population residuals, 0.122 for independent residuals.
- Variance in the second half is 0.914 of the first half, indicating mild residual nonstationarity.

The covariance is high-dimensional and weakly stable across halves. That explains why a full 103-dimensional Ledoit-Wolf covariance overfits and why the full-covariance model (J3) fails.

## 5. Conditional residual mean

A ridge regression from the current 103-neuron state to the residual vector, fitted on the first 70% of the residual sample and tested on the last 30%, has held-out R² of **-4.88** (population) and **-3.21** (independent). The negative values mean the conditional mean is not estimable at this sample size; it is worse than predicting zero.

No state-dependent residual mean was added.

## 6. Low-rank temporal factor model

A rank-8 factor model was fitted as a diagnostic:

| Quantity | Population residuals |
|---|---:|
| Factor AR(1) median | 0.013 |
| Maximum absolute factor AR coefficient | — |
| Captured variance, rank 8 | 0.208 |

The factors are nearly non-autoregressive and explain only 21% of residual variance. A compact temporal model would therefore add little temporal structure while discarding most covariance. It was **not fitted as D3**.

## 7. Historical invalid pool

The previous audit established that multi-step residuals inflate covariance with lead. In that audit the adaptive block-50 pool had ACF1 0.553 and ACF5 0.505, while the lead-1 pool had ACF1 about 0.07 after whitening. Those historical numbers are carried forward as background, not re-used as new evidence.

The reproduction in this branch confirms the consequence: the historical block model has H20 energy 4.609, 95% coverage 0.999 and trajectory energy 3.211; the corrected lead-1 pool has energy 2.700, 95% coverage 0.947 and trajectory energy 2.060.

## 8. Process vs observation uncertainty

The data contain one calcium trace per neuron. Process noise and observation noise are not separately identified. No attempt is made to estimate a measurement-error variance. Any statement about "process innovation" is therefore conditional on the observed standardized calcium signal.

## 9. Decision

- G3 passes: population median |ACF1| = 0.0217 <= 0.15 and |ACF5| = 0.0078 <= 0.10.
- The residual process is not strongly temporally dependent.
- D3 is marked `NOT_FITTED` because Stage C did not justify it.
- The unexplained structure is cross-sectional covariance instability and mild nonstationarity, not serial innovation dependence.

## 10. Artifacts

- `innovation_structure.csv`
- `innovation_acf.csv`
- `innovation_factors.csv`
- `innovation_reconstruction.csv`
- `innovation_decision.json`
- `figures/jd_autocorrelation.png`
