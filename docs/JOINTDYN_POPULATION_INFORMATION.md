# JOINTDYN population information

**Question:** does the population model add useful information beyond a strong independent-neuron forecast?

## 1. Headline

The population mean adds a reproducible multivariate and trajectory advantage in the fixed 10-D PCA. The full innovation covariance does **not** add value and makes every joint diagnostic worse. The improvement is therefore about conditional-mean population dynamics, not about recovering the cross-neuron innovation covariance.

Marginal per-neuron forecasting is essentially unchanged. This is CLAIM A not supported, CLAIM B partially supported, CLAIM C not supported.

## 2. Scores, 175 landmarks

| Model | H20 energy | H20 trajectory energy | Full-neuron CRPS | 50% coverage | 95% coverage | Neural covariance error |
|---|---:|---:|---:|---:|---:|---:|
| J0 persistence | 2.950 | 2.286 | 0.4524 | 0.494 | 0.936 | 0.630 |
| J1 independent diag | 3.146 | 2.435 | 0.4182 | 0.471 | 0.890 | 0.505 |
| J2 population diag | **2.700** | **2.058** | 0.4137 | 0.586 | 0.944 | 0.572 |
| J3 population full | 3.011 | 2.277 | 0.4594 | 0.767 | 0.992 | 0.762 |
| J4 independent full | 3.007 | 2.359 | 0.4242 | 0.560 | 0.934 | 0.517 |
| J5 population lead-1 | **2.700** | **2.060** | 0.4134 | 0.588 | 0.947 | 0.604 |

Lower energy and CRPS are better. Coverage is full-neuron at H20.

## 3. Paired contrasts

Positive improvement means the model has a lower score than the reference. Intervals are moving-block bootstrap over landmarks, block 5.

| Model vs reference | Metric | Improvement | 95% interval | Fraction better |
|---|---|---:|---:|---:|
| J5 vs J1 | H20 energy | 0.446 | [0.183, 0.660] | 0.611 |
| J5 vs J1 | H20 trajectory energy | 0.375 | [0.218, 0.522] | — |
| J5 vs J0 | H20 energy | 0.250 | [0.020, 0.427] | — |
| J5 vs J0 | H20 trajectory energy | 0.226 | [0.089, 0.337] | — |
| J2 vs J1 | H20 energy | 0.445 | [0.184, 0.662] | — |
| J2 vs J1 | H20 trajectory energy | 0.377 | [0.224, 0.527] | — |
| J3 vs J1 | H20 energy | 0.135 | [-0.133, 0.355] | — |
| J3 vs J1 | H20 trajectory energy | 0.159 | [-0.005, 0.309] | — |

The population advantage over independent AR survives block lengths 3, 5 and 10. The advantage over persistence is less robust: the block-10 interval for H20 energy is `[-0.032, 0.442]`.

## 4. Marginal forecasting: claim A

Full-neuron CRPS:

- J5 0.41337 vs J1 0.41821;
- improvement 0.00484, 95% interval `[-0.00830, 0.01566]`.

The interval includes zero. The population model **does not** provide a reliable per-neuron marginal forecasting improvement. This is consistent with the previous study: the independent AR model is already strong on marginal CRPS.

## 5. Representation sensitivity

| Model | 10-D PCA energy | All-neuron energy | Exclude top-5 variance neurons | Top-20 variance neurons |
|---|---:|---:|---:|---:|
| J5 | 2.700 | 6.847 | 6.772 | 2.498 |
| J1 | 3.146 | 7.050 | 6.973 | 2.588 |
| J0 | 2.950 | 7.380 | 7.281 | 2.637 |
| J3 | 3.011 | 7.351 | 7.263 | 2.705 |

The population advantage persists in every representation, but the all-neuron energy interval includes zero:

- J5 vs J1 all-neuron energy: 0.203 `[-0.015, 0.412]`;
- excluding the top-5 training-variance neurons: 0.202 `[-0.017, 0.411]`;
- top-20 training-variance neurons: positive.

The 10-D PCA advantage is the strongest and most stable result. It should not be generalised to all 103 neurons without qualification.

## 6. Factorial decomposition

The 2 x 2 design crosses mean (independent vs population) with innovation covariance (diagonal vs full) at H20.

| Metric | Independent diag | Independent full | Population diag | Population full | Mean main effect | Covariance main effect | Interaction |
|---|---:|---:|---:|---:|---:|---:|---:|
| H20 energy | 3.146 | 3.007 | 2.700 | 3.011 | **+0.221** | **-0.086** | +0.449 |
| H20 trajectory | 2.435 | 2.359 | 2.058 | 2.277 | **+0.230** | **-0.071** | +0.296 |

Positive main effect means the factor improves the score.

- The population mean effect is positive and material.
- The full-covariance effect is negative: full covariance hurts.
- The interaction is large because full covariance is especially harmful when combined with the population mean.

The covariance result is not a subtle null. J3 over-covers at every level (0.767/0.959/0.992) and has the worst neural covariance error (0.762).

## 7. Why full covariance fails

- The lead-1 residual covariance is high-dimensional: effective rank about 50 of 103; the top eigenvalue is only about 5% of trace.
- Split-half covariance correlation is only 0.107 for population residuals and 0.153 for independent residuals.
- Ledoit-Wolf shrinkage keeps the matrix positive definite but not scientifically stable at this sample size.
- Sampling a 103-dimensional full covariance adds variance along weakly identified directions; the 10-D projection then sees overdispersion and distorted covariance.

The full covariance is therefore rejected. A diagonal innovation model is the better-identified choice for this dataset.

## 8. Session-level consistency

J5 vs J1 H20 energy by session:

| Session | Improvement | 95% interval | Better fraction |
|---:|---:|---:|---:|
| 0 | +0.638 | [0.276, 1.007] | 0.77 |
| 1 | -0.152 | [-0.416, 0.042] | 0.31 |
| 2 | +0.243 | [-0.237, 0.683] | 0.50 |
| 3 | +0.752 | [0.159, 1.362] | 0.74 |
| 4 | +0.730 | [0.070, 0.957] | 0.73 |

The direction is positive in 4/5 sessions, which passes the frozen G6 rule, but session 1 is negative and session 2 is not distinguishable from zero. Five repeated sessions are not five independent animals.

## 9. Conclusion

- **CLAIM A:** not supported. Marginal CRPS improvement is small and uncertain.
- **CLAIM B:** partially supported in the fixed 10-D joint representation and variogram score; not supported in all-neuron energy.
- **CLAIM C:** not supported. Calibration and covariance gates fail.
- The useful signal is the population conditional mean, not the innovation covariance.

## 10. Artifacts

- `score_summary.csv`
- `paired_scores.csv`
- `secondary_paired.csv`
- `session_paired_scores.csv`
- `session_paired_ci.csv`
- `bootstrap_block_sensitivity.csv`
- `factorial_decomposition.csv`
- `sensitivity_representation.csv`
- `figures/jd_scores.png`
- `figures/jd_factorial.png`
