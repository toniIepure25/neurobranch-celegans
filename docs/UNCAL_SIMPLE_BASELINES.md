# Simple probabilistic baselines

This document answers Stage C: what can the simplest probabilistic forecasters do, and does the added adaptation earn its complexity? The outputs are `scores.csv`, `score_summary.csv`, and `paired_scores.csv`.

## 1. Frozen comparators

| Model | Conditional mean | Innovation | Parameters |
|---|---|---|---|
| `P0_persistence` | last state | moving-block bootstrap of first differences | block 20, history 1000 |
| `P1_local_mean` | rolling mean, window 100 | moving-block bootstrap of local residuals | block 10, history 1000 |
| historical frozen AR | independent AR(1) | diagonal Gaussian | frozen from the earlier report |
| `P6_adaptive_block` | rolling population VAR, window 500 | block bootstrap, block-50 residual pool, leads 1-50 | historical |
| `P2_independent_ar_lead1` | independent per-neuron AR(1), window 200, ridge 1.0, cap 0.995 | diagonal Gaussian from lead-1 prequential residuals | primary |

All models use the same 175 landmarks, the same 10-D training PCA, the same 100 members, and the same seed policy. The historical comparator is rerun with its original block-50 residual definition and 1000-frame history, so its 4.609 energy is comparable to the frozen 4.6046.

## 2. H20 scores

| Model | Energy | CRPS | Coverage 50 | Coverage 80 | Coverage 95 | Width 95 | Variogram | Lag-1 autocorr error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 persistence | 2.950 | 0.452 | 0.494 | 0.791 | 0.936 | 3.276 | 0.248 | 0.192 |
| P1 local mean | 3.282 | 0.440 | 0.469 | 0.763 | 0.912 | 2.868 | 0.270 | 0.272 |
| P2 fixed train scale | 3.149 | 0.419 | 0.478 | 0.750 | 0.893 | 2.397 | 0.266 | 0.423 |
| **P2 independent AR lead 1** | **3.146** | **0.418** | **0.471** | **0.746** | **0.890** | **2.385** | **0.265** | **0.427** |
| P3 state AR | 3.112 | 0.419 | 0.463 | 0.739 | 0.884 | 2.412 | 0.263 | 0.423 |
| P6 adaptive block | 4.609 | 0.649 | 0.910 | 0.992 | 0.999 | 10.041 | 0.699 | 0.464 |
| P6 adaptive lead 1 | 2.700 | 0.413 | 0.588 | 0.849 | 0.947 | 3.090 | 0.218 | 0.409 |

Lower energy and CRPS are better. Coverage closer to nominal is better.

## 3. Paired comparisons for the primary

Paired over the 175 landmarks, block bootstrap with block 5 and 1000 replicates:

| Metric | Reference | Mean improvement | 95% interval | Fraction better |
|---|---|---:|---:|---:|
| CRPS H20 | P0 persistence | +0.034 | [0.020, 0.048] | 0.709 |
| CRPS H20 | P1 local mean | +0.022 | [0.008, 0.038] | 0.634 |
| CRPS H20 | P6 adaptive block | +0.046 | [0.034, 0.059] | 0.806 |
| Energy H20 | P0 persistence | -0.195 | [-0.413, 0.003] | 0.469 |
| Energy H20 | P1 local mean | +0.137 | [-0.119, 0.419] | 0.623 |
| Energy H20 | P6 adaptive block | -0.050 | [-0.307, 0.243] | 0.520 |
| Variogram H20 | P0 persistence | -0.017 | [-0.039, 0.008] | 0.469 |
| Temporal autocorr error | P0 persistence | -0.235 | [-0.271, -0.198] | 0.103 |

Positive improvement means the reference is worse. The CRPS comparisons are the prespecified primary and both lower bounds are above zero: **Gate D passes**. The energy comparison against persistence is not significant and the point estimate favors persistence; the trajectory and autocorrelation comparisons favor persistence.

This is a genuine trade-off, not a win. The independent AR has the best marginal CRPS but a worse joint trajectory and worse temporal dependence than persistence.

## 4. Why local mean is not the winner

P1 has a worse energy than P2 (3.282 vs 3.146) and a worse CRPS than P2 (0.440 vs 0.418). Its 50% coverage is 0.469, which is close to nominal; its 95% coverage is 0.912, still under. It is a strong simple baseline on marginal CRPS and a weak model on the joint path.

The comparison matters because it rules out the simplest explanation "any adaptive model is better." P2 only narrowly beats P1 on CRPS, and P6 lead-1 beats both on CRPS and energy while failing 50% coverage and joint covariance.

## 5. What the baselines establish

- **Persistence is a serious probabilistic baseline.** Its 95% interval is nearly calibrated (0.936) and its joint trajectory energy is better than the primary's.
- **The historical adaptive model is not competitive.** Its energy is 4.609 and its 95% interval covers 99.9% of outcomes. The low deterministic MSE did not translate into a usable probability model.
- **The independent AR improvement is marginal-only.** It wins the proper marginal score but does not win the multivariate energy, variogram, or trajectory score.
- **The corrected population model is a diagnostic, not a replacement.** It has the best energy and CRPS, but its 50% coverage is 0.588 and its covariance relative error is 0.492. It is not a validated generative model.

## 6. Reproducibility

```powershell
python scripts/uncal_evaluation.py
python scripts/uncal_summarize.py
```

The first command writes per-landmark scores, per-neuron H20 scores, coverage by horizon, variance summaries, covariance summaries, paired scores, and example updates. The second writes the summary and gate records.

## Related documents

- [Probabilistic models](UNCAL_PROBABILISTIC_MODELS.md)
- [Calibration](UNCAL_CALIBRATION.md)
- [Trajectory fidelity](UNCAL_TRAJECTORY_FIDELITY.md)
- [Results](UNCAL_RESULTS.md)
