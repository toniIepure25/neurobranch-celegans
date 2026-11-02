# Calibration of marginal predictive distributions

This document answers Stage F/G for the marginal target. The joint trajectory target is in [Trajectory fidelity](UNCAL_TRAJECTORY_FIDELITY.md).

A marginal predictive distribution is calibrated if, across repeated forecasts, the observed value falls inside a nominal central interval at the nominal rate. That is necessary but not sufficient for a generative simulator.

## 1. Primary H20 result

| Model | CRPS | Coverage 50 | Coverage 80 | Coverage 95 | PIT error | Std-error variance | Width 95 | Width ratio vs persistence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **P2 primary** | **0.418** | **0.471** | **0.746** | **0.890** | **0.0145** | **1.730** | **2.385** | **0.731** |
| P0 persistence | 0.452 | 0.494 | 0.791 | 0.936 | — | — | 3.276 | 1.000 |
| P1 local mean | 0.440 | 0.469 | 0.763 | 0.912 | — | — | 2.868 | 0.875 |
| P2 fixed scale | 0.419 | 0.478 | 0.750 | 0.893 | — | — | 2.397 | 0.732 |
| P3 state AR | 0.419 | 0.463 | 0.739 | 0.884 | — | — | 2.412 | 0.736 |
| P6 block | 0.649 | 0.910 | 0.992 | 0.999 | — | — | 10.041 | 3.065 |
| P6 lead 1 | **0.413** | 0.588 | 0.849 | **0.948** | — | — | 3.090 | 0.943 |
| D4 scaled | 0.446 | 0.691 | 0.915 | 0.974 | — | — | 4.064 | 1.240 |

The primary is under-covered at every level. The 95% miss is 0.060, outside the frozen ±0.05 tolerance. The standardized-error variance is 1.73, above the 1.4 upper bound; the intervals are too narrow conditional on the realized errors. The PIT error of 0.0145 passes its threshold, which shows why a single aggregate PIT statistic is not enough: it can look good while the tails are too light.

**Gate E: FAIL.**

## 2. Coverage by horizon

Primary, averaged over landmarks and neurons:

| Horizon | Coverage 50 | Coverage 80 | Coverage 95 |
|---|---:|---:|---:|
| H1 | 0.542 | 0.822 | 0.947 |
| H5 | 0.539 | 0.815 | 0.937 |
| H10 | 0.500 | 0.781 | 0.916 |
| H20 | 0.471 | 0.746 | 0.890 |
| H50 | 0.439 | 0.712 | 0.866 |
| H100 | 0.413 | 0.676 | 0.836 |

The under-coverage grows with horizon. At H1 the 95% interval is close to nominal (0.947), but the 50% interval already covers 0.542. By H100 the 95% interval covers only 0.836.

This pattern is consistent with a model that underestimates the growth of uncertainty: the one-step innovation is approximately right, but the forecast cone does not widen enough, or the conditional mean drifts more than the model allows.

The maximum absolute horizon deviation is **0.124**, above the frozen 0.12 gate. So the primary fails Gate E both at H20 and in the horizon profile.

## 3. Corrected population model

`P6_adaptive_lead1` has a very different profile:

| Horizon | Coverage 50 | Coverage 80 | Coverage 95 |
|---|---:|---:|---:|
| H1 | 0.516 | 0.808 | 0.944 |
| H5 | 0.621 | 0.887 | 0.972 |
| H10 | 0.602 | 0.870 | 0.961 |
| H20 | 0.588 | 0.849 | 0.947 |
| H50 | 0.571 | 0.836 | 0.945 |
| H100 | 0.563 | 0.841 | 0.948 |

The 95% interval is close to nominal at every horizon. The 50% interval over-covers by 5-12 points. The correction removes the catastrophic overdispersion but leaves the distribution too concentrated in the center, or too heavy-tailed relative to a Gaussian, depending on the level.

This is why a model can "pass 95%" and still fail calibration. Gate E requires all three levels.

## 4. Historical model

The historical block-50 bootstrap is over-covered at every level:

| Horizon | Coverage 50 | Coverage 80 | Coverage 95 |
|---|---:|---:|---:|
| H1 | 0.630 | 0.908 | 0.983 |
| H5 | 0.836 | 0.982 | 0.999 |
| H10 | 0.880 | 0.990 | 0.999 |
| H20 | 0.910 | 0.992 | 0.999 |
| H50 | 0.932 | 0.995 | 0.999 |
| H100 | 0.942 | 0.995 | 0.998 |

The width at H20 is 10.04, more than four times the primary. The model is not sharp and its intervals are not informative. This is the failure the experiment was designed to explain.

## 5. Per-session calibration

Primary, H20:

| Session | Coverage 50 | Coverage 80 | Coverage 95 | CRPS | Width 95 |
|---|---:|---:|---:|---:|---:|
| 0 | 0.445 | 0.722 | 0.879 | 0.372 | 2.167 |
| 1 | 0.510 | 0.790 | 0.922 | 0.363 | 2.346 |
| 2 | 0.499 | 0.769 | 0.909 | 0.505 | 2.683 |
| 3 | 0.442 | 0.718 | 0.864 | 0.381 | 2.122 |
| 4 | 0.463 | 0.731 | 0.878 | 0.479 | 2.654 |

No session reaches nominal 95% coverage. Session 1 is closest (0.922); session 3 is worst (0.864). The per-session CRPS varies by a factor of 1.4, so the pooled number hides meaningful heterogeneity.

These are repeated sessions, not independent animals. The spread across sessions is a warning about treating the pooled mean as a stable estimate.

## 6. PIT and sharpness

The PIT histogram for the primary has an aggregate error of 0.0145, which is small. But the tails are not uniform: the under-coverage at 95% is consistent with too little mass in the tails. A histogram distance can be dominated by the central bins, so it should not be used alone.

The primary's 95% width is 2.385, 0.731 times persistence. It is sharper than persistence, which is desirable only if calibration holds. Here the sharpness is bought by under-coverage.

## 7. D4 variance scaling

D4 fits a global factor on validation. The factors are unstable (1.84 to 4.00), and on test D4 gives 0.691/0.915/0.974 coverage with width 4.064 and CRPS 0.446. It over-covers at 50% and 80% and is worse than the primary on CRPS. A global inflation factor is not a calibration mechanism; it is a diagnostic that shows how much variance would need to be added to fix the 95% level.

The factor is not selected as a model.

## 8. Interpretation

The primary model's marginal distribution is **not calibrated**. It improves CRPS over persistence and local mean, but it is too narrow at 95%, too narrow at 80%, and approximately calibrated at 50% only by coincidence at the pooled level. The horizon trend is the clearest evidence: the model's uncertainty grows too slowly.

The corrected population model gets the 95% level right but over-covers at 50%. The historical model is severely overdispersed. No model passes Gate E.

## 9. Reproduction

```powershell
python scripts/uncal_evaluation.py
python scripts/uncal_figures.py
python scripts/uncal_summarize.py
```

Relevant artifacts: `coverage_by_horizon.csv`, `coverage_summary.csv`, `calibration_h20.csv`, `calibration_curves.csv`, `per_neuron_h20.csv`, `per_session_calibration.csv`, `figures/uncal_coverage_by_horizon.png`, `figures/uncal_pit.png`.

## Related documents

- [Probabilistic models](UNCAL_PROBABILISTIC_MODELS.md)
- [Trajectory fidelity](UNCAL_TRAJECTORY_FIDELITY.md)
- [Results](UNCAL_RESULTS.md)
- [Limitations](UNCAL_LIMITATIONS.md)
