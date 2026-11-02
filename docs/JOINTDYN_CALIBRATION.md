# JOINTDYN calibration

**Result: G4 FAIL. The corrected population model is not calibrated at 50%, 80% and 95% simultaneously, and the prespecified Student-t correction makes the test distribution worse.**

## 1. Two calibration spaces

Two different questions are reported separately:

- **marginal / full-neuron:** average coverage over all 103 recorded neurons. This matches the previous report and is the reproduction endpoint.
- **joint / 10-D PCA:** coverage of each principal component after the fixed training projection. This is the joint evaluation representation used for energy and trajectory scores.

They disagree, and that disagreement is itself diagnostic. A single global variance scale cannot fix both.

## 2. Corrected population model, H20

| Space | 50% | 80% | 95% | Width 95 |
|---|---:|---:|---:|---:|
| Full neuron (J5) | 0.588 | 0.849 | 0.947 | 4.736 |
| 10-D PCA (J5) | 0.546 | 0.830 | 0.938 | — |

The full-neuron model **over-covers** at 50% by 8.8 percentage points. The joint PCA model **under-covers** at 50% by 4.6 points and at 95% by 1.2 points. This is a shape/covariance mismatch, not a simple scalar spread error.

PIT calibration error at H20 is 0.0134, which is numerically small. PIT is therefore not sufficient on its own: interval coverage fails even when the average PIT histogram is close to uniform.

Width ratio versus persistence is 0.843. The population model is narrower than persistence but still mis-shaped.

## 3. Coverage by horizon, 10-D PCA

| Horizon | 50% | 80% | 95% |
|---|---:|---:|---:|
| H1 | 0.510 | 0.807 | 0.949 |
| H5 | 0.609 | 0.890 | 0.975 |
| H10 | 0.571 | 0.841 | 0.958 |
| H20 | 0.546 | 0.830 | 0.938 |
| H50 | 0.577 | 0.827 | 0.941 |
| H100 | 0.567 | 0.822 | 0.942 |

The maximum absolute horizon deviation is **0.109**, above the frozen 0.10 gate. The model is closest at H1, over-covers around H5, and under-covers at the upper tail by H20-H100.

## 4. Prespecified correction methods

### D1 - historical corrected lead-1, unchanged

This is J5. It remains the best-calibrated candidate but fails 50% and the max-horizon deviation.

### D2 - lead-1 Gaussian with regularized full covariance

This is J3. It is dramatically overdispersed:

| Space | 50% | 80% | 95% | H20 energy |
|---|---:|---:|---:|---:|
| Full neuron | 0.767 | 0.957 | 0.992 | 3.011 |
| 10-D PCA | 0.764 | 0.958 | 0.991 | 3.011 |

Full covariance is rejected. The over-coverage is not a small calibration adjustment; it reflects unstable estimation of a 103-dimensional covariance from roughly 264 one-step residuals.

### D3 - compact temporal innovation

**NOT FITTED.** Stage C found population ACF1 = 0.0217 and ACF5 = 0.0078. The prespecified justification threshold was not met. Fitting a temporal model would be an architecture search without evidence.

### D4 - one Student-t scale correction fitted on validation

Method: multivariate Student-t with 5 degrees of freedom, normalized to unit covariance, plus a global scale. The scale grid `{0.85, 0.95, 1.0, 1.1, 1.25}` was scored on validation landmarks only, selecting the scale with the smallest maximum absolute 50/80/95 coverage error.

Validation results:

| Scale | 50% | 80% | 95% | Max abs error |
|---:|---:|---:|---:|---:|
| 0.85 | 0.650 | 0.888 | 0.975 | 0.150 |
| 0.95 | 0.692 | 0.915 | 0.984 | 0.192 |
| 1.00 | 0.715 | 0.927 | 0.988 | 0.215 |
| 1.10 | 0.757 | 0.953 | 0.992 | 0.257 |
| 1.25 | 0.808 | 0.970 | 0.995 | 0.308 |

Chosen scale: **0.85**. Even on validation, the full-covariance base is already overdispersed; shrinking the scale cannot repair the shape.

Test results:

| Model | 50% | 80% | 95% | H20 energy | Trajectory energy |
|---|---:|---:|---:|---:|---:|
| J5 D1 | 0.546 | 0.830 | 0.938 | 2.700 | 2.060 |
| J3 D2 | 0.764 | 0.958 | 0.991 | 3.011 | 2.277 |
| D4 Student-t | 0.690 | 0.921 | 0.984 | 2.829 | 2.141 |

D4 improves over D2 but is worse than D1 on every score and still badly over-covers. It is rejected. The validation fit was honest: no test outcome entered the scale choice.

## 5. Distribution shape

- Heavy tails alone do not fix the problem. D4 has heavier tails than Gaussian and still over-covers at 50%.
- The failure is horizon-dependent and space-dependent.
- Centeredness is adequate; mean RMSE is not the issue.
- Conditional skewness was not modelled because the residual sample does not support a stable conditional shape estimate.
- Forecast error variance and ensemble spread disagree in the full covariance model, consistent with overfitting.

## 6. What would be needed

A calibrated model needs a better observation/state-space formulation or a lower-dimensional covariance with a stable held-out estimate. A global scale, a single heavy-tail family, or a per-level coverage patch is not sufficient. No test-set calibration was performed.

## 7. Artifacts

- `calibration_h20.csv`
- `calibration_models.csv`
- `calibration_curves.csv`
- `coverage_by_horizon.csv`
- `coverage_summary.csv`
- `d4_validation_grid.csv`
- `d4_validation_summary.csv`
- `d4_decision.json`
- `d4_scores_by_landmark.csv`
- `d4_summary.csv`
- `figures/jd_calibration.png`
- `figures/jd_pit.png`
