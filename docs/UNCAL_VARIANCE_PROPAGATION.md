# Variance propagation and the overdispersion audit

This document answers Stage A: where did the historical ensemble variance come from? It uses executed diagnostics in `reports/simulator/uncertainty_calibration/operator_diagnostics.csv`, `variance_propagation.csv`, and `variance_decomposition.csv`.

## 1. The propagation identity

For a linear state forecast

\[
x_{h+1} = A x_h + \epsilon_{h+1}, \qquad \epsilon \sim (0,Q),
\]

with independent innovations, the predictive covariance obeys

\[
P_{h+1} = A P_h A^\top + Q, \qquad P_0 = 0.
\]

The ensemble implementation should converge to this recursion as the number of members grows. For the independent AR model, `A` is diagonal and `Q` is diagonal. For the population VAR, `A` is the fitted 103-dimensional operator; `Q` is the innovation covariance used by the ensemble.

This identity is only valid for independent innovations. If innovations are serially correlated, the correct recursion needs cross-time covariance terms or an augmented state that carries the innovation history. A block bootstrap deliberately preserves within-block dependence; treating those draws as independent in the recursion understates the correct variance at longer horizons.

## 2. Implementation check

The analytical and Monte Carlo traces agree. Across landmarks the median ratio of Gaussian Monte Carlo trace to analytical lead-1 trace is **1.0004** (synthetic control: 1.043 for an independent AR; non-normal system: 1.047). There is no factor-of-two bug in the ensemble or the recursion.

This is important: the historical overdispersion is not explained by a broken `A P A^T + Q` implementation.

## 3. The residual pool was not a one-step innovation pool

The historical adaptive residual builder fitted a model once and pooled residuals at leads 1 through 50. Those residuals are **multi-step forecast errors**, not one-step innovations. Their covariance grows with lead because each residual contains accumulated forecast error.

Executed medians across landmarks:

| Quantity | Median |
|---|---:|
| residual covariance trace, lead 1 | 41.33 |
| residual covariance trace, all leads 1-50 | 47.42 |
| analytical lead-1 trace / all-lead trace | 0.821 |

The all-lead pool is larger, but the difference in this pooled statistic is modest. The larger defect appears when the block bootstrap draws a **trajectory** from that pool: the sampled innovation at each step is not independent, and at H20 the accumulated variance is much larger than the analytical white-noise recursion.

Executed variance decomposition:

| Comparison | Median ratio | Interpretation |
|---|---:|---|
| block bootstrap, all leads / analytical, all leads | **2.762** | historical bootstrap inflates variance ~2.8x |
| block bootstrap, lead 1 / analytical, lead 1 | **1.035** | lead-1 bootstrap is nearly consistent with the white-noise recursion |
| block bootstrap, all leads / block bootstrap, lead 1 | **3.233** | most of the historical inflation is the all-lead block pool |
| Gaussian MC / analytical, lead 1 | **1.0004** | implementation agrees |
| P20 / Q, lead 1 | **2.250** | operator amplifies one-step noise over 20 steps |
| P100 / P20, lead 1 | **1.658** | further accumulation to H100 |

The 2.76x and 3.23x factors are the key attribution. A rerun of the historical comparator reproduces the frozen energy 4.6046 as **4.6088** and coverage 0.904/0.992/0.999 as **0.910/0.992/0.999** (maximum relative difference 0.74%). The defect is therefore real and reproducible, not a version drift.

## 4. The operator also amplifies uncertainty

The fitted rolling population operator is stable by spectral radius but strongly non-normal:

| Operator quantity | Median |
|---|---:|
| spectral radius | 0.991 |
| 95th percentile spectral radius | 0.995 (cap) |
| `||A^20||_2` | 5.92 |
| maximum `||A^20||_2` | 10.18 |
| non-normality index | 0.692 |

A spectral radius below one means asymptotic decay, not monotone decay. A non-normal `A` can amplify a perturbation transiently before the asymptotic decay dominates. At H20 the analytical covariance is already **2.25x** the one-step `Q`; by H100 it is another **1.66x** larger. This is a genuine property of the fitted dynamics, not an error.

The practical consequence is that even a perfect one-step innovation model should have a widening predictive cone, and a model fitted only to one-step residuals can still under-cover at H20 if it ignores parameter uncertainty or regime change.

## 5. Candidate mechanisms, one by one

| Candidate | Verdict | Evidence |
|---|---|---|
| innovation scaling bug | **not supported** | Gaussian MC matches analytics within 0.04% |
| repeated residual injection | **supported, dominant** | same bootstrap residual reused at every step; block/all vs lead-1 ratio 3.23 |
| serial dependence in innovations | **supported** | adaptive block-50 lag-1 ACF 0.553; block bootstrap preserves it |
| residual vs innovation confusion | **supported, dominant** | historical pool contains leads 1-50 from one fixed fit |
| accumulation over horizon | **supported** | P20/Q 2.25; P100/P20 1.66 |
| non-normal transient amplification | **supported, secondary** | radius 0.991 but `||A^20||_2` 5.92; non-normality 0.692 |
| observation vs process noise | **not separable in this dataset** | only one observed calcium trace; no measurement model or simultaneous electrophysiology |
| mean misspecification | **possible but not established** | residual mean predictability from state is negative out of sample (R² ≈ -0.38 adaptive, -0.44 independent) |
| state heteroscedasticity | **partial** | state-variance log R² near 0/negative; cross-neuron covariance split-half correlation 0.20 adaptive vs 0.60 independent |
| implementation error | **not supported** | analytical, Gaussian MC and block checks agree; reproduction within 0.74% |

Negative out-of-sample R² for residual mean prediction is not proof that the mean is correctly specified. It means the fitted state-to-residual map did not generalize, which can reflect distribution shift, overfitting, or a genuinely state-independent mean. The honest label is **no positive evidence of mean misspecification**, not **mean is correct**.

## 6. Why the historical model looked good on MSE and bad on probability

The rolling population VAR had lower deterministic H20 MSE (0.9406) than the frozen mean (1.5972). A lower MSE is a statement about the conditional mean, not about the predictive distribution. The same model's ensemble had a total standard-deviation ratio of 3.398 and 95% coverage of 0.999. The historical ensemble was not merely conservative; it was so wide that it could hide mean errors.

The corrected lead-1 population model has a much smaller energy (2.700) but still over-covers at 50% (0.590) and has a poor joint covariance. Fixing the dominant variance defect is necessary but not sufficient.

## 7. Reproducible commands

```powershell
python scripts/uncal_diagnostics.py
python scripts/uncal_reproduction.py
python scripts/uncal_summarize.py
```

The diagnostics script writes `variance_propagation.csv`, `operator_diagnostics.csv`, `innovation_validation.csv`, and `diagnostic_summary.json`. The summary script writes `variance_decomposition.csv` and `variance_decomposition.json`.

## 8. Bottom line

The historical overdispersion is primarily a **residual-pool construction and block-resampling defect**: multi-step residuals were treated as innovations and resampled as blocks. That defect is amplified by a near-unit-root, non-normal population operator. The propagation code itself is correct. Correcting the pool reduces the H20 energy from 4.609 to 2.700, but does not by itself produce a calibrated marginal or joint model.

## Related documents

- [Protocol](UNCAL_RESEARCH_PROTOCOL.md)
- [Innovation validity](UNCAL_INNOVATION_VALIDITY.md)
- [Calibration](UNCAL_CALIBRATION.md)
- [Results](UNCAL_RESULTS.md)
