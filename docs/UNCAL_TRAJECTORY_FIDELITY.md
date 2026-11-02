# Joint trajectory fidelity

This document answers Stage F for the joint target. A calibrated marginal forecast gives a credible distribution for one neuron at one horizon. A credible **trajectory** distribution must also reproduce the dependence across neurons, across time, and across the path as a whole. Those are different objects.

## 1. Why marginal calibration is not enough

The primary model treats neurons as independent conditional on their own past. It can have a reasonable per-neuron CRPS and still generate paths whose population geometry is wrong: too little cross-neuron covariance, too-fast or too-slow temporal decorrelation, or a path that does not look like a real continuation.

The joint score used here is the energy score of the flattened `H20 x 10` PCA trajectory, normalized by `sqrt(20)`. The observation is the realized held-out path in the same space. Lower is better. The score is computed per landmark and averaged over 175 landmarks.

## 2. H20 trajectory scores

| Model | Mean trajectory energy | Median |
|---|---:|---:|
| P0 persistence | **2.286** | 2.005 |
| P1 local mean | 3.177 | 2.807 |
| P2 primary | 2.435 | 2.128 |
| P3 state AR | 2.417 | 2.126 |
| P6 adaptive block | 3.211 | 3.080 |
| P6 adaptive lead 1 | **2.060** | 1.801 |

The primary is worse than persistence. Paired over landmarks:

| Comparison | Mean improvement | 95% interval | Fraction better |
|---|---:|---:|---:|
| P2 vs P0 | **-0.149** | **[-0.298, -0.018]** | 0.469 |
| P2 vs P1 | +0.742 | [0.549, 0.973] | 0.874 |
| P2 vs P6 lead 1 | -0.375 | [-0.531, -0.209] | 0.291 |
| P6 lead 1 vs P0 | +0.226 | [0.088, 0.344] | 0.623 |
| P6 lead 1 vs P6 block | +1.151 | [1.022, 1.292] | 0.983 |

Positive improvement means the reference is worse. The primary's interval against persistence is entirely below zero: this is not a marginal loss, it is a consistent joint disadvantage.

**Gate F fails on trajectory energy.**

## 3. Variance and covariance

At H20:

| Model | Total std ratio | Covariance relative error | Lag-1 autocorr error | OOD fraction |
|---|---:|---:|---:|---:|
| P0 persistence | 0.657 | 0.517 | 0.192 | — |
| P1 local mean | 0.584 | 0.562 | 0.272 | — |
| P2 primary | 0.517 | 0.729 | 0.427 | 0.0756 |
| P3 state AR | 0.510 | 0.718 | 0.423 | — |
| P6 block | 1.814 | 0.862 | 0.464 | — |
| P6 lead 1 | 0.638 | 0.492 | 0.409 | — |

The primary's total standard deviation is **0.517** times the observed, below the frozen lower bound of 0.75. Its covariance relative error is **0.729**, above the 0.50 upper bound. Its lag-1 autocorrelation error is **0.427**, above the 0.25 upper bound. OOD is 0.076 and finite, which passes, but that does not rescue the other three failures.

No model reproduces the joint covariance: the best relative error is P6 lead 1 at 0.492, barely at the threshold, and its total std ratio is 0.638, still too narrow. P0 has the best trajectory energy but its covariance error is 0.517.

## 4. What the joint failure looks like

The primary's marginal intervals are too narrow at 95% and its trajectory variance is too small. The lag-1 autocorrelation error is large, consistent with a path that decorrelates at the wrong rate. The covariance error says the cross-neuron geometry is wrong, which is expected for a model with no cross-neuron innovation.

The corrected population model is better on the joint score than the primary and better than persistence, but its 50% marginal coverage is 0.588 and its covariance error is at the boundary. It also inherits the near-unit-root, non-normal operator, which can produce transient amplification even when the long-run variance is scaled correctly.

## 5. Executed example

`example_trajectories.csv`, `example_bands.csv`, and `example_observed.csv` contain an executed ensemble at one held-out landmark (session 0, landmark 2750), with 200 members and the observed continuation. `figures/uncal_example_trajectory.png` shows the population mean fan for the primary and the historical model.

The example is not decorative: the bands come from the same rollout code used in evaluation. It is a single landmark, so it illustrates the mechanism rather than estimating calibration.

## 6. Variogram

The variogram score compares the model's expected pairwise absolute differences with the observed path. At H20:

- P2: 0.265
- P0: 0.248
- P6 lead 1: 0.218
- P6 block: 0.699

The primary is worse than persistence and much better than the historical block model. The variogram is sensitive to the temporal dependence structure, and the primary's error is consistent with the autocorrelation failure.

## 7. Gate F decision

Gate F requires all of:

- trajectory energy improvement over P0 with 95% lower bound > 0: **fail** (interval entirely below zero);
- total std ratio in [0.75, 1.35]: **fail** (0.517);
- lag-1 autocorrelation error < 0.25: **fail** (0.427);
- covariance relative error < 0.50: **fail** (0.729);
- OOD < 0.20: pass (0.076);
- finite: pass.

**Gate F: FAIL.**

Gate G requires D, E and F. D passes; E and F fail. **Gate G: FAIL.**

## 8. Stop label

For the primary, the exact label is:

**PROBABILISTIC CALIBRATION NOT ESTABLISHED**

The corrected population model is a diagnostic that improves the joint score and the 95% marginal level, but it does not pass 50% coverage, total variance, autocorrelation or covariance. It does not earn the label `CALIBRATED MARGINAL FORECASTING WITHOUT VALIDATED GENERATIVE TRAJECTORIES` either, because its marginal calibration is not established at all levels.

No model in this experiment is a validated generative simulator.

## 9. Perturbation status

No perturbation screen was launched. If any model-dependent virtual perturbation is shown from this work, it must carry:

**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

Biological validation is `NOT_EVALUABLE`. No neuron ranking, stimulation target, or control claim is supported.

## 10. Reproduction

```powershell
python scripts/uncal_evaluation.py
python scripts/uncal_examples.py
python scripts/uncal_figures.py
```

Relevant artifacts: `trajectory_scores.csv`, `trajectory_paired.csv`, `covariance_h20.csv`, `variance_summary.csv`, `example_trajectories.csv`, `example_bands.csv`, `figures/uncal_example_trajectory.png`.

## Related documents

- [Calibration](UNCAL_CALIBRATION.md)
- [Results](UNCAL_RESULTS.md)
- [Limitations](UNCAL_LIMITATIONS.md)
- [Next experiments](UNCAL_NEXT_EXPERIMENTS.md)
