# Results: neural uncertainty and calibration

**Study status:** exploratory repeated-session reanalysis. Not a systematic review. Not a new external validation. Not a causal intervention study.

**Primary model:** `P2_independent_ar_lead1`, fixed before final evaluation.

**Exact stop label:** `PROBABILISTIC CALIBRATION NOT ESTABLISHED`

## 1. Executive result

The historical overdispersion is real and reproducible. Its dominant measurable cause is the construction of the innovation pool: multi-step residuals from a fixed fit were pooled across leads 1-50 and resampled as blocks, so the ensemble injected serially dependent, horizon-inflated errors at every step. The population operator is also near unit root and non-normal, which amplifies even correctly scaled one-step noise.

Correcting the pool works. The historical H20 energy falls from **4.609** to **2.700** and 95% coverage moves from **0.999** to **0.948**. But the corrected model over-covers at 50% (0.588), does not reproduce the joint covariance, and is not the prespecified primary.

The prespecified independent AR improves the primary proper score: H20 CRPS **0.418** versus **0.452** for persistence and **0.440** for local mean, with paired 95% intervals above zero. It does not calibrate: 50/80/95 coverage is **0.471/0.746/0.890**, the standardized-error variance is **1.73**, and its joint trajectory score is significantly worse than persistence.

**Gates:** A pass, B pass, C pass for the primary only, D pass, E fail, F fail, G fail, H `NOT_EVALUABLE`.

## 2. Reproduction

The historical comparator was rerun with its original block-50 residual pool and 1000-frame history:

| Quantity | Historical | Rerun | Relative difference |
|---|---:|---:|---:|
| H20 energy | 4.6046 | 4.6088 | 0.09% |
| H20 CRPS | 0.6481 | 0.6492 | 0.16% |
| Width 95 | 9.9670 | 10.0406 | 0.74% |
| Coverage 50 | 0.9040 | 0.9100 | 0.66% |
| Coverage 80 | 0.9915 | 0.9921 | 0.06% |
| Coverage 95 | 0.9988 | 0.9989 | 0.01% |

Maximum relative difference **0.74%**, within the prespecified 1% Gate A tolerance. No historical artifact was overwritten.

## 3. Variance attribution

| Comparison | Median ratio |
|---|---:|
| block bootstrap all leads / analytical all leads | **2.762** |
| block bootstrap lead 1 / analytical lead 1 | **1.035** |
| block bootstrap all leads / block bootstrap lead 1 | **3.233** |
| Gaussian MC / analytical lead 1 | **1.0004** |
| P20 / Q, lead 1 | **2.250** |
| P100 / P20, lead 1 | **1.658** |

Operator: spectral radius **0.991**, `||A^20||_2` **5.92** (max 10.18), non-normality **0.692**. The propagation code is correct; the pool and the operator are the story.

## 4. Model comparison at H20

| Model | Energy | CRPS | Cov 50 | Cov 80 | Cov 95 | Width 95 | Trajectory energy |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 persistence | 2.950 | 0.452 | 0.494 | 0.791 | 0.936 | 3.276 | **2.286** |
| P1 local mean | 3.282 | 0.440 | 0.469 | 0.763 | 0.912 | 2.868 | 3.177 |
| P2 fixed scale | 3.149 | 0.419 | 0.478 | 0.750 | 0.893 | 2.397 | — |
| **P2 primary** | **3.146** | **0.418** | **0.471** | **0.746** | **0.890** | **2.385** | 2.435 |
| P3 state AR | 3.112 | 0.419 | 0.463 | 0.739 | 0.884 | 2.412 | 2.417 |
| P6 block | 4.609 | 0.649 | 0.910 | 0.992 | 0.999 | 10.041 | 3.211 |
| P6 lead 1 | **2.700** | **0.413** | 0.588 | 0.849 | **0.948** | 3.090 | **2.060** |
| D4 scaled | 2.961 | 0.446 | 0.691 | 0.915 | 0.974 | 4.064 | 2.358 |

## 5. Paired primary comparisons

| Metric | Reference | Improvement | 95% interval |
|---|---|---:|---:|
| CRPS | P0 persistence | +0.034 | [0.020, 0.048] |
| CRPS | P1 local mean | +0.022 | [0.008, 0.038] |
| CRPS | P6 block | +0.046 | [0.034, 0.059] |
| Energy | P0 persistence | -0.195 | [-0.413, 0.003] |
| Energy | P1 local mean | +0.137 | [-0.119, 0.419] |
| Trajectory | P0 persistence | -0.149 | [-0.298, -0.018] |
| Trajectory | P6 lead 1 | -0.375 | [-0.531, -0.209] |

Positive improvement means the reference is worse. Gate D passes on CRPS. The joint comparison is significantly worse than persistence.

## 6. Calibration by horizon

Primary:

| Horizon | 50% | 80% | 95% |
|---|---:|---:|---:|
| H1 | 0.542 | 0.822 | 0.947 |
| H5 | 0.539 | 0.815 | 0.937 |
| H10 | 0.500 | 0.781 | 0.916 |
| H20 | 0.471 | 0.746 | 0.890 |
| H50 | 0.439 | 0.712 | 0.866 |
| H100 | 0.413 | 0.676 | 0.836 |

Under-coverage grows with horizon. Maximum horizon deviation **0.124**, above the 0.12 gate.

P6 lead 1:

| Horizon | 50% | 80% | 95% |
|---|---:|---:|---:|
| H1 | 0.516 | 0.808 | 0.944 |
| H5 | 0.621 | 0.887 | 0.972 |
| H10 | 0.602 | 0.870 | 0.961 |
| H20 | 0.588 | 0.849 | 0.947 |
| H50 | 0.571 | 0.836 | 0.945 |
| H100 | 0.563 | 0.841 | 0.948 |

The corrected population model calibrates 95% but over-covers 50%.

## 7. Per-session primary calibration

| Session | 50% | 80% | 95% | CRPS |
|---|---:|---:|---:|---:|
| 0 | 0.445 | 0.722 | 0.879 | 0.372 |
| 1 | 0.510 | 0.790 | 0.922 | 0.363 |
| 2 | 0.499 | 0.769 | 0.909 | 0.505 |
| 3 | 0.442 | 0.718 | 0.864 | 0.381 |
| 4 | 0.463 | 0.731 | 0.878 | 0.479 |

No session is calibrated at 95%. Repeated sessions are not independent animals.

## 8. Joint trajectory

| Model | Trajectory energy | Total std ratio | Covariance error | Lag-1 error | OOD |
|---|---:|---:|---:|---:|---:|
| P0 | 2.286 | 0.657 | 0.517 | 0.192 | — |
| P2 primary | 2.435 | 0.517 | 0.729 | 0.427 | 0.076 |
| P6 lead 1 | 2.060 | 0.638 | 0.492 | 0.409 | — |
| P6 block | 3.211 | 1.814 | 0.862 | 0.464 | — |

The primary is significantly worse than persistence on the joint score. No model has both acceptable variance and covariance.

## 9. Synthetic controls

| System | Result |
|---|---|
| independent Gaussian AR | 95% coverage 0.956; MC/analytic 1.043; noise trace recovery 0.993 |
| temporally correlated innovations | residual ACF1 0.808; i.i.d. model 95% coverage 0.938 (mild under-coverage) |
| state heteroscedasticity | constant-variance model 95% coverage 0.949; scale correlation 0.597 (marginal coverage hides conditional error) |
| low-rank population noise | top-2 variance fraction 0.966; trace recovery 0.999 |
| measurement vs process noise | observed innovation trace 0.730 vs true process 0.180; process variance not identifiable |
| non-normal transient | spectral radius 0.900, max singular value H20 15.51; MC/analytic 1.047 |
| misspecified mean with predictable residuals | residual ACF1 0.151, ACF2 -0.262; 95% coverage 0.943 |

These controls show that the evaluation pipeline can recover known noise and can detect the failure modes that matter. They do not validate the biological model.

## 10. Low-rank decision

Held-out Gaussian likelihood selected diagonal in sessions 0, 2, 3, 4 and rank 3 in session 1. Mean rank 0.6. `build_lowrank_model` is false. The low-rank branch is a negative result, not an omitted analysis.

## 11. D4 diagnostic

Validation factors: 2.974, 1.843, 4.000, 3.794, 2.181 (mean 2.959). Test: CRPS 0.446, coverage 0.691/0.915/0.974, width 4.064, energy 2.961, trajectory 2.358. It over-inflates and worsens CRPS. It is not a repair.

## 12. Answers to the 18 review questions

1. **Why were the historical simulators overdispersed?** Mainly a block-resampled, multi-step residual pool treated as one-step innovation; amplified by a near-unit-root, non-normal population operator.
2. **Was the propagation code wrong?** No. Gaussian Monte Carlo and analytical covariance agree to about 0.04%; the historical result reproduces within 0.74%.
3. **Was the residual pool a valid innovation pool?** No for block-50/all-lead; it has ACF1 0.553 and contains leads 1-50. The independent lead-1 pool is conditionally acceptable, with ACF5 0.195 caveat.
4. **Did the simplest probabilistic baseline work?** Persistence is strong on 95% coverage (0.936) and joint trajectory (2.286); local mean is competitive on CRPS; neither is a generative model.
5. **Did the primary model beat the baselines on the proper score?** Yes on H20 CRPS: +0.034 vs P0 and +0.022 vs P1, both intervals above zero. Gate D passes.
6. **Did the primary model calibrate?** No. 50/80/95 coverage 0.471/0.746/0.890; standardized-error variance 1.73; under-coverage grows with horizon.
7. **Did the primary model produce credible joint trajectories?** No. Trajectory energy 2.435, significantly worse than persistence 2.286; covariance error 0.729; lag-1 error 0.427.
8. **Did correcting the residual pool fix the historical model?** Partially. Energy 4.609 to 2.700 and 95% coverage 0.999 to 0.948, but 50% coverage 0.588 and covariance error 0.492; no.
9. **Was the historical low MSE misleading?** Yes. MSE 0.9406 coexisted with 99.9% coverage and a total std ratio of 3.398; a low mean error is not a probability model.
10. **Is the problem non-normality?** Partly. Spectral radius 0.991 but `||A^20||_2` 5.92 and non-normality 0.692; transient amplification is real, but it is secondary to the pool defect.
11. **Is the problem state heteroscedasticity?** Not established. State-variance held-out R² is near zero/negative; P3 does not improve CRPS or coverage.
12. **Is the problem cross-neuron dependence?** Yes for the joint target. The primary is diagonal and its covariance error is 0.729; low-rank structure was not stable enough to model.
13. **Would a validation-fitted variance factor fix it?** No. It over-covers and worsens CRPS; factors are unstable across sessions.
14. **Is there leakage?** No evidence of future-observation leakage: prequential residuals are causal, model selection and the PCA use training/validation data, and online forecasts use only frames up to the landmark. The evaluation uses overlapping landmarks, which is a dependence issue, not leakage.
15. **Are results stable across sessions?** Qualitatively yes for the failure; quantitatively no. Primary 95% coverage ranges 0.864-0.922; CRPS ranges 0.363-0.505.
16. **Can this model support perturbation claims?** No. No perturbation screen was launched; any virtual perturbation is `MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.` Biological validation is `NOT_EVALUABLE`.
17. **Is there a validated generative simulator?** No. Gate G fails. The exact label is `PROBABILISTIC CALIBRATION NOT ESTABLISHED`.
18. **What is the next experiment?** A joint-state model with an explicit observation model and a held-out covariance target, evaluated on new animals, not another architecture sweep on these five sessions.

## 13. Negative outcomes retained

- The primary improves CRPS but fails calibration and joint fidelity.
- The corrected population model fixes 95% coverage but over-covers 50% and fails covariance/variance/autocorrelation.
- State-conditioned variance and low-rank covariance do not earn their complexity.
- D4 global scaling hides the defect at the cost of sharpness and CRPS.
- No perturbation analysis is justified.

## 14. Gates

| Gate | Status |
|---|---|
| A reproduction | PASS |
| B variance failure diagnosed | PASS |
| C innovation assumptions | PASS FOR PRIMARY ONLY |
| D proper score improvement | PASS |
| E marginal calibration | FAIL |
| F joint trajectory fidelity | FAIL |
| G generative credibility | FAIL |
| H biological intervention | NOT_EVALUABLE |

## 15. Reproduction

See `metadata.json` for the full command sequence. The core sequence is:

```powershell
python scripts/uncal_diagnostics.py
python scripts/uncal_evaluation.py
python scripts/uncal_d4.py
python scripts/uncal_lowrank.py
python scripts/uncal_synthetic.py
python scripts/uncal_reproduction.py
python scripts/uncal_summarize.py
python scripts/uncal_examples.py
python scripts/uncal_figures.py
python scripts/uncal_metadata.py
python scripts/uncal_manifest.py
python scripts/validate_uncal_reports.py
```

Single-threaded BLAS is used for reproducibility.

## Related documents

- [Protocol](UNCAL_RESEARCH_PROTOCOL.md)
- [Variance propagation](UNCAL_VARIANCE_PROPAGATION.md)
- [Innovation validity](UNCAL_INNOVATION_VALIDITY.md)
- [Simple baselines](UNCAL_SIMPLE_BASELINES.md)
- [Probabilistic models](UNCAL_PROBABILISTIC_MODELS.md)
- [Calibration](UNCAL_CALIBRATION.md)
- [Trajectory fidelity](UNCAL_TRAJECTORY_FIDELITY.md)
- [Limitations](UNCAL_LIMITATIONS.md)
- [Supervisor brief](UNCAL_SUPERVISOR_BRIEF.md)
- [Next experiments](UNCAL_NEXT_EXPERIMENTS.md)
