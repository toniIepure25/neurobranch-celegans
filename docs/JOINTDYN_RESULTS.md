# JOINTDYN results

**Final label: `PROBABILISTIC CALIBRATION NOT ESTABLISHED`**

The corrected population model adds useful joint trajectory information beyond independent AR and persistence in the fixed 10-D representation. The advantage comes from conditional-mean population dynamics, not from the innovation covariance. Calibration and covariance fidelity still fail, so no generative simulator is validated.

This is an exploratory repeated-session reanalysis. It is not a systematic review, not an external validation, and not a biological intervention study.

## 1. Scores

| Model | H20 energy | H20 trajectory | Full-neuron CRPS | 50% | 80% | 95% | Covariance error | Autocorr error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| J0 persistence | 2.950 | 2.286 | 0.4524 | 0.494 | 0.791 | 0.936 | 0.630 | 0.195 |
| J1 independent diag | 3.146 | 2.435 | 0.4182 | 0.471 | 0.746 | 0.890 | 0.505 | 0.174 |
| J2 population diag | **2.700** | **2.058** | 0.4137 | 0.586 | 0.846 | 0.944 | 0.572 | 0.160 |
| J3 population full | 3.011 | 2.277 | 0.4594 | 0.767 | 0.957 | 0.992 | 0.762 | 0.213 |
| J4 independent full | 3.007 | 2.359 | 0.4242 | 0.560 | 0.820 | 0.934 | 0.517 | 0.189 |
| J5 population lead-1 | **2.700** | **2.060** | 0.4134 | 0.588 | 0.849 | 0.947 | 0.604 | 0.164 |

The J5 full-neuron coverage reproduces the previous report. The J5 10-D PCA coverage is 0.546/0.830/0.938, which is the joint representation used for energy.

## 2. Paired evidence

| Contrast | Metric | Improvement | 95% interval |
|---|---|---:|---:|
| J5 vs J1 | H20 energy | 0.446 | [0.183, 0.660] |
| J5 vs J1 | trajectory | 0.375 | [0.218, 0.522] |
| J5 vs J0 | H20 energy | 0.250 | [0.020, 0.427] |
| J5 vs J0 | trajectory | 0.226 | [0.089, 0.337] |
| J5 vs J1 | full-neuron CRPS | 0.0048 | [-0.0083, 0.0157] |
| J5 vs J1 | all-neuron energy | 0.203 | [-0.015, 0.412] |
| J5 vs J1 | variogram | 0.0477 | [0.0165, 0.0728] |

The marginal CRPS and all-neuron energy intervals include zero. The 10-D multivariate and trajectory advantages do not.

## 3. Factorial

| Metric | Mean main effect | Covariance main effect | Interaction |
|---|---:|---:|---:|
| H20 energy | +0.221 | -0.086 | +0.449 |
| H20 trajectory | +0.230 | -0.071 | +0.296 |

Population mean helps. Full covariance hurts. The full-covariance model is overdispersed and has the worst covariance error.

## 4. Sessions

J5 vs J1 H20 energy:

| Session | Improvement | 95% interval |
|---:|---:|---:|
| 0 | +0.638 | [0.276, 1.007] |
| 1 | -0.152 | [-0.416, 0.042] |
| 2 | +0.243 | [-0.237, 0.683] |
| 3 | +0.752 | [0.159, 1.362] |
| 4 | +0.730 | [0.070, 0.957] |

Four of five sessions are positive. Session 1 is negative and session 2 is ambiguous. The five sessions are repeated recordings, not five animals.

Against persistence, only sessions 0, 3 and 4 are positive; the block-10 paired interval crosses zero. The population advantage over persistence is therefore less robust than the advantage over independent AR.

## 5. Gates

| Gate | Status | Evidence |
|---|---|---|
| G1 reproduction | **PASS** | max relative difference 5.5e-9 |
| G2 population information | **PASS** | +0.446 [0.183, 0.660] |
| G3 innovation validity | **PASS** | |ACF1| 0.0217, |ACF5| 0.0078 |
| G4 marginal calibration | **FAIL** | 50% full 0.588; max horizon deviation 0.109 |
| G5 joint trajectory | **FAIL** | covariance 0.604 > 0.5; energy and autocorr pass |
| G6 robustness | **PASS** | 4/5 sessions; but persistence sensitivity is fragile |
| G7 generative credibility | **FAIL** | requires G4 and G5 |
| G8 biological intervention | **NOT_EVALUABLE** | no intervention data |

## 6. Uncertainty experiments

- **D4 Student-t:** chosen scale 0.85 on validation; test 0.690/0.921/0.984, energy 2.829, trajectory 2.141. Worse than D1.
- **D3 temporal innovation:** not fitted; Stage C did not justify it.
- **Parameter ensemble:** 2.807 energy, 2.271 trajectory, 0.615/0.883/0.969 coverage. Worse than the diagonal population model. Stop rule applied.

## 7. Sensitivity

| Representation | J5 | J1 | J5 advantage |
|---|---:|---:|---:|
| 10-D PCA | 2.700 | 3.146 | positive, CI > 0 |
| all neurons | 6.847 | 7.050 | positive, CI includes 0 |
| exclude top-5 variance neurons | 6.772 | 6.973 | positive, CI includes 0 |
| top-20 variance neurons | 2.498 | 2.588 | positive |

The 10-D joint representation is the strongest result. The all-neuron result is not statistically resolved.

## 8. Answers to the 18 questions

1. **Are the corrected population results reproducible?** Yes. Energy, full-neuron CRPS and full-neuron coverage reproduce at machine precision; the maximum relative difference is 5.5e-9.
2. **Is the apparent joint-score advantage statistically supported within the available sessions?** In the 10-D PCA, yes for J5 vs J1 energy and trajectory; it is exploratory because the test split has been reused.
3. **Does the population model add useful information beyond independent AR?** Yes for multivariate and trajectory scores; no reliable improvement in marginal per-neuron CRPS.
4. **Does the benefit come from mean, covariance or temporal dependence?** Conditional mean. The mean main effect is +0.221; the full-covariance effect is -0.086; residual temporal dependence is negligible.
5. **Are lead-1 residuals valid innovations?** As one-step causal residuals, yes. Their median ACF1 is 0.022 and ACF5 is 0.008. The bootstrap still assumes exchangeability, which is an approximation.
6. **What temporal innovation structure remains unexplained?** Very little serial structure. The unresolved structure is cross-sectional covariance instability, mild nonstationarity and observation noise.
7. **Can intervals be calibrated at 50%, 80% and 95%?** No. Full-neuron 50% is 0.588; PCA 50% is 0.546; D4 is worse.
8. **Does the model reproduce realistic neural covariance?** No. Covariance error is 0.604, above the 0.5 gate; full covariance is 0.762.
9. **Does it reproduce temporal trajectory structure?** Partially: trajectory energy and autocorrelation pass; covariance fails; occupancy is better than independent AR.
10. **Does parameter uncertainty improve calibration?** No. It worsens energy, trajectory and coverage.
11. **Are positive findings consistent across sessions?** Four of five for J5 vs J1; not all, and the persistence contrast is weaker.
12. **Is the model numerically stable at long horizons?** All ensembles are finite at H100; J5 maximum absolute PC value is 26.2, so stable but with large excursions.
13. **Does it outperform probabilistic persistence on joint trajectory scores?** Yes, 2.060 vs 2.286, interval [0.089, 0.337].
14. **Can it reproduce distributions of alternative futures?** It reproduces trajectory energy better, but not calibrated covariance; this is not a faithful generative distribution.
15. **Are results robust to evaluation space?** Positive in PCA and top-20 neurons; all-neuron energy is unresolved. Full covariance is not robust.
16. **Which conclusions are exploratory because of test reuse and missing animals?** All test-split conclusions. Five repeated sessions are not five animals; there is no external confirmation.
17. **Is a generative neural simulator scientifically validated?** No. `PROBABILISTIC CALIBRATION NOT ESTABLISHED`.
18. **Most valuable next experiment?** An external animal/stimulus holdout with a prespecified observation model and a low-dimensional covariance estimated without test reuse. Do not run more architecture searches on this test split.

**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

Biological validation: `NOT_EVALUABLE`.

## 9. Negative findings retained

- Full covariance is worse than diagonal covariance.
- Marginal CRPS does not improve reliably.
- All-neuron energy does not improve reliably.
- 50% calibration fails in both representations.
- Neural covariance fails.
- D4 Student-t is worse than D1.
- Parameter uncertainty worsens dispersion.
- Session 1 and session 2 do not support the population advantage versus independent AR.
- The population advantage over persistence is not robust to block-10 uncertainty.

## 10. Artifacts

`reports/simulator/joint_dynamics/` contains the score tables, paired and session tables, calibration tables, covariance and autocorrelation exports, uncertainty tables, model cards, configurations, gate reports, manifest, and eight scientific figures plus one interactive HTML trajectory view.
