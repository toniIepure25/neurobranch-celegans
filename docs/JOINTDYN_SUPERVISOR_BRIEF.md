# JOINTDYN supervisor brief

**Date:** 2026-11-02  
**Branch:** `research/joint-dynamics-calibration`  
**Decision:** retain the corrected population model as a useful **conditional-mean forecasting result**, not as a validated generative simulator.

## Bottom line

Correcting the innovation pool was necessary and reproducible. The historical overdispersed model is confirmed pathological: H20 energy 4.609, 95% coverage 0.999. The corrected lead-1 population model has H20 energy 2.700, full-neuron 95% coverage 0.947 and trajectory energy 2.060.

The corrected model beats independent AR on joint scores:

- H20 energy improvement 0.446, 95% interval [0.183, 0.660];
- trajectory energy improvement 0.375, [0.218, 0.522].

It does **not** beat independent AR on marginal per-neuron CRPS:

- improvement 0.0048, [-0.0083, 0.0157].

The benefit comes from the population conditional mean, not the covariance. Full covariance makes every joint diagnostic worse (H20 energy 3.011, covariance error 0.762).

## Why it is not validated

- 50% calibration fails: full-neuron 0.588; 10-D PCA 0.546.
- Neural covariance error is 0.604, above the frozen 0.5 gate.
- A validation-fitted Student-t correction is worse than the uncorrected model (energy 2.829 vs 2.700).
- Parameter uncertainty worsens dispersion (energy 2.807, trajectory 2.271, 95% coverage 0.969).
- Session 1 is negative and session 2 is ambiguous.
- The advantage over persistence is not robust to block-10 uncertainty.
- The test split has been reused; five sessions are not five animals.

**Final label: `PROBABILISTIC CALIBRATION NOT ESTABLISHED`.**

## What is scientifically useful

1. The one-step causal residual pool is now correct and reproducible.
2. The population mean carries real multivariate information in this preparation.
3. Full covariance is not estimable well enough to help at this sample size.
4. The remaining error is covariance shape and observation/process confounding, not serial innovation dependence.
5. The negative results are specific and actionable.

## Recommendation

Do not run another architecture search on this test split. The next credible step is an external animal or stimulus holdout with a prespecified observation model and a covariance estimated without test reuse. A perturbation experiment should be designed only with independent ground truth and safety review.

No stimulation target or causal claim is produced.

**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

Biological validation: `NOT_EVALUABLE`.

## Reproduction

```powershell
python scripts/jd_reproduce_historical.py
python scripts/jd_evaluate.py --members 100 --stride 10
python scripts/jd_calibration.py
python scripts/jd_parameter.py
python scripts/jd_summarize.py
python scripts/jd_figures.py
python scripts/jd_examples.py
python scripts/validate_jointdyn_reports.py
```

Primary evidence: `reports/simulator/joint_dynamics/score_summary.csv`, `paired_scores.csv`, `gates.json`, `reproduction_check.csv`.
