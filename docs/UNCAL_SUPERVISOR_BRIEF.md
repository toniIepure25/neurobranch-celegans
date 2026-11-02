# Supervisor brief: neural uncertainty and calibration

## Bottom line

The earlier probabilistic adaptive simulator was severely overdispersed for a concrete, reproducible reason: it treated multi-step forecast errors as one-step innovations and resampled them in blocks. That defect inflated H20 variance by about 2.8x relative to a white-noise propagation, and a near-unit-root, non-normal population operator amplified the result. The ensemble code itself was correct.

A smaller, causal independent AR model with lead-1 diagonal innovations improves the proper marginal score but **does not calibrate**. It wins H20 CRPS against persistence and local mean, then fails 95% coverage, standardized-error variance, and joint trajectory fidelity. The exact status is:

**PROBABILISTIC CALIBRATION NOT ESTABLISHED**

No model in this experiment is a validated generative simulator.

## What we can now say

- Historical H20 energy 4.6046 reproduces as 4.6088 (0.09% difference), so the failure is not version drift.
- Analytical covariance and Gaussian Monte Carlo agree to about 0.04%; the propagation implementation is correct.
- The block/all-lead pool inflates variance by 3.23x relative to the lead-1 pool. Correcting the pool cuts energy from 4.609 to 2.700 and moves 95% coverage from 0.999 to 0.948.
- The corrected population model still over-covers at 50% (0.588), has covariance relative error 0.492, and is not the prespecified primary.
- The primary independent AR has H20 CRPS 0.418, beats P0 (0.452) and P1 (0.440) with paired intervals above zero, but has 50/80/95 coverage 0.471/0.746/0.890 and trajectory energy 2.435, worse than persistence 2.286.
- The best joint trajectory score among the corrected models is P6 lead 1 at 2.060, but its total std ratio is 0.638 and its lag-1 error is 0.409; it is not a validated generator.

## What we cannot say

- That the worm's neural dynamics are linear or that the 10-D PCA is sufficient.
- That process noise has been separated from calcium measurement noise. It has not.
- That the model generalizes to new animals, labs, or stimuli. It has not been tested there.
- That any neuron should be stimulated or suppressed. No perturbation was run.
- That the five repeated sessions are five independent animals. They are not.

Any model-dependent virtual perturbation must carry:

**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

Biological validation: `NOT_EVALUABLE`.

## Decision

Gate D (proper score) passes. Gate E (marginal calibration) fails. Gate F (joint trajectory) fails. Gate G (generative credibility) fails. Gate H (biological intervention) is `NOT_EVALUABLE`.

The correct decision is **not** to promote the smallest model as a simulator. It is to keep the negative result, use the independent AR only as a marginal forecasting baseline, and require a joint-state model with an explicit observation model before any generative or perturbation claim.

## Why this is useful

The experiment converts a vague "the simulator is too uncertain" into a measured decomposition:

- pool construction and block resampling: dominant;
- near-unit-root non-normal amplification: secondary but real;
- state heteroscedasticity and low-rank covariance: not supported by held-out evidence;
- global variance scaling: hides the defect, worsens sharpness and CRPS;
- remaining under-coverage: likely mean drift, parameter uncertainty, or unmodeled dependence, not a single scaling constant.

That is a sharper basis for the next experiment than another architecture sweep.

## Next step

Fit a joint-state model with an explicit observation model and parameter-uncertainty ensemble, and evaluate it on held-out covariance and new animals. The falsifiable target is 50/80/95 coverage within tolerance **and** covariance relative error below 0.50 at H20. Do not add low-rank or nonlinear components unless the held-out residual audit justifies them.

## Related documents

- [Results](UNCAL_RESULTS.md)
- [Variance propagation](UNCAL_VARIANCE_PROPAGATION.md)
- [Calibration](UNCAL_CALIBRATION.md)
- [Trajectory fidelity](UNCAL_TRAJECTORY_FIDELITY.md)
- [Limitations](UNCAL_LIMITATIONS.md)
- [Next experiments](UNCAL_NEXT_EXPERIMENTS.md)
