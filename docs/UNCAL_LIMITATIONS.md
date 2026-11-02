# Limitations

This experiment answers a narrower question than "build a digital twin." The limitations below are part of the result, not a disclaimer added afterward.

## 1. Repeated sessions are not independent animals

The dataset contains five repeated recording sessions. They are treated as repeated measurements, not five independent worms. Pooled coverage and CRPS averages can look more precise than the underlying evidence warrants. Per-session results are reported because the spread across sessions is substantial: primary 95% coverage ranges from 0.864 to 0.922, and CRPS ranges from 0.363 to 0.505.

No animal-level generalization claim is supported.

## 2. No external validation

Every result is on the same Kato WT_NoStim dataset and the same held-out frames used in the earlier work. No new animal, lab, stimulus, or recording modality was evaluated. This is an exploratory repeated-session reanalysis, not a prospective external validation and not a systematic review.

## 3. Observation noise cannot be separated from process noise

Only one calcium trace per neuron is available. There is no simultaneous electrophysiology, no repeated measurement of the same latent state, and no explicit observation model. The synthetic control shows that a single observed innovation covariance confounds process and measurement variance: the process variance is `NOT_IDENTIFIABLE` in that design.

Consequently, statements about "process noise" in this report mean "innovation variance of the observed calcium signal," not a biophysical process variance.

## 4. Linear, low-dimensional mean model

The primary is an independent AR(1) in the original 103-dimensional predictor space; evaluation uses a training-fitted 10-component PCA. It cannot represent nonlinear dynamics, latent switches, cross-neuron interactions in the mean, or behavior-dependent modulation.

A good CRPS for this model would not establish that the true dynamics are linear. A bad CRPS does not rule out a better nonlinear model.

## 5. Ten-component evaluation space

The 10-D PCA retains only part of the variance (historical effective rank 27.4). Scores in this space do not measure errors in the discarded components. The neuron-level marginal scores project back, but the joint trajectory score is a 10-D object.

The PCA is training-fitted, so there is no test leakage through the projection, but the choice of 10 components is inherited from the historical experiment rather than optimized here.

## 6. Overlapping landmarks

The 175 test landmarks are spaced by 10 frames, while H20 windows overlap by 90 frames and H50/H100 overlap more. Landmark-level observations are therefore dependent. The block bootstrap over landmarks reduces, but does not eliminate, this dependence. The intervals are not animal-level confidence intervals and should not be read as such.

## 7. The primary innovation is not literally i.i.d.

The independent AR lead-1 pool has whitened ACF1 0.015, which is close to white, but raw ACF5 is 0.195 and ACF10 is 0.147. The Gaussian diagonal model ignores this remaining dependence. The label is "conditionally acceptable," not "validated."

## 8. Negative state-predictability R² is not proof of correct specification

Out-of-sample R² for the residual mean from state is negative (-0.38 adaptive, -0.44 independent). That means the fitted state map did not generalize. It does not prove that the conditional mean is correct. Distribution shift, finite-sample overfitting, or a genuine state-independent mean could all produce this pattern, and the experiment cannot distinguish them.

## 9. Conditional variance is not stable

The state-variance model has near-zero or negative held-out R², and the residual variance second-half/first-half ratio is about 0.88. A constant diagonal innovation is therefore an approximation. The variance drift is not modeled, and it may contribute to the horizon-dependent under-coverage.

## 10. Joint covariance estimates are noisy

Cross-neuron covariance estimated from about 100 residuals per session has substantial sampling error. The split-half correlation is only 0.20 for the adaptive residual and 0.60 for the independent residual. The covariance relative errors (0.49-0.86) are correspondingly uncertain. The low-rank decision (diagonal in 4/5 sessions) is a reasonable held-out choice, not a definitive statement that no population factor exists.

## 11. D4 is diagnostic and unstable

The validation-fitted global variance factors range from 1.84 to 4.00 across sessions. A single global factor cannot be transportable if the sessions differ this much. D4's test over-coverage and worse CRPS are evidence against using it as a repair, not a calibrated alternative.

## 12. Gate thresholds are conventions

The coverage tolerances (0.07/0.07/0.05), standardized-error range (0.7-1.4), width ratio (0.3-2.0), covariance error (<0.50), and autocorrelation error (<0.25) are prespecified conventions. They were frozen before the final evaluation, which protects against moving the goalposts, but they are not universal laws. A different threshold would change a pass/fail label, not the underlying numbers.

## 13. Multiple models, but not a model zoo

Seven models appear in the main comparison, plus D4, plus the historical comparator. They were prespecified as baselines, primary, secondary, and diagnostics; the primary was not selected after seeing test results. Still, the experiment is not a neutral exhaustive search, and the reported intervals are not adjusted for multiple comparisons across models and horizons.

## 14. No parameter uncertainty

Ensembles propagate innovation noise conditional on a fitted mean. They do not integrate over parameter uncertainty, model uncertainty, or regime uncertainty. A Bayesian or bootstrap-over-fits ensemble could be wider and better calibrated without changing the innovation variance. This is a plausible explanation for part of the remaining under-coverage and is a next experiment, not a result of this one.

## 15. No perturbation or biological validation

No neuron ranking, stimulation-target search, or perturbation screen was run. The generative gate failed, so perturbation would be model-dependent. Any such display must carry:

**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

Biological validation is `NOT_EVALUABLE`. No control claim, target claim, or causal claim is supported.

## 16. Synthetic controls are not biological validation

The controls show that the pipeline recovers known noise and detects temporal correlation, state heteroscedasticity, low-rank structure, measurement confounding, non-normal transient amplification, and mean misspecification. They do not show that the biological data follow any of those generators. A control passing is evidence about the method, not about the worm.

## 17. What remains unknown

- Whether a joint-state model with an explicit observation model would calibrate.
- Whether parameter uncertainty explains the remaining 95% under-coverage.
- Whether the five sessions share a stable operator or differ by regime.
- Whether a nonlinear mean would reduce the horizon-dependent under-coverage.
- Whether any of this generalizes to new animals or stimuli.

These are listed as next experiments, not as implied results.

## Related documents

- [Results](UNCAL_RESULTS.md)
- [Calibration](UNCAL_CALIBRATION.md)
- [Trajectory fidelity](UNCAL_TRAJECTORY_FIDELITY.md)
- [Next experiments](UNCAL_NEXT_EXPERIMENTS.md)
