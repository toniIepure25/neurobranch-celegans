# JOINTDYN limitations

## 1. Design limitations

- **The test split has been reused across the project.** All test-set contrasts are exploratory. The corrected lead-1 model was examined after earlier test results were known, so this experiment cannot serve as an independent confirmation.
- **Five repeated sessions are not five animals.** Session-level agreement measures recording stability, not animal-to-animal generalisation.
- **The 175 landmarks overlap.** Windows share frames. Moving-block bootstrap intervals account for local dependence only approximately.
- **Only one preparation is available.** The repository contains `WT_NoStim`; no stimulation or external animal dataset was used. Cross-preparation generality is unknown.
- **No prospective preregistration.** The protocol was frozen before final scoring, but the broader analysis path had already seen this dataset.

## 2. Representation limitations

- Evaluation uses a fixed 10-component PCA fitted on training frames. It explains 54.7%-71.0% of training variance across sessions, so 29%-45% of variance is outside the joint representation.
- PCA components are ordered by variance, not by biological meaning. A population advantage in PC1-PC10 need not extend to all neurons.
- All-neuron and top-variance sensitivity analyses are reported, but the all-neuron energy interval includes zero.
- The predictors are standardized calcium-derived traces. Deconvolution, saturation, bleaching and indicator kinetics are not modelled explicitly.
- The analysis is linear in the mean. No switching, context input, saturation or nonlinear population dynamics is included.

## 3. Statistical limitations

- Intervals are moving-block bootstrap intervals over landmarks, not confidence intervals over animals.
- No multiplicity adjustment is applied across the many secondary scores. Secondary results are descriptive.
- The 100-member ensembles carry Monte Carlo error. It was not separated from sampling error in every table.
- Covariance split-half correlation is low (0.107 population, 0.153 independent), so the full covariance is weakly identified.
- Parameter draws are a frequentist moving-block bootstrap, not a Bayesian posterior. No posterior probability, credible interval or model evidence is claimed.
- The full-covariance model has about 264 one-step residuals for a 103-dimensional covariance. Shrinkage keeps it positive definite but does not make it stable.
- The residual bootstrap assumes approximate exchangeability of one-step residuals. Mild nonstationarity remains.

## 4. Identifiability limitations

- **Process and observation noise are not separately identified.** One calcium trace per neuron cannot distinguish latent dynamics from measurement noise.
- **Initial-state uncertainty is fixed, not integrated.** Evaluation is conditional on the observed starting state.
- **No latent state-space model was fitted.** The comparison is among observed-state predictors.
- **No neuron-specific mechanism is identified.** The population VAR is a statistical predictor, not a wiring diagram.

## 5. Calibration limitations

- Full-neuron and PCA coverage disagree. Neither is "the" calibration truth; they answer different questions.
- A global scale, Student-t tails or parameter bootstrap cannot repair a covariance shape error.
- D4 was tuned on validation only, but its base covariance was already overdispersed; validation and test agree that it is worse than D1.
- No test-set recalibration was performed, by design.
- PIT is close to uniform while interval coverage fails, showing that a single average calibration statistic is insufficient.

## 6. Trajectory limitations

- Trajectory energy is proper but not a full likelihood. A model can improve energy while retaining the wrong covariance.
- Visual trajectory fans are descriptive. Divergence is not biological branching.
- Occupancy and recurrence depend on the six-state k-means fit and are not mechanistic.
- Long-horizon finiteness is guaranteed partly by the spectral-radius cap; it is not evidence of biological stability.
- No intervention, stimulation or closed-loop validation was performed.

## 7. Claims that are not supported

- "The model is a validated generative simulator."
- "The model reproduces biological stochasticity."
- "Population covariance is recovered."
- "The model supports causal neuron targeting."
- "The result generalises to other animals, stimuli or preparations."
- "The marginal forecast is better than independent AR."
- "The result is a systematic review or meta-analysis."

## 8. Required safety statement

**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

Biological validation: `NOT_EVALUABLE`.

No neuron ranking, stimulation protocol or perturbation target was produced.

## 9. Artifact limitations

- Raw recordings and large model weights are not committed.
- Only derived scores, diagnostics, compressed projected arrays and figures are versioned.
- Reproduction is CPU- and version-sensitive at the level of linear algebra; the exact environment is recorded in the manifest.
- The interactive HTML figure is a derived visualization, not raw evidence.
