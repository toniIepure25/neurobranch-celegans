# Next experiments

These are ordered by information value, not by model complexity. Each has a falsifiable gate and a stop condition. The purpose is to avoid another architecture sweep on the same five sessions.

## 1. Joint state-space model with an explicit observation model

**Question.** Does a model that carries a latent joint state and a separate calcium observation model calibrate both marginals and covariance?

**Design.**

- latent state dimension selected on validation, not test;
- linear Gaussian state equation with a joint innovation covariance;
- explicit observation equation for calcium dynamics, including a measurement-noise term;
- parameter uncertainty via bootstrap over fits or a Bayesian posterior;
- same 175 landmarks and frozen PCA for comparability, plus a new-animal holdout.

**Primary gate.** H20 50/80/95 coverage within 0.07/0.07/0.05, standardized-error variance in [0.7, 1.4], covariance relative error < 0.50, trajectory energy beats persistence with 95% lower bound > 0.

**Stop.** If the joint model still fails covariance, stop adding mean complexity and diagnose the observation model or the stationarity assumption.

## 2. Separate process and measurement noise

**Question.** How much of the observed innovation variance is calcium measurement noise rather than latent process noise?

**Design.** This dataset cannot identify the split from one trace per neuron. The next experiment needs one of:

- repeated measurements of the same neuron/state;
- simultaneous electrophysiology and calcium imaging;
- a known noise injection or calibration protocol;
- a hierarchical observation model with a validated noise model.

**Gate.** The process-variance estimate must be stable across sessions and must not be a relabeled residual variance. If the design cannot identify it, report `NOT_IDENTIFIABLE` rather than fitting a split.

## 3. Horizon-dependent mean and regime model

**Question.** Is the growing under-coverage caused by mean drift or regime change rather than by innovation scale?

**Design.**

- fit a slowly varying mean or a small number of regimes on training/validation only;
- evaluate H1, H5, H10, H20, H50, H100 with the same proper scores;
- require no degradation at H1 and improved coverage at H20/H50/H100;
- compare against a parameter-uncertainty ensemble with the same mean.

**Gate.** H20 coverage improves without widening 95% width beyond 2x persistence, and H1 CRPS does not get worse. If the mean model only widens intervals, reject it.

## 4. Stable joint covariance, only if justified

**Question.** Is there a stable cross-neuron covariance that a low-rank-plus-diagonal model can exploit?

**Design.**

- estimate covariance on independent halves and across sessions;
- require split-half correlation and cross-session stability above a prespecified threshold;
- select rank on held-out Gaussian likelihood, not test score;
- compare against a shrinkage estimator and a diagonal model.

**Gate.** Low-rank must beat diagonal on held-out likelihood in at least four of five sessions and must improve covariance relative error without breaking marginal calibration. The current evidence does not meet this bar.

## 5. Prospective new-animal validation

**Question.** Do the calibration conclusions generalize beyond the five repeated sessions?

**Design.**

- freeze the model and thresholds before seeing the new animal;
- evaluate per animal, not pooled;
- report calibration, sharpness, joint score, and failure cases;
- no test-time model selection.

**Gate.** The same coverage and covariance gates on the new animal. A pooled pass with a per-animal failure is a fail.

## 6. Perturbation, only after generative credibility

**Question.** If a model passes Gate G, can a model-dependent virtual perturbation generate a testable prediction?

**Design.**

- no neuron ranking or stimulation-target search before Gate G;
- prespecify the perturbation and the predicted direction;
- report model dependence and uncertainty;
- require an experimental collaborator and a causal design before any biological claim.

**Gate.** Gate G must pass first. Until then, the only permissible label is:

**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

Biological validation remains `NOT_EVALUABLE`.

## 7. Calibration stress tests

**Question.** Is the calibration result robust to the evaluation protocol?

**Design.**

- session-held-out and animal-held-out splits;
- non-overlapping landmarks at each horizon;
- block bootstrap by session, not only by landmark;
- coverage conditional on activity state and time;
- PIT by neuron and by session, not only pooled.

**Gate.** The conclusion must survive at least two independent splitting schemes. A result that only appears under overlapping landmarks is not robust.

## 8. Extend the synthetic controls

**Question.** Can the pipeline distinguish the remaining failure modes?

**Design.**

- parameter-uncertainty-only ensembles;
- slow mean drift with correct innovation;
- regime switching;
- heavy-tailed but calibrated innovations;
- observation noise with repeated measurements;
- a known joint factor plus diagonal noise.

**Gate.** The pipeline must recover the known generator and must not label a misspecified model as calibrated. Add controls before adding biological models.

## What not to do

- Do not run another broad architecture search on these five sessions.
- Do not promote D4 global scaling as a calibration fix.
- Do not add low-rank covariance without stable held-out evidence.
- Do not claim process-noise separation from a single calcium trace.
- Do not present a perturbation as causal without an experiment.

## Priority

1. Joint state-space with observation model and parameter uncertainty.
2. Process/measurement separation with a design that can identify it.
3. Prospective new-animal validation.
4. Horizon-dependent mean/regime diagnosis.
5. Stable covariance, only if justified.
6. Perturbation, only after generative credibility.

## Related documents

- [Results](UNCAL_RESULTS.md)
- [Limitations](UNCAL_LIMITATIONS.md)
- [Supervisor brief](UNCAL_SUPERVISOR_BRIEF.md)
- [Protocol](UNCAL_RESEARCH_PROTOCOL.md)
