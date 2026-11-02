# JOINTDYN next experiments

The present result is a conditional-mean forecasting signal with failed calibration. More model variants on the same 175 landmarks would add test reuse, not evidence. The next experiments are ordered by information value per risk.

## 1. External animal/stimulus holdout (highest priority)

**Question:** does the population conditional-mean advantage survive on animals and stimuli not used to choose the model?

Design:

- freeze the J2/J5 specification and the 10-D PCA rule;
- collect or obtain at least five animals not in Kato `WT_NoStim`;
- use a per-animal train/validation/test split, then evaluate across animals;
- never pool test frames into fitting or covariance estimation;
- primary endpoint: H20 energy and trajectory energy against independent AR;
- co-primary: 50/80/95 full-neuron and PCA coverage;
- analysis by animal, with a hierarchical summary, not pooled landmarks alone;
- require the direction in at least 4/5 new animals before calling it general.

This is the only design that directly addresses the current lack of independent animals.

## 2. Observation/state-space identification

**Question:** how much of the residual covariance is measurement noise rather than process noise?

Options:

- simultaneous calcium imaging and electrophysiology in the same preparation;
- two indicators or repeated acquisitions with different noise levels;
- a negative-control neuron or blank channel;
- an explicit state-space model with observation variance and process variance.

Deliverable: a posterior or likelihood-based decomposition of observation and process variance. Until this exists, no claim about "process innovation" should be treated as biological.

## 3. Stable low-dimensional covariance

**Question:** can a covariance model beat diagonal without over-covering?

Design:

- low-rank plus diagonal covariance with rank selected by animal-level cross-validation;
- factor analysis with shrinkage on the low-rank block only;
- covariance estimated on a neuron split and evaluated on held-out neurons;
- graphical lasso or hierarchical shrinkage as comparators;
- report covariance error, 50/80/95 coverage and energy together.

Do not refit a 103-dimensional full covariance on the same 264 residuals. The present full-covariance failure is a sample-size result, not a tuning problem.

## 4. Hierarchical parameter uncertainty

**Question:** does between-animal parameter variation matter more than within-animal bootstrap variation?

Design:

- hierarchical VAR with animal-level partial pooling;
- posterior or empirical-Bayes draws over animal means and variances;
- evaluate on held-out animals, not held-out frames from the same animal;
- compare against the fixed-parameter and diagonal models;
- require improved coverage without increased energy before adopting.

The current bootstrap test worsened dispersion; a hierarchical model is justified only if between-animal variation is demonstrably larger than the within-animal bootstrap spread.

## 5. Perturbation experiment with independent ground truth

**Question:** does a model-predicted response match an actually evoked response?

This is a **new biological experiment**, not a dashboard analysis.

Minimum design:

- prespecify the perturbation, dose and targeting;
- randomized sham and blinded scoring;
- independent animals for prediction and confirmation;
- a measured response that is not used to fit the model;
- safety and ethics review before any stimulation;
- report effect sizes with animal-level intervals, not only landmark-level intervals.

No neuron ranking or stimulation target should be issued from the current model. The required warning remains:

**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

Biological validation: `NOT_EVALUABLE`.

## 6. Evaluation and reporting standard

For every future experiment:

- split by animal, not by frame;
- freeze the PCA or use a train-only projection;
- report all horizons, not the best one;
- report energy, trajectory energy, CRPS, coverage, covariance error and autocorrelation together;
- include a persistence baseline and an independent-neuron baseline;
- state the number of animals, sessions and independent landmarks;
- retain failed calibration and negative results;
- version thresholds before scoring.

## 7. Do not do next

- Do not tune another scale, tail family or covariance on this test split.
- Do not add a temporal innovation model when ACF1 is 0.02.
- Do not call PCA energy improvement "generative validation".
- Do not pool multi-step forecast errors into a one-step innovation pool.
- Do not treat repeated sessions as independent animals.
- Do not issue a stimulation target from a model that fails G4, G5 and G7.

## 8. Decision rule for the next phase

Proceed to a perturbation pilot only after:

1. an external animal holdout confirms the conditional-mean advantage;
2. a stable covariance model passes calibration and covariance gates on held-out animals;
3. observation and process noise are separated or explicitly bounded;
4. an independent, preregistered perturbation design with safety review is approved.

Until then, the correct output is the present negative calibration result and the reproducible conditional-mean finding.
