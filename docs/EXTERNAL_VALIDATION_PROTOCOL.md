# External validation protocol

**Status:** pre-registered for the current branch; no external test result was
used to choose a model or threshold.

## Research question

Can the NeuroBranch free-running neural dynamics model predict an animal's
future population activity from observed history without teacher forcing, and
does it do so better than a constant-state persistence baseline on animals that
were not used to select the model?

Supporting questions are deliberately separate:

1. Does a **frozen** fitted model transfer to another animal unchanged?
2. Does a **per-animal refit** on an early segment predict that animal's
   untouched later segment?
3. Does a **shared model family** generalize across animals under an explicit
   neuron mapping?
4. How does the model respond to **measured environmental input**? This is not
   the same as a neuronal perturbation and is not tested here.

## Evidence tiers

| Tier | Meaning |
| --- | --- |
| confirmatory | untouched cohort, pre-registered split and model, no test-driven decisions |
| exploratory | test block previously inspected or external pilot with unproven independence |
| descriptive | no inferential claim; used for provenance or feasibility |

Phase 6 test block 2 was already inspected in Phase 5. Every Phase 6 test/gate
number is therefore **exploratory**, including the 99th-percentile support pass.
The Kato Stim pilot is exploratory because animal independence is not documented
and the cohort is small.

## Definitions

### A — frozen transfer

Apply all of the following without refitting or rescaling:

- the same ordered predictor set and feature definition;
- the same training mean and standard deviation;
- the same fitted coefficients, behaviour probe and support model;
- the same history length and observation/noise model;
- the same evaluation horizon and start-selection rule.

If any item differs, A is **blocked**, not “approximately transferred”. A
dimension-matched subset chosen after seeing the target animal is a new model,
not frozen transfer.

### B — per-animal refit

For each animal independently:

1. Use a chronological training segment.
2. Use the next contiguous segment for validation/support calibration.
3. Remove a purge gap of at least the history and forecast window.
4. Keep the final segment untouched until one evaluation.
5. Fit scaler, model, probe and support model on training/validation only.
6. Use fixed hyperparameters from Phase 6; do not select on the target test
   segment.
7. Report every animal, including failures.

B does not test cross-animal generalization. It tests whether the pipeline can
predict a new animal's future after seeing only that animal's past.

### C — cross-animal model-family generalization

C requires, before fitting:

- a documented one-to-one neuron/ROI mapping across animals;
- a stated annotation version and a rule for missing or multi-neuron traces;
- no use of worm-specific numeric IDs as if they were shared identities;
- leave-one-animal-out training and untouched held-out animals;
- no pooling of unequal panels by truncation or padding without an explicit
  observation model.

Without these, C is **blocked**. A common name is not a common panel.

## Input semantics

Oxygen concentration is a measured environmental variable. In Kato Stim it
alternates between 4% and 21%; in Nichols it steps between 10% and 21%. These
are not targeted neuronal stimulations and cannot establish a causal role for a
named neuron. The current simulator has no exogenous-input channel, so an
oxygen-response test would require a new model and a separate protocol.

The only external test executed here holds oxygen constant at 21% for the first
360 s of Kato Stim, before any switch. It is an autonomous-dynamics test under a
constant environmental condition.

## Data eligibility

An external dataset is eligible for B only if:

- it contains continuous population calcium traces with a time vector;
- the animal/session can be split chronologically with a purge gap;
- at least 100 test frames remain after the split;
- traces are finite, or a pre-specified finite subset is defined before fitting;
- the preparation and signal modality are compatible enough that a negative
  result is interpretable;
- the dataset was not used to choose the model, horizon, support threshold or
  baseline.

Kato Stim meets these conditions for the constant-oxygen segment. Nichols is
technically readable but is not eligible for a like-for-like B test without a
separate lethargus/arousal protocol.

## Kato Stim B pilot (pre-registered)

- Cohort: all seven `WT_Stim` sessions.
- Segment: `0 <= t < 360 s`, constant 21% oxygen; the 360 s switch is excluded.
- Traces: all non-behaviour traces, in the animal's own order. A trace carrying
  any AVA/SMD/RIB behaviour-label neuron is excluded.
- Split: first 2/3 train, next 1/6 validation, 12-frame purge, remainder test.
- Model: VAR, lag 2, ridge 10.0, nominal spectral cap 0.995, support quantile
  0.99. These are frozen Phase 6 settings.
- Fit: scaler, VAR and behaviour probe on train only; support on validation only.
- Evaluation: 30 deterministic test starts, free-running horizon 20, no teacher
  forcing, no noise.
- Baseline: constant initial state repeated over the horizon (the same
  persistence baseline used in Phase 6).
- Primary statistic: `skill = 1 - MSE_model / MSE_persistence` at H=20.
- Secondary: one-step R², energy distance, autocorrelation error, covariance
  error, OOD fraction, finite fraction, behaviour composition.
- Uncertainty: animal-level median/mean and a cluster bootstrap over animals.
  The bootstrap is descriptive; seven sessions cannot rescue a failed effect.
- Decision rule: no gate. A positive pilot would motivate a larger independent
  study; a negative pilot is retained as a negative result.

## Metrics and baselines

- **MSE / skill:** lower MSE is better; skill 0 equals persistence, negative is
  worse.
- **Persistence:** constant initial state, not one-step persistence.
- **Distribution:** energy distance, autocorrelation error, covariance error and
  variance ratio describe distributional mismatch, not mechanism.
- **Support/OOD:** kNN support calibrated on validation; in-support is not
  accuracy and can reward shrunken rollouts.
- **Uncertainty:** report animal-level spread and bootstrap intervals; never
  report only the best animal.
- **Frozen Phase 5 VAR:** not applicable to external dimensions; the Phase 6
  comparison is descriptive and unpaired.

## Split and leakage rules

- No test frame enters scaling, fitting, early stopping, support calibration or
  figure selection.
- Chronological order is mandatory for B; blocked splits are allowed only when
  the block structure is pre-specified.
- Purge at least the history plus forecast horizon. The pilot uses 12 frames.
- No random start sampling that crosses split boundaries.
- No test-based model family, horizon, threshold or target selection.
- Negative and failed runs are reported; no silent dropping.

## External-input and intervention claims

- Oxygen is environmental input, not neuronal stimulation.
- A sensory input experiment can test whether an environmental variable changes
  population dynamics; it cannot establish that a specific neuron causes a
  behavioural effect.
- A model-dependent perturbation is a virtual sensitivity analysis, not a
  biological intervention.
- Any output from the perturbation interface must retain the warning
  `MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.`

## Decision matrix

| Dataset | A | B | C | Input-response |
| --- | --- | --- | --- | --- |
| Kato Stim | blocked | go, limited pilot | blocked | limited, not executed |
| Nichols 2017 | blocked | limited, not executed | blocked | blocked/limited |
| Kato NoStim | n/a | no untouched test | blocked | n/a |

## Stop rules

Stop or downgrade the claim if:

- dimensions or feature order cannot be matched for A;
- test data influenced any fit, threshold or model choice;
- fewer than five animals/sessions remain;
- non-finite rollouts exceed 5%;
- support is below 90% for the evaluated condition;
- the result depends on one animal or one seed;
- animal independence cannot be stated honestly.

## Reproduction

- Adapter: `src/neurobranch/data/external.py`.
- Inventory: `scripts/external_validation_inventory.py`.
- Pilot: `scripts/external_validation_kato_stim.py`.
- Outputs: `reports/external_validation/kato_stim_pilot_*.{json,csv}`.
- Phase 6 freeze/audit: `scripts/phase6_threshold_audit.py`.
