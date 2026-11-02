# Limitations

**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

## Data and design

- Only Kato et al. (2015) `WT_NoStim` is used. There are five repeated recording
  sessions, not five documented independent animals. Session-level intervals are
  not animal-level intervals.
- The test split has been inspected in earlier project phases and is reused here.
  Every result is exploratory; this is not a confirmatory or systematic review.
- No intervention, sham, dose, neuron-specific stimulation, or behavioural
  ground truth is available. Biological validation is `NOT_EVALUABLE`.
- Calcium imaging is a slow, indirect observation of neural activity. The 103
  predictors are a selected panel, not the full connectome.
- Kato macro labels are decoded from a subset of the same neurons. They are not
  independent behaviour and are not used as state ground truth.

## Model and statistics

- The dynamical object is a lag-one linear VAR with ridge regularization and a
  spectral cap. It cannot represent nonlinear gating, adaptation, or
  unobserved common input.
- B3's off-diagonal coefficients are not stable across chronological training
  segments (median pairwise correlation `0.052`). A single fitted operator is
  not a reliable interaction graph.
- B3 beats B1 on H20 energy, but the mean-forecast MSE contrast is weaker and its
  interval includes zero. The energy advantage therefore cannot be described as
  a pure conditional-mean improvement.
- Uncertainty intervals are moving-block intervals within sessions, pooled with
  equal session weight. They do not include model-selection uncertainty, animal
  variation, or the fact that landmarks overlap.
- J2 and J5 are mean-equivalent but not per-landmark identical; individual
  landmark differences can be large because innovation realizations differ.
- The surrogate null is deliberately destructive: it preserves marginals and
  lag-1 autocorrelation but can damage other temporal structure. A significant
  B3-versus-surrogate contrast is evidence against a simple alignment null, not a
  decomposition of direct versus indirect effects.
- B4 (diagonalized fitted operator) and B4r (independent AR refit) answer
  different questions. B4 keeps the full-fit diagonal and is a poor mean model;
  B4r is a matched own-neuron-only comparator. Neither is a biological lesion.
- The KMeans state partition has six arbitrary states and is fitted on training
  PCA geometry only. Transition TV is sensitive to the partition and to dwell
  discretization.
- The sensitivity amplitude (`+1 SD`) is a mathematical displacement, not a
  calibrated biological dose. OOD support is measured against the training
  1st-99th percentile envelope, which is a coarse support proxy.

## Claim boundary

- No result identifies a synapse, a direct interaction, a command neuron, or a
  causal driver.
- No neuron is ranked as a stimulation target.
- No perturbation output is a prediction of an experimental outcome.
- A technical gate PASS is not biological validation.
- The joint calibration and joint trajectory gates remain `FAIL`; the simulator
  is not a calibrated generative model.
- The correct label for the whole audit is **exploratory repeated-session
  reanalysis**, not a systematic review and not a validated digital twin.

## What would change the conclusion

Independent animals, a pre-registered split, matched neuron-specific
perturbation with sham and dose, an independently measured common input, and a
held-out-animal/held-out-target falsification test. Until then, the strongest
defensible claim is predictive and model-dependent.
