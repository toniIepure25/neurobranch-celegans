# Adaptive dynamics and stability: frozen protocol

## Question

Does a computationally simple dynamical model that adapts to recently observed
neural activity improve prospective, autonomous multistep simulation compared
with a frozen dynamical model?

This is an exploratory methodological experiment on Kato *C. elegans* whole-brain
recordings that have already been inspected repeatedly. It is not untouched
external confirmation and it is not a test of biological unpredictability.

## Data and split

- Kato WT_NoStim, five sessions, approximately 1080 s each, 2.80–3.07 fps.
- Existing train/validation/test split from `forward_stress.yaml`; test is a
  chronological tail.
- `data.X` is the existing train-standardized predictor matrix.
- No test frame is used to fit, select, or stabilize the frozen model.
- Adaptive models may use frames at or before each prediction landmark. Once a
  forecast starts, no ground-truth observation from the horizon is used.
- The five sessions are repeated recordings, not independent animals.

## Stage A: nonstationarity audit

- Sliding PCA windows of 256 frames, step 64, 10 components, on all frames.
- Reference subspace and covariance fitted on the training split only.
- Metrics: principal angles to the train reference, adjacent-window angles,
  participation ratio, covariance change, log amplitude shift, median occupancy
  shift, local AR(1) drift, local spectral radius.
- Null: multivariate phase randomization preserving each channel's power
  spectrum and the cross-spectrum. This destroys temporal nonstationarity while
  retaining stationary covariance and autocorrelation.
- Synthetic controls: a stationary VAR(1) latent system and a system whose
  latent operator and observation loadings switch at the midpoint. Both use
  matched windows and train-only reference segments.
- Principal angles alone are not interpreted as biological transitions.

## Stage B: calcium stability

The previously frozen `CausalCalciumLDS` is diagnosed without modifying its
checkpoint. Diagnostics: transition spectral radius, observation conditioning,
high-pass gain, latent initialization, one-step versus free rollout, numerical
growth, and finite trajectories.

A new versioned model, `causal_calcium_lds_corrected_v1`, keeps the latent drive
recursion and the calcium observation recursion separate. It does not re-encode
predicted calcium as a new drive. Its operator is capped at spectral radius
0.995. No trajectory clipping is applied. The tradeoff between stability and
predictive fidelity is evaluated on the same landmarks and a known synthetic
calcium system.

## Stage C: candidate set

1. **C1 Frozen VAR** — lag-one weighted ridge VAR fitted on the training split,
   spectral radius capped at 0.995, then frozen.
2. **C2 Rolling VAR** — equal-weight ridge refitted at every landmark on the
   last 200, 500, or 1000 transitions with target at or before the landmark.
3. **C3 EWMA VAR** — exponentially weighted ridge using all transitions up to
   the landmark, with half-lives 100, 300, or 1000 frames.
4. **C4 Local latent dynamics** — not fitted. Stage A provides no stable
   train-only latent orientation, so introducing a local encoder would mix
   incompatible latent coordinate systems. The comparison is resolved with
   C1–C3 first, as required.

Ridge penalty is fixed at 1.0 on standardized traces. Spectral cap is 0.995.
No window or half-life is selected on test performance; all candidates are
reported and `rolling_w500` is the prespecified primary adaptive candidate.

## Stage D: prospective evaluation

- Landmarks are every 10 frames whose full 100-frame target lies in the test
  split. All models use exactly the same landmarks and target frames.
- At each landmark the model is fitted/updated using only frames at or before
  that landmark, the full 100-step forecast is generated autonomously, and the
  next landmark is processed only afterward.
- Horizons: 1, 5, 10, 20, 50, 100 frames, also reported in seconds using each
  session's fps.
- Primary metric: paired H20 free-running MSE improvement of adaptive versus
  frozen VAR, computed on identical landmarks.
- Secondary: skill versus persistence, per-neuron H20 skill, error growth,
  finite/long-horizon stability, covariance error, autocorrelation error,
  amplitude ratio, state-occupancy/OOD rate, spectral radius, update cost.
- Uncertainty: moving-block bootstrap over landmarks (block 5, 1000 draws),
  plus session-level reporting. No animal-level claim is made.

## Stage E: heterogeneity

At each landmark, only past data are used for: recent covariance shift, recent
subspace angle to train PCA, amplitude shift, occupancy shift, and recent frozen
one-step error. These indicators are related to paired H20 improvement with
session fixed effects and a within-session block bootstrap. Quartile strata are
reported without selecting a best case. Oxygen transitions are not available in
WT_NoStim and are not imputed.

## Stage F: perturbation rule

No new perturbation screen is run. Descriptive impulse comparisons between
frozen and primary adaptive dynamics are allowed only as sensitivity analysis,
with a prespecified alphabetical neuron subset and bounded one-SD impulses.
Every result carries:

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

No neuron is ranked as a biological intervention candidate.

## Decision gates

| Gate | Criterion |
|---|---|
| A Nonstationarity characterized | Matched window/surrogate analysis separates estimation variability from observed drift; synthetic controls behave as expected. |
| B Numerical stability | Corrected calcium model produces finite trajectories; original defect is localized. |
| C Adaptive advantage | Prespecified primary adaptive model improves paired H20 MSE over frozen VAR with a bootstrap CI excluding zero. |
| D Distributional improvement | Adaptive model does not materially worsen temporal/covariance fidelity relative to frozen and persistence. |
| E Perturbation interpretation | Model-dependent sensitivity only unless compatible experimental intervention evidence exists. |
| F Independent biological validation | `NOT_EVALUABLE` without an independent dataset and intervention experiment. |

Thresholds are frozen before seeing the test metrics. A gate may fail. No
threshold is changed to force a positive conclusion.

## Deviations recorded after execution

- The synthetic regime-switch control initially changed only the latent
  operator, leaving the observation subspace almost unchanged; it was corrected
  before final reporting to switch observation loadings as well.
- C4 was not fitted, as documented above.
- No oxygen-transition analysis was performed in this stage because WT_NoStim
  has no oxygen schedule and WT_Stim is a separate repeated-session dataset.
