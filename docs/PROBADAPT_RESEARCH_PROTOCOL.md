# Probabilistic adaptive dynamics: research protocol

## Status and scope

Frozen protocol for the stage following the deterministic adaptive-VAR result.
This is a focused extension, not a model search. All analyses use the existing
Kato WT_NoStim sessions and the existing chronological split. No raw recordings
or model weights are committed.

- Evidence tier: **exploratory repeated-session reanalysis**. The test tails
  have been inspected in earlier stages. Sessions are not confirmed independent
  animals; landmark uncertainty is not animal-level inference.
- Starting revision: `a33c1fd9dfcba63ab6cbd79ac9c897d2fa6b27ce`.
- Branch: `research/probabilistic-adaptive-dynamics`.

## Central question

Can the improved adaptive conditional-mean forecast be extended into a
calibrated probabilistic simulator that preserves forecast accuracy while
reproducing the conditional distribution of future neural trajectories?

The conditional mean, a probabilistic forecast, a generative trajectory and a
biological intervention are four different claims. Passing one does not imply
another.

## Hypotheses

- **H1 — local level tracking.** The adaptive VAR advantage is mostly local
  mean tracking, not multineuron dynamics.
- **H2 — missing innovation uncertainty.** The deterministic adaptive model
  estimates the conditional mean but omits process/observation uncertainty.
- **H3 — correlated innovations.** Residual uncertainty has cross-neuron and
  temporal structure that independent Gaussian noise cannot reproduce.
- **H4 — residual nonstationarity.** Innovation scale or structure depends on
  recent neural state.
- **H5 — structural limitation.** Even a correctly calibrated stochastic
  extension may fail at longer horizons or state transitions.

These are tested separately. Adding noise is not assumed to improve the model.

## Data and split

- Kato WT_NoStim, five repeated sessions.
- Existing chronological train/validation/test split. `WormData.X` is the
  existing train-standardized predictor matrix.
- Test evaluation landmarks: `split_starts(test, history=2, horizon=100)[::10]`,
  35 per session (34 in session 2), 175 total.
- H20 forecasts at stride 10 overlap 10 frames. Landmarks are dependent.
- No test frame is used for baseline selection or innovation-model choice.
- Baseline settings are selected on the last 20% of the training split only.
- The prior primary model (`rolling_w500`) remains prespecified; it is not
  reselected on test.

## Stage A — freeze and audit

1. Recompute frozen VAR H20 MSE and rolling w500 H20 MSE at exactly the
   historical landmarks.
2. Recompute the paired difference and block-bootstrap interval.
3. Report three legitimate relative summaries separately:
   - aggregate ratio: `1 - mean(adaptive) / mean(frozen)`;
   - equal-weight per-landmark mean: `mean[(frozen-adaptive)/frozen]`;
   - difference divided by mean frozen.
4. Report block-length sensitivity for 1, 2, 5, 10, 20, 30 and 50.
5. Report session-level estimates and overlap. Do not imply animal-level
   inference.

Gate A passes if the recomputed landmark MSEs agree with the historical
artifact to numerical precision and all relative definitions are explicit.

## Stage B — local-mean and independent-AR ablation

Fixed comparison set:

- B0 persistence;
- B1 frozen VAR;
- B2 rolling local mean;
- B3 exponentially weighted local mean;
- B4 independent per-neuron adaptive AR(1);
- B5 prespecified full Rolling VAR w500.

Candidate windows: local mean 100/200/500/1000; EWMA half-life 100/300/1000;
independent AR window 200/500/1000. Select each on the last 20% of train by H20
MSE, then freeze for test. Same landmarks, normalization, horizons and metric
code for all models.

Gate B criteria, fixed before test:

- **PASS** if B5 improves over both B2 and B4 with paired 95% block-bootstrap
  lower bound > 0.
- **MIXED** if it improves over one but not the other.
- **FAIL** if it improves over neither.

A failure means the adaptive advantage cannot be attributed to useful
cross-neuron dynamics.

## Stage C — residual diagnostics

Use block-prequential residuals: fit the rolling model at a block start, then
predict one-step transitions forward inside the block. This makes residuals
out-of-sample with respect to the fitted operator while remaining causal.
Retrospective diagnostics are labelled; test residuals are never used to choose
the innovation model.

Diagnostics:

1. mean and per-neuron variance;
2. temporal autocorrelation at lags 1, 2, 5, 10;
3. cross-neuron covariance and correlation;
4. covariance rank, effective rank and condition number;
5. kurtosis and skewness;
6. dependence of residual norm on recent state norm;
7. dependence on within-block lead;
8. covariance drift across thirds;
9. out-of-block predictability from recent state;
10. covariance estimator sensitivity (Ledoit-Wolf, diagonal, fixed shrinkage).

Covariance is not fitted unconstrained when effective sample size is weak.
Ledoit-Wolf shrinkage plus an eigenvalue floor is the default. A diagonal
alternative is retained for sensitivity.

Innovation-model choice is fixed by a training-only rule:

- moving-block bootstrap if median absolute lag-1 residual autocorrelation
  exceeds 0.1;
- iid empirical bootstrap otherwise.

## Stage D — stochastic adaptive VAR

Conditional mean: prespecified Rolling VAR w500, ridge 1.0, spectral cap 0.995.

At each landmark:

- fit using observations at or before the landmark;
- freeze the operator and innovation model for the full horizon;
- generate independent trajectories;
- never assimilate future truth.

Candidate innovation models:

- D1 Gaussian with Ledoit-Wolf covariance;
- D2 iid empirical residual bootstrap;
- D3 moving-block residual bootstrap (block selected by the Stage C rule).

Innovations are centered. The deterministic and stochastic semantics are
reported separately. Monte Carlo recovery of the conditional mean is tested on
synthetic linear systems.

## Stage E — conditional distribution evaluation

Primary metric: **multivariate energy score** at H20 on a per-session,
training-fitted 10-component PCA projection. Lower is better. The projection
dimension is fixed before test.

Secondary metrics:

- per-neuron CRPS at H20;
- 50/80/95% central interval coverage and width;
- ensemble spread versus realized squared error;
- PIT histogram;
- variogram score over projection pairs;
- temporal covariance and lag-one autocorrelation error;
- OOD fraction outside the train 1–99% range;
- numerical stability.

Variance decomposition explicitly separates:

1. observed unconditional variance;
2. across-landmark variance of conditional means;
3. mean within-ensemble predictive variance;
4. total model-implied variance (law of total variance);
5. forecast-error variance.

Coverage is not allowed to pass by making intervals arbitrarily wide; width is
reported alongside coverage.

## Stage F — strong probabilistic baselines

All models use the same landmarks, initial states, ensemble size (100),
projection and score code:

- stochastic persistence with moving-block difference innovations;
- stochastic local mean with moving-block residual innovations;
- independent adaptive AR with diagonal Gaussian innovations;
- frozen VAR with moving-block innovations;
- adaptive VAR with Gaussian, iid bootstrap, moving-block bootstrap and
  diagonal Gaussian innovations.

Gate D: primary stochastic model must improve energy score over **both** the
local-mean and independent-AR probabilistic baselines with paired 95%
block-bootstrap lower bound > 0.

Gate E: calibration passes only if 50/80/95 coverage is within 0.10/0.10/0.05
of nominal, PIT calibration error is below 0.05, and mean spread/error is
between 0.5 and 2.0.

Gate F: total implied standard-deviation ratio between 0.5 and 2.0, temporal
covariance relative error below 1.0, and no non-finite trajectory.

A failure is retained. Deterministic MSE improvement is not sufficient for
Gates D–F.

## Stage G — trajectories and branching

Starting from identical observed histories, show the deterministic mean,
individual stochastic trajectories, predictive bands, and the real continuation.
Report ensemble diversity, occupancy, transition statistics, temporal
covariance and long-horizon stability. Do not call divergent paths biological
bifurcations. A compelling plot is not evidence of a generative model.

## Stage H — perturbation safety

No large screen. If Gates D–F fail, perturbation analysis is descriptive only or
not launched. If a small bounded perturbation is reported, it uses common random
numbers, a fixed small neuron set, predictive-distribution changes, OOD
fractions and model disagreement. No biological target ranking. Every result
carries:

**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

## Decision gates

| Gate | Meaning |
|---|---|
| A | Reproduction and metric reconciliation |
| B | Adaptive population information beyond local mean/independent AR |
| C | Innovation model validity and covariance support |
| D | Proper probabilistic score advantage |
| E | Calibration |
| F | Trajectory fidelity |
| G | Mathematical perturbation sensitivity: model-dependent only |
| H | Independent biological validation: `NOT_EVALUABLE` |

Gate criteria are not changed after seeing results.
