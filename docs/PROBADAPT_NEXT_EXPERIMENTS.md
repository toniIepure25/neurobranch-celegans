# Next experiments

The current result is negative but diagnostic. Do not add model complexity until
the variance failure is isolated. The following sequence is ordered by
information gain per unit risk.

## E1. Separate mean error from variance error

**Question:** Is the overdispersion caused by the near-unit-root operator, the
innovation covariance, or the block resampling?

**Design:** Hold the mean operator exactly fixed at Rolling VAR w500. Compare
four innovation models fitted on training prequential residuals only:

1. diagonal covariance;
2. low-rank plus diagonal;
3. full Ledoit-Wolf covariance;
4. full covariance with a prespecified eigenvalue floor.

For each, report the implied H20 variance analytically and by Monte Carlo. Do
not tune a global inflation factor on test.

**Primary metric:** H20 energy score on the fixed 10-D projection.

**Calibration gate:** 50/80/95 coverage within `0.10/0.10/0.05`; total std ratio
between `0.8` and `1.25`; spread/error between `0.8` and `1.25`.

**Go/no-go:** Proceed only if at least one innovation model beats the local mean
and independent AR with a paired 95% interval excluding zero, **and** passes the
calibration gate.

## E2. State-conditioned innovation scale

**Question:** Does residual scale depend predictably on pre-landmark state?

**Design:** Fit a train-only model for log residual variance using features
available at the landmark: recent population norm, recent change norm, session
index (as a random effect, not a test lookup), and horizon. Compare:

- constant variance;
- state-conditioned scale with the same correlation shape;
- state-conditioned low-rank covariance.

Use nested chronological validation inside train. Freeze the scale model before
test.

**Metric:** H20 energy, coverage, PIT, variance ratio; per-session consistency.

**Risk:** If session index explains everything, the model is learning recording
identity, not dynamics. Report this explicitly and do not deploy it as a
biological simulator.

## E3. Whitened or sieve residual model

**Question:** Can the temporal dependence be modeled without block resampling
artifacts?

**Design:** Fit a low-order VAR to the prequential residuals (residual
whitening), or use a sieve/bootstrap of whitened residuals. Compare against the
moving-block bootstrap at matched innovation covariance.

**Metric:** energy, autocorrelation error, variance ratio, long-horizon energy.

**Go/no-go:** A residual model must reduce both energy and variance error, not
only one.

## E4. Leave-one-session-out and external validation

**Question:** Does any stochastic model generalize beyond the five repeated
sessions?

**Design:** Leave one session out entirely. Fit mean and innovation models on
the other four; evaluate on the held-out session with the same landmarks and
metrics. If an independent recording becomes available, repeat without
reusing its test tail.

**Metric:** session-level energy, coverage and variance ratio; report every
session, not only the mean.

**Decision:** A model that fails on two or more held-out sessions is not a
general simulator, even if the pooled mean looks good.

## E5. Full-space proper scoring

**Question:** Is the 10-D projection hiding a failure or creating one?

**Design:** Compute energy score and CRPS in the full predictor space at H20 for
the primary and best baseline, using a fixed neuron ordering. This is
computationally heavier; use 100 ensemble members and vectorized pairwise
distances.

**Metric:** full-space energy, CRPS, coverage, variance ratio.

**Purpose:** Confirm the ranking is not a projection artifact. Do not switch the
primary metric after seeing the result; report both.

## E6. Nonlinear or latent-state mean model — only if E1–E3 fail

**Question:** Is the missing cross-neuron information a limitation of linear
lag-one dynamics?

**Design:** A small, prespecified nonlinear model (e.g., low-rank RNN or
switching linear model) with the same split and proper scoring. Compare against
the adaptive local mean and independent AR, not only the frozen VAR.

**Go/no-go:** It must beat both simple baselines on energy and calibration. A
lower MSE alone is insufficient.

## E7. Perturbation readiness

Do not run a perturbation study until a stochastic model passes E1 and E4. When
it does:

- use common random numbers for baseline and intervention;
- preselect a small fixed neuron set;
- report distributional changes, OOD fraction and model disagreement;
- do not rank neurons as drug targets;
- retain the exact label:

**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

## Synthetic validation suite

Before any new biological claim, test the pipeline on:

1. a linear Gaussian system with known covariance;
2. a system with correlated multivariate innovations;
3. a system with temporally correlated residuals;
4. a non-Gaussian innovation system;
5. a persistence-optimal random walk;
6. a system where cross-neuron information is genuinely useful.

For each, check that the score ranking recovers the known best model and that
coverage is nominal. A pipeline that cannot pass these checks cannot support a
biological conclusion.

## What not to do

- Do not tune noise scale until coverage looks right on test.
- Do not select the best of many adaptive variants and report it as
  prespecified.
- Do not call a wide predictive band “biological variability.”
- Do not run a large perturbation screen on an uncalibrated model.
- Do not add a nonlinear model before testing whether a diagonal or
  state-conditioned covariance fixes the simpler failure.
