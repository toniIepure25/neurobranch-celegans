# Probabilistic evaluation

## Evaluation contract

- 175 chronological test landmarks, 5 repeated sessions.
- H20 is the primary horizon; horizons 5, 50 and 100 are secondary.
- 100 ensemble members per landmark and model.
- Primary score: **multivariate energy score** in a 10-component PCA projection
  fitted on the training split only. The projection is fixed before test.
- Secondary: per-neuron CRPS, central-interval coverage and width, PIT,
  variogram score, temporal covariance/autocorrelation error, OOD fraction and
  numerical stability.
- Lower is better for energy, CRPS, variogram and calibration error.
- Improvement is `reference score − model score`; positive favors the model.
- Intervals use a moving-block bootstrap over landmarks. They are not
  animal-level intervals.

The energy score is

\[
S(P,y) = \mathbb{E}_{X\sim P}\|X-y\| - \tfrac12
\mathbb{E}_{X,X'\sim P}\|X-X'\|.
\]

The factor `1/2` is required for propriety. The implementation was checked
against known Gaussian ensembles: a calibrated ensemble beats an overdispersed
one at the true mean.

## Main result

Mean H20 scores:

| model | energy ↓ | CRPS ↓ | coverage 50 | coverage 80 | coverage 95 | width 95 | spread/error | PIT error ↓ | OOD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 persistence | 2.948 | 0.452 | 0.496 | 0.793 | 0.937 | 3.28 | 1.17 | 0.007 | 0.168 |
| P1 local mean | 3.263 | 0.439 | 0.467 | 0.766 | 0.914 | 2.85 | 0.91 | 0.008 | 0.116 |
| P2 independent AR | 3.077 | **0.420** | 0.497 | 0.774 | 0.911 | 2.66 | 1.10 | 0.013 | 0.109 |
| P3 frozen VAR | 3.681 | 0.589 | 0.363 | 0.627 | 0.815 | 2.75 | 0.56 | 0.038 | 0.064 |
| P4 adaptive Gaussian | 3.038 | 0.469 | 0.790 | 0.964 | 0.993 | 5.02 | 3.80 | 0.060 | 0.241 |
| P5 adaptive iid | 2.962 | 0.455 | 0.750 | 0.947 | 0.989 | 4.71 | 3.34 | 0.052 | 0.221 |
| **P6 adaptive block (primary)** | **4.605** | **0.648** | **0.904** | **0.992** | **0.999** | **9.97** | **14.85** | **0.085** | **0.380** |
| P7 adaptive diagonal | 2.838 | 0.450 | 0.744 | 0.941 | 0.986 | 4.50 | 3.12 | 0.052 | 0.214 |

The primary model is the worst model on every proper score and on variogram
score. Its nominal 95% interval covers 99.9% of held-out outcomes and is about
3.5 times wider than the local-mean interval. This is not calibration.

## Paired comparison against the primary model

Positive means the reference is worse than P6; all values are negative, so P6 is
worse than every baseline.

| reference | energy improvement | 95% CI | landmarks where P6 better |
|---|---:|---:|---:|
| P0 persistence | −1.656 | [−2.031, −1.359] | 8.6% |
| P1 local mean | −1.341 | [−1.767, −0.956] | 18.3% |
| P2 independent AR | −1.528 | [−1.887, −1.211] | 14.3% |
| P3 frozen VAR | −0.924 | [−1.361, −0.487] | 31.4% |

**Gate D: FAIL.** The prespecified stochastic adaptive model does not beat the
simple probabilistic baselines.

## Calibration and variance

| model | within variance | total implied variance | observed variance | total std ratio |
|---|---:|---:|---:|---:|
| P1 local mean | 0.671 | 0.909 | 0.861 | 1.11 |
| P2 independent AR | 0.753 | 1.151 | 0.861 | 1.11 |
| P0 persistence | 0.900 | 1.691 | 0.861 | 1.58 |
| P7 adaptive diagonal | 2.099 | 2.673 | 0.861 | 1.87 |
| P5 adaptive iid | 2.237 | 2.811 | 0.861 | 1.90 |
| P4 adaptive Gaussian | 2.515 | 3.105 | 0.861 | 2.01 |
| **P6 adaptive block** | **11.461** | **12.122** | **0.861** | **3.40** |

The law of total variance is reported explicitly:

\[
\operatorname{Var}(X_{t+h}) =
\operatorname{Var}(\mathbb{E}[X_{t+h}\mid X_t]) +
\mathbb{E}[\operatorname{Var}(X_{t+h}\mid X_t)].
\]

P6's mean is accurate (`error_var 1.042`), but its within-ensemble variance is
13 times the observed variance. A model can have low conditional-mean error and
still be a bad distribution.

**Gate E: FAIL.** 50/80/95 coverage is 0.904/0.992/0.999, PIT error is 0.085,
and spread/error is 14.85.

**Gate F: FAIL.** Total standard-deviation ratio is 3.40 and OOD fraction is
0.380. The 10-D temporal covariance error is near 1 for all models because the
truth covariance is close to singular in this projection; it is reported but is
not used as the sole fidelity criterion.

## Horizon behavior

Mean energy:

| model | H5 | H20 | H50 | H100 |
|---|---:|---:|---:|---:|
| P0 persistence | 1.706 | 2.948 | 4.001 | 4.770 |
| P1 local mean | 2.988 | 3.263 | 3.596 | 4.043 |
| P2 independent AR | 1.793 | 3.077 | 3.783 | 4.241 |
| P5 adaptive iid | 1.669 | 2.962 | 4.300 | 4.845 |
| P6 adaptive block | 2.007 | 4.605 | 7.387 | 8.838 |
| P7 adaptive diagonal | **1.625** | **2.838** | 4.145 | 4.711 |

P7 is best at short horizons, but its error grows faster than persistence and
local mean at H50–H100. This is consistent with variance accumulation under a
near-unit-root operator.

## Post-hoc exploratory finding

P7 (diagonal Gaussian innovations) was not the primary model. It has the best
energy among adaptive variants:

- energy 2.838 vs local mean 3.263, paired improvement `+0.425`
  `[+0.078, +0.789]`;
- vs independent AR `+0.239` `[−0.018, +0.470]`, interval includes zero;
- 95% coverage 0.986 and total std ratio 1.87: still overdispersed.

This is a hypothesis for future work, not a gate pass. It is reported because
it narrows the failure: full covariance and block dependence make the variance
problem worse, while diagonal innovations reduce it but do not solve it.

## Why energy and coverage disagree

A proper score rewards both sharpness and calibration. P7 has a better mean than
the local mean, which can offset some overdispersion in energy, while coverage
clearly exposes the overdispersion. P1 has better coverage but a less accurate
conditional mean. Neither is a calibrated simulator. The prespecified decision
uses the primary model, not the post-hoc best variant.

## Artifacts

- `prob_summary.csv`, `prob_paired.csv`, `prob_paired_all.csv`
- `prob_per_landmark.csv`, `prob_per_neuron_h20.csv`
- `prob_per_session.csv`, `prob_horizon.csv`
- `prob_variance_decomposition.csv`, `prob_variance_summary.csv`
- `prob_pit.csv`, `prob_configurations.json`, `prob_metadata.json`
- `figures/prob_energy_comparison.png`, `prob_calibration.png`,
  `prob_variance_decomposition.png`, `prob_horizon.png`

All scores are model-based and observational. They do not validate a biological
generative mechanism.
