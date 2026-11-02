# Stochastic adaptive models

## Conditional mean

At landmark `t`, the adaptive model fits

\[
x_{s+1} = A_t x_s + b_t + \varepsilon_{s+1},
\]

using transitions with both endpoints at or before `t`. The ridge penalty is
`1.0` and the spectral radius is capped at `0.995`. The fitted `A_t, b_t` are
**frozen for the whole rollout**. No future observation is assimilated and no
operator is re-estimated inside the horizon.

The deterministic forecast is the noise-free recursion from the same observed
`x_t`. The stochastic model adds sampled innovations but does not change the
conditional mean.

## Innovation models

| id | innovation | covariance / dependence | role |
|---|---|---|---|
| P4 | Gaussian | Ledoit-Wolf full covariance, eigenvalue floor | smooth uncertainty |
| P5 | iid empirical bootstrap | centered prequential residuals, independent draws | non-Gaussian marginal |
| **P6** | moving-block bootstrap | contiguous 10-frame residual blocks, centered | preserves short-range dependence |
| P7 | diagonal Gaussian | per-neuron variance only | covariance sensitivity |

P6 is the prespecified primary model because training residual lag-1
autocorrelation exceeded the frozen 0.1 threshold.

Innovations are centered before sampling. The covariance matrix used by the
Gaussian model is symmetrized, shrunk and eigenvalue-floored; positive
semidefiniteness is tested. No variance is multiplied by an arbitrary
calibration factor.

## Rollout

- 100 ensemble members per landmark.
- Horizons 5, 20, 50 and 100 are stored from one 100-step rollout.
- Initial state is the observed `x_t`; the observed future is never used.
- The same fitted operator and innovation pool are used for every member.
- Seeds are fixed per session and model; repeated runs are reproducible.
- All trajectories are checked for finiteness.

The recursion is

\[
x^{(m)}_{t+h} = A_t^h x_t + \sum_{j=1}^{h} A_t^{h-j}(b_t + \varepsilon^{(m)}_{t+j}).
\]

For a stable `A_t`, the conditional mean converges to the local fixed point. The
forecast-error covariance is a sum of propagated innovation covariances, so a
spectral radius near one amplifies innovation variance over the horizon.

## Strong probabilistic baselines

Every baseline uses the same landmarks, initial states, ensemble size,
projection and score functions.

| id | mean model | innovation |
|---|---|---|
| P0 | persistence | moving-block bootstrap of one-step differences, block 20 |
| P1 | rolling local mean w100 | moving-block bootstrap, block 10 |
| P2 | independent adaptive AR w200 | diagonal Gaussian |
| P3 | frozen train-only VAR | moving-block bootstrap, block 10 |

This is a deliberately hard baseline set. P0 and P1 are not straw men: they are
calibrated with block resampling and are competitive on proper scores.

## Why the primary model overdisperses

The fitted adaptive operator has a spectral radius close to the cap. Its
one-step residual trace is `82.5`, versus `31.2` for the frozen VAR. Propagating
that residual covariance through a near-unit-root operator yields a large
stationary variance.

With block bootstrap, positive residual dependence adds low-frequency power.
The observed and implied variances at H20 are:

| model | within-ensemble variance | total implied variance | total std ratio |
|---|---:|---:|---:|
| P1 local mean | 0.671 | 0.909 | 1.11 |
| P2 independent AR | 0.753 | 1.151 | 1.11 |
| P0 persistence | 0.900 | 1.691 | 1.58 |
| P7 adaptive diagonal | 2.099 | 2.673 | 1.87 |
| P5 adaptive iid | 2.237 | 2.811 | 1.90 |
| P4 adaptive Gaussian | 2.515 | 3.105 | 2.01 |
| **P6 adaptive block** | **11.461** | **12.122** | **3.40** |
| observed | — | 0.861 | 1.00 |

The failure is not that the mean forecast is bad. It is that the innovation
model and the near-unit-root mean operator are mutually inconsistent with the
observed covariance.

## Synthetic checks

The test suite includes:

- Monte Carlo recovery of the conditional mean from 3,000 trajectories
  (tolerance 0.03);
- recursive horizon alignment at step 1;
- long-horizon finiteness with a spectral radius of 0.99;
- seed reproducibility;
- block resampling preserving lag-1 dependence better than iid resampling;
- PSD covariance;
- centered innovations;
- a synthetic linear system where cross-neuron coupling is genuinely useful,
  used to check that the code can represent cross-neuron information.

These tests validate implementation behavior, not the biological model.

## Model cards

- **P6_adaptive_block** — prespecified primary. Rolling VAR w500, moving-block
  empirical innovations, block 10. Status: failed calibration and variance
  fidelity.
- **P5_adaptive_boot** — iid bootstrap comparator. Remains overdispersed
  (total std ratio 1.90).
- **P4_adaptive_gauss** — Gaussian comparator. Total std ratio 2.01.
- **P7_adaptive_diag** — post-hoc sensitivity, **not** the primary model.
  Best energy among adaptive variants, but total std ratio 1.87 and 95% coverage
  98.6%; not calibrated.

## Artifacts

- `prob_configurations.json`
- `prob_model_cards.json`
- `prob_per_landmark.csv`, `prob_per_neuron_h20.csv`
- `prob_variance_decomposition.csv`, `prob_variance_summary.csv`
- `prob_updates.csv`

The stochastic model is a mathematical object fitted to observational data. It
is not a biological generative mechanism and it has no interventional
validation.
