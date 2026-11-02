# Residual and innovation diagnostics

## Why residuals must be prequential

In-sample residuals from a rolling VAR are optimistically small and temporally
dependent by construction. They cannot be used as out-of-sample innovations.
We therefore use **block-prequential** residuals:

1. fit the rolling operator at a block start using only data up to that point;
2. predict transitions one step ahead inside a block;
3. record `target − prediction`;
4. move to the next block and refit.

The default block length is 50 frames. Frozen, local-mean, independent-AR and
persistence residuals use their own causal fits and the same target grid.

Training residuals (`target ≤ last train frame`) are diagnostics and the only
data used to choose the innovation model. Test residuals are reported
descriptively and never used for that choice.

## Adaptive residual diagnostics

Aggregated over the five sessions:

| diagnostic | adaptive VAR | local mean | independent AR | frozen VAR | persistence |
|---|---:|---:|---:|---:|---:|
| residual trace | 82.5 | 74.8 | 42.0 | 31.2 | 56.5 |
| median lag-1 \|ACF\| | 0.584 | 0.722 | 0.108 | 0.107 | 0.469 |
| mean \|cross-neuron correlation\| | 0.108 | 0.124 | 0.059 | 0.029 | 0.030 |
| mean excess kurtosis | 3.06 | 2.11 | 4.94 | 1.25 | 1.23 |
| mean absolute skew | 0.59 | 0.48 | 0.78 | 0.20 | 0.14 |
| effective rank | 27.4 | 20.3 | 54.6 | 58.5 | 53.0 |
| condition number | 381.5 | 1319.1 | 92.8 | 76.9 | 107.7 |
| off-diagonal covariance fraction | 0.743 | 0.896 | 0.452 | 0.303 | 0.301 |
| covariance drift across thirds | 1.49 | 0.90 | 0.78 | 0.44 | 0.48 |
| residual-norm vs state-norm correlation | 0.379 | 0.457 | 0.370 | 0.095 | 0.192 |
| out-of-block predictability R² | −0.92 | −0.30 | −0.88 | −2.15 | −0.80 |

The negative out-of-block R² is not evidence of strong nonlinear structure. It
means a linear map from recent state to the next residual did **not** generalize
across blocks and was often worse than predicting the held-out residual mean.
The large positive residual-norm/state-norm correlation and drift indicate that
the residual scale is state- and time-dependent.

### Temporal dependence

Training residual median absolute autocorrelation:

| model | lag 1 | lag 2 | lag 5 | lag 10 |
|---|---:|---:|---:|---:|
| adaptive VAR | 0.584 | 0.590 | 0.514 | 0.397 |
| local mean | 0.722 | — | — | — |
| independent AR | 0.108 | 0.264 | 0.239 | 0.207 |
| frozen VAR | 0.107 | 0.073 | 0.043 | 0.026 |
| persistence | 0.469 | 0.028 | 0.022 | 0.021 |

Adaptive residuals are not white. The lag-1 and lag-2 values are similar, which
is consistent with slow calcium dynamics and with block-boundary effects in the
diagnostic sequence. The dependence is too large to justify iid Gaussian
innovations.

### Covariance support

Ledoit-Wolf shrinkage was small for the adaptive residuals (mean 0.041), but the
trace was large (`82.5`) and the effective rank was only `27.4` out of roughly
103 predictors. Cross-neuron covariance is substantial
(`off-diagonal fraction 0.743`). A full unconstrained covariance would be poorly
conditioned; the implementation therefore combines Ledoit-Wolf shrinkage with an
eigenvalue floor and verifies positive semidefiniteness.

The covariance is not stationary: the relative drift across thirds is `1.49`,
and residual variance depends on recent state. A single frozen innovation
covariance is an approximation, not a complete measurement model.

## Innovation-model choice

The prespecified rule was:

> use a moving-block bootstrap if the median absolute lag-1 training residual
> autocorrelation exceeds 0.1; otherwise use an iid empirical bootstrap.

The adaptive estimate was `0.570` (the median over sessions of the per-session
median), so the primary model became **P6_adaptive_block** with block length 10.
The iid bootstrap (P5), Gaussian covariance (P4) and diagonal Gaussian (P7)
variants were retained as sensitivity analyses.

This is a legitimate training-only decision. It does **not** validate the model.
The block choice is exactly what later produces catastrophic variance inflation
when combined with a near-unit-root operator.

## Why the block bootstrap fails here

A stationary VAR with companion spectral radius near `0.99` has a large
unconditional variance for a given innovation covariance. Preserving positive
residual autocorrelation with 10-frame blocks increases the effective
low-frequency innovation power. The resulting simulated variance is far above
the observed variance:

| quantity | adaptive block P6 | observed |
|---|---:|---:|
| mean within-ensemble variance | 11.46 | 0.861 |
| total implied variance | 12.12 | 0.861 |
| total standard-deviation ratio | 3.40 | 1.0 |

The mean forecast remains accurate; the uncertainty model is not. This is the
central structural failure documented in the probabilistic evaluation.

## What is supported

- Innovations are strongly non-iid and cross-neuron correlated.
- The residual covariance is high-dimensional, ill-conditioned and
  nonstationary.
- A training-only rule favors block resampling.
- The selected block model overdisperses severely.

## What is not supported

- A claim that the residual distribution is Gaussian.
- A claim that a single covariance matrix is adequate.
- A claim that block bootstrap residuals produce calibrated trajectories.
- A biological interpretation of residual state dependence.

## Artifacts

- `residual_diagnostics.csv`
- `residual_autocorrelation.csv`
- `residual_covariance.csv`
- `residual_covariance_sensitivity.csv`
- `residual_summary.csv`
- `innovation_choice.json`
- `figures/prob_residual_autocorrelation.png`

All diagnostics are retrospective descriptions of repeated sessions. No
animal-level inference is made.
