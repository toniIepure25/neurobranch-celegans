# Baseline ablation: adaptive population VAR versus local tracking

## Question

The deterministic extension showed that a rolling VAR at window 500 predicts H20
population activity better than a frozen train-only VAR. Does that advantage
require cross-neuron dynamics, or is it explained by local level adaptation?

## Design

All models use the existing chronological train/test split, the same test
landmarks (`35` per session, stride 10, `175` total), the same train-standardized
`X`, and H20 MSE. No test frame was used to choose a window.

Baseline settings were selected on the last 20% of the **training** split:

| model | candidate settings | selected | validation H20 MSE |
|---|---|---:|---:|
| rolling local mean | 100, 200, 500, 1000 | 100 | 0.703398 |
| EWMA local mean | half-life 100, 300, 1000 | 100 | 0.702249 |
| independent per-neuron AR | 200, 500, 1000 | 200 | 0.626809 |

The prespecified adaptive model B5 is the full Rolling VAR w500, ridge 1.0,
spectral cap 0.995. B0 is persistence; B1 is the frozen VAR; B4 fits one AR(1)
per neuron with no cross-neuron terms.

## Overall H20 results

Mean free-running H20 MSE across 175 landmarks:

| model | mean H20 MSE | paired improvement vs frozen | 95% block-bootstrap CI |
|---|---:|---:|---:|
| B0 persistence | 1.0601 | 0.5370 | [0.2307, 0.8946] |
| B1 frozen VAR | 1.5972 | 0 | — |
| B2 rolling local mean w100 | 0.9671 | 0.6301 | [0.3669, 0.9251] |
| B3 EWMA mean h100 | 0.9743 | 0.6229 | [0.4286, 0.8582] |
| B4 independent AR w200 | 0.9516 | 0.6455 | [0.3669, 0.9637] |
| **B5 rolling VAR w500** | **0.9406** | **0.6566** | **[0.3261, 0.9856]** |

The historical 41.1% aggregate relative improvement reproduces exactly. It is
`1 - mean(B5)/mean(B1)`. The equal-weight mean of per-landmark relative
improvements is 33.9%. Both are correct summaries of different estimands.

## The critical comparison

Positive improvement means the row model has lower MSE than the reference.

| comparison | mean improvement | 95% CI | landmarks better |
|---|---:|---:|---:|
| B5 − B2 local mean | +0.0265 | [−0.1292, +0.1518] | 70.3% |
| B5 − B4 independent AR | +0.0110 | [−0.0798, +0.0775] | 57.7% |
| B5 − B0 persistence | +0.1195 | [+0.0167, +0.2058] | 74.3% |

The B5 advantage over local mean and independent AR is small and its interval
includes zero. B5 is better than persistence, but so are the simpler adaptive
baselines:

| model | improvement over persistence | 95% CI |
|---|---:|---:|
| B2 local mean | +0.0931 | [−0.0727, +0.2765] |
| B3 EWMA | +0.0859 | [−0.1192, +0.2908] |
| B4 independent AR | +0.1085 | [+0.0304, +0.1737] |
| B5 rolling VAR | +0.1195 | [+0.0167, +0.2058] |

## Session heterogeneity

H20 MSE by repeated session:

| session | B1 frozen | B2 local mean | B4 indep. AR | B5 rolling VAR | B5 best? |
|---:|---:|---:|---:|---:|---|
| 0 | 1.220 | 0.541 | 0.471 | **0.446** | yes |
| 1 | 2.791 | 0.789 | 0.593 | **0.557** | yes |
| 2 | 2.178 | **2.092** | 2.442 | 2.505 | no |
| 3 | 0.855 | 0.702 | 0.519 | **0.468** | yes |
| 4 | 0.986 | **0.754** | 0.805 | 0.805 | no |

B5 wins in three sessions, but in session 2 it is worse than the local mean and
in session 4 the local mean is clearly best. This is not a uniform gain across
recordings.

Per-session paired B5 − B2:

| session | mean improvement | 95% CI |
|---:|---:|---:|
| 0 | +0.0957 | [+0.0383, +0.1688] |
| 1 | +0.2321 | [+0.1119, +0.3571] |
| 2 | −0.4133 | [−0.9542, +0.1201] |
| 3 | +0.2340 | [+0.1207, +0.4033] |
| 4 | −0.0510 | [−0.1792, −0.0102] |

The aggregate positive estimate is therefore sensitive to session composition.

## Decision

**Gate B: FAIL.** B5 does not improve over both B2 and B4 with a paired interval
excluding zero.

The defensible interpretation is:

> Adaptive local level tracking explains nearly all of the deterministic gain
> over the frozen VAR in this dataset. The experiment does not demonstrate that
> cross-neuron coupling adds predictive information beyond independent adaptive
> AR or a rolling local mean.

This is a negative result about the *added information* of the full VAR, not a
claim that the recorded population has no cross-neuron interactions. A linear
lag-one model with these windows may simply not capture them, and the local mean
may absorb slow shared state.

## Artifacts

- `reports/simulator/probabilistic_adaptive/baseline_validation.csv`
- `baseline_per_landmark.csv`, `baseline_per_neuron_h20.csv`
- `baseline_per_session_h20.csv`, `baseline_horizon.csv`
- `baseline_paired_vs_frozen.csv`, `baseline_paired_vs_local_mean.csv`,
  `baseline_paired_vs_independent_ar.csv`, `baseline_paired_by_session.csv`
- `baseline_updates.csv`, `baseline_metadata.json`

All intervals are moving-block bootstrap intervals over overlapping landmarks.
They are not animal-level confidence intervals.
