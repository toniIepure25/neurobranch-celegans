# Adaptive dynamics models and evaluation

## Model set

| ID | Model | Estimation | Stability | Cost per update |
|---|---|---|---|---|
| C1 | Frozen VAR | Weighted ridge lag-one VAR on the train split; fixed thereafter | Uniform spectral scaling to 0.995 | ~3 µs (no update) |
| C2 | Rolling VAR | Equal-weight ridge on the last 200/500/1000 transitions ending at the landmark | Same cap at each refit | 14–15 ms |
| C3 | EWMA VAR | Exponentially weighted ridge on all transitions ending at the landmark; half-life 100/300/1000 frames | Same cap at each refit | 26–27 ms |
| C4 | Local latent | Not fitted | — | — |

C1 reproduces the validated `StableVARModel` coefficients to numerical precision
(mean absolute coefficient difference below 2×10⁻⁴ on the checked session). The
frozen model is fitted on the train split only. Adaptive models may use any
frame up to the landmark, including earlier test frames, but never a frame from
the forecast horizon.

The prespecified primary adaptive model is `rolling_w500`. All six candidates
are reported; selecting the best candidate after seeing test results would be
test-set selection.

## Evaluation

- 175 test landmarks (35, 35, 34, 38, 33 by recording; every 10 frames),
  identical for every model. The earlier "35 per session" wording was a
  documentation error (X03); it did not change the analysis.
- Horizons 1, 5, 10, 20, 50, 100 frames. At 3 fps this is approximately
  0.3–34 s.
- Forecasts are fully autonomous after the landmark.
- Primary metric: paired H20 MSE improvement versus frozen VAR.
- Uncertainty: moving-block bootstrap over landmarks, block 5, 1000 draws.
- All five sessions are reported; no session is dropped.

## Paired H20 results

| Model | Mean MSE | MSE improvement vs frozen | 95% CI | Relative improvement | Landmarks better | Sessions better |
|---|---:|---:|---:|---:|---:|---:|
| EWMA h1000 | 0.839 | 0.758 | [0.523, 1.029] | 44.0% | 94.3% | 5/5 |
| Rolling w1000 | 0.845 | 0.752 | [0.504, 1.036] | 42.8% | 93.1% | 5/5 |
| EWMA h300 | 0.883 | 0.714 | [0.394, 1.035] | 39.5% | 92.0% | 4/5 |
| **Rolling w500 (primary)** | **0.941** | **0.657** | **[0.326, 0.986]** | **33.9%** | **86.3%** | **4/5** |
| EWMA h100 | 1.091 | 0.506 | [−0.004, 0.911] | 24.1% | 84.0% | 4/5 |
| Rolling w200 | 1.390 | 0.207 | [−0.377, 0.706] | −2.6% | 78.9% | 3/5 |
| Frozen VAR | 1.597 | — | — | — | — | — |

The primary model passes the prespecified Gate C criterion: its 95% bootstrap
interval excludes zero. The advantage is not confined to one session, although
session 2 is the weakest for the primary model and session 1 is the weakest for
the best aggregate model.

## Horizon behaviour

| Model | H1 | H5 | H10 | H20 | H50 | H100 |
|---|---:|---:|---:|---:|---:|---:|
| Frozen | 1.179 | 1.427 | 1.465 | 1.597 | 1.809 | 1.975 |
| Rolling w500 | 0.332 | 0.374 | 0.601 | 0.941 | 1.328 | 1.455 |
| EWMA h1000 | 0.284 | 0.400 | 0.594 | 0.839 | 1.147 | 1.336 |
| Rolling w1000 | 0.297 | 0.387 | 0.585 | 0.845 | 1.185 | 1.346 |

The adaptive advantage is present at every horizon, not only at H20. It shrinks
from roughly a factor of 3.5 at H1 to roughly 1.4 at H100. At long horizons all
models converge toward low-variance trajectories, and MSE growth slows because
the forecasts collapse toward a local mean rather than because they recover the
true dynamics.

## Skill versus persistence

| Model | Mean H20 skill | Median per-neuron skill | Fraction neurons >0 | Fraction neurons >0.1 |
|---|---:|---:|---:|---:|
| Frozen | −1.218 | −0.997 | 39.0% | 37.4% |
| Rolling w500 | 0.134 | 0.158 | 53.5% | 51.4% |
| EWMA h300 | 0.201 | 0.194 | 54.1% | 52.1% |
| EWMA h1000 | 0.146 | 0.115 | 52.2% | 50.3% |

Mean per-neuron skill is dominated by a few neurons with near-zero persistence
error; medians and fractions are the appropriate summaries. Roughly half of
neuron-landmark series beat persistence under adaptation, compared with about
39% under the frozen model. This is better than the 20–30% reported in the
previous stage, but it is not a majority of all neurons and is not evidence of
a recovered generative mechanism.

## Distributional fidelity at H20

| Model | Std ratio | Autocorrelation error | Covariance relative error | OOD fraction |
|---|---:|---:|---:|---:|
| Frozen | 0.360 | 0.748 | 1.080 | 2.1% |
| Rolling w500 | 0.238 | 0.556 | 1.018 | 5.4% |
| EWMA h1000 | 0.224 | 0.783 | 0.996 | 4.1% |
| Rolling w1000 | 0.224 | 0.697 | 0.999 | 4.3% |

Adaptation improves autocorrelation and slightly improves covariance relative
to frozen, but forecasts are severely under-dispersed: predicted standard
deviations are only about one quarter of the observed standard deviations. The
covariance error remains approximately the size of the observed covariance
norm. Gate D therefore **fails**. A lower MSE obtained by shrinking toward a
local mean is not a distributional reproduction of the neural dynamics.

## Update stability and cost

- All adaptive spectral radii are capped at 0.995. Realized means are
  0.990–0.994; 19–47% of refits required clipping.
- No non-finite adaptive forecast occurred.
- Rolling updates cost 14–15 ms; EWMA updates 26–27 ms; frozen inference is
  effectively free.
- The 200-frame rolling model is the least stable in error growth
  (H100 MSE 2.35), showing that faster adaptation is not monotonically better.

## C4 decision

C4 was not fitted. Stage A found large, partly artifactual orientation changes
and no stable train-only latent frame. A local latent model would require an
explicit causal alignment mechanism, and none of C1–C3 had yet established that
the frozen/adaptive comparison was resolved. Adding C4 would have been an
architecture expansion without a diagnostic payoff.
