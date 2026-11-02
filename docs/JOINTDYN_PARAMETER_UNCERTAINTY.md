# JOINTDYN parameter uncertainty

**Result: parameter uncertainty is measurable but unhelpful. It increases dispersion, worsens coverage and worsens both H20 energy and trajectory energy. The stop rule therefore applies.**

## 1. Decomposition target

The forecast distribution can contain four conceptually distinct sources:

1. process innovation uncertainty;
2. estimated transition-parameter uncertainty;
3. initial-state uncertainty;
4. observation uncertainty.

This experiment holds the initial state at the observed value and cannot separate observation from process noise. It tests source 2 against source 1.

## 2. Synthetic audit

A stable two-dimensional VAR was generated with

```
A = [[0.72, 0.08], [-0.10, 0.62]]
Q = diag(0.20, 0.12)
```

One 700-frame training series was fitted by ridge VAR. Twenty parameter draws were produced by a moving-block bootstrap of training transition pairs (block 25). Coverage was evaluated against 200 independent true future paths, not a single path.

| Model | 50% | 80% | 95% | Width 95 | Energy H20 |
|---|---:|---:|---:|---:|---:|
| Innovation only | 0.460 | 0.748 | 0.930 | 2.036 | 0.498 |
| Parameter ensemble | 0.513 | 0.750 | 0.915 | 1.916 | 0.490 |
| Oracle A | 0.493 | 0.793 | 0.948 | 2.173 | 0.488 |

The synthetic audit behaves as expected: the oracle is close to nominal, and parameter uncertainty slightly improves 50% coverage while slightly reducing the 95% width. The effect is small.

## 3. Real Kato test experiment

At each of the 175 test landmarks:

- five VAR models were fitted on moving-block bootstrap resamples of the causal 500-frame history;
- block length 25;
- each draw supplied 20 ensemble members, for 100 total;
- the process covariance was diagonal from the lead-1 residuals and held fixed across draws;
- no test refit and no future frames entered the fit.

This isolates parameter uncertainty from innovation-covariance uncertainty.

| Model | H20 energy | H20 trajectory energy | 50% | 80% | 95% | Neural covariance error |
|---|---:|---:|---:|---:|---:|---:|
| J2 population diag | **2.700** | **2.058** | 0.586 | 0.846 | 0.944 | 0.572 |
| Parameter ensemble | 2.807 | 2.271 | 0.615 | 0.883 | 0.969 | 0.659 |

Parameter uncertainty:

- increases 50% coverage from 0.586 to 0.615 (further from nominal);
- increases 95% coverage from 0.944 to 0.969 (further from nominal);
- worsens H20 energy by 0.107;
- worsens trajectory energy by 0.213;
- worsens neural covariance error.

The parameter draw spread is modest, but the resulting added variance is in the same direction as the existing overdispersion. It does not repair the shape mismatch.

## 4. Interpretation

Parameter uncertainty is not a free calibration improvement. In this dataset the transition parameters are estimated from hundreds of frames, but the bootstrap distribution still injects variance into a model that is already overdispersed at the center. Adding it makes the predictive distribution less useful.

The result matches the prespecified stop rule: when parameter uncertainty worsens dispersion, stop rather than adding further stochastic components.

## 5. Initial state and observation uncertainty

- **Initial state:** fixed at the observed test state. No initial-state ensemble was added, because the evaluation question is conditional on that state.
- **Observation uncertainty:** `NOT_IDENTIFIABLE`. One calcium trace per neuron cannot separate measurement noise from latent process noise without an external noise model or repeated measurements.
- **Posterior interpretation:** the bootstrap draws are a frequentist resampling ensemble, not a Bayesian posterior. No posterior probability is claimed.

## 6. Artifacts

- `parameter_synthetic_audit.csv`
- `parameter_scores.csv`
- `parameter_summary.csv`
- `parameter_draws.csv`
- `parameter_decision.json`
- `uncertainty_decomposition.csv`
- `figures/jd_parameter.png`
