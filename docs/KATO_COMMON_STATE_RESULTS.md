# Kato common-state controls: results

Status: exploratory reanalysis of the five Kato `WT_NoStim` recordings. The test
block has been reused; this is not fresh independent confirmation. All intervals
are repeated-session moving-block intervals, not animal-level intervals.

## Bottom line

The matched population VAR (B3) retains a positive H20 energy advantage over the
primary own-neuron-plus-ten-factor control (B5-10):

- `B5 - B3 = +0.2644` energy units, block-5 95% interval `[+0.0750, +0.4031]`,
  B3 better at 64.0% of paired landmarks.
- The population advantage over the matched-window own-neuron baseline (B4m) is
  `+0.5395`; the common-factor increment is `+0.2751` (51.0% of that gain), and
  the residual cross-neuron increment is `+0.2644` (49.0%).
- The residual advantage is not uniform: it appears at H5-H20, is absent at H1,
  and is positive in four of five sessions. Session 2 reverses.

This is conditional predictive information beyond the prespecified own-history,
drift and common-factor controls. It is not direct coupling, a synapse, a command
neuron, or a biological intervention target.

## Historical reproduction

`reports/simulator/kato_common_state/reproduction_summary.json`:

- maximum absolute H20 energy difference across B1h, B3, B4r, J2 and J5:
  `9.43e-7`, below the `1e-6` energy tolerance;
- maximum absolute H20 MSE difference: `1.65e-6`, a floating-point-level
  discrepancy reported separately;
- J2/J5 mean signed energy difference: `+0.0001337`;
  block-5 interval `[-0.02664, +0.02468]`; correlation `0.99224`;
- G1 PASS for the declared H20 energy endpoint.

The previous "103 predictors" statement was correct only for session 0. The
native dimensions used here are 103, 127, 123, 118 and 121.

## Matched ladder

Mean H20 over 175 landmarks:

| Model | Energy | MSE | 95% coverage |
| --- | ---: | ---: | ---: |
| B0 persistence | 3.6570 | 2.0560 | 0.982 |
| B1m own AR (200) | 3.0459 | 1.9298 | 0.777 |
| B2 own AR + drift | 3.5685 | 2.2942 | 0.635 |
| B4m own AR (500) | 3.2397 | 2.1265 | 0.791 |
| B5-10 own + factors | 2.9646 | 1.7612 | 0.725 |
| B3 population VAR | 2.7001 | 1.7067 | 0.938 |

B3 improves over B1m by `0.3458` energy (`[0.1018, 0.5013]`; B3 better at 59.4%).
The longer-window own model B4m is worse than B1m, so window length alone does not
explain the population advantage. The local linear drift term B2 is worse than
B1m; a simple global trend does not help.

Factor-count sensitivity is monotone:

| B5 factors | Energy | MSE |
| ---: | ---: | ---: |
| 1 | 3.1379 | 2.0174 |
| 3 | 3.0795 | 1.9262 |
| 5 | 3.0149 | 1.8423 |
| 10 | 2.9646 | 1.7612 |

Even the maximal prespecified factor control leaves B3 better.

## Decomposition

Positive values mean the candidate model has lower energy.

| Contrast | Mean | Block-5 interval | Fraction better |
| --- | ---: | --- | ---: |
| own history (B1m over B0) | +0.6110 | `[+0.4929, +0.8370]` | 0.794 |
| local drift (B2 over B1m) | -0.5226 | `[-0.7150, -0.3021]` | 0.223 |
| window (B4m over B1m) | -0.1938 | `[-0.3654, -0.0689]` | 0.406 |
| common factors (B5 over B4m) | +0.2751 | `[+0.1505, +0.4082]` | 0.651 |
| residual cross-neuron (B3 over B5) | +0.2644 | `[+0.0750, +0.4031]` | 0.640 |
| population (B3 over B4m) | +0.5395 | `[+0.3470, +0.6863]` | 0.726 |

The common and residual increments sum to the population-over-B4m gain by
construction. The factor control explains about half of the population advantage;
the other half is residual to the ten factors.

## Sessions, representation and horizon

Session-level `B5 - B3`:

| Session | Mean | Landmarks |
| ---: | ---: | ---: |
| 0 | +0.3003 | 35 |
| 1 | +0.3292 | 35 |
| 2 | -0.1931 | 34 |
| 3 | +0.4075 | 38 |
| 4 | +0.4645 | 33 |

B3 is better in sessions 0, 1, 3 and 4; session 2 reverses.

Representation sensitivity for `B5 - B3`:

| Space | Mean | Interval |
| --- | ---: | --- |
| PCA-10 energy | +0.2644 | `[+0.0750, +0.4031]` |
| PCA-10 MSE | +0.0545 | `[-0.1574, +0.2409]` |
| neuron energy | +0.1617 | `[-0.1387, +0.4653]` |
| neuron MSE | -0.0354 | `[-0.1682, +0.0705]` |

The energy direction survives in neuron space; the MSE direction is mixed and its
interval includes zero. This is why the result is reported as an energy result,
not as a general forecast improvement.

Horizon profile, mean energy:

| Horizon | B3 | B5-10 | B5 - B3 |
| ---: | ---: | ---: | ---: |
| 1 | 1.273 | 1.209 | -0.064 |
| 5 | 1.505 | 1.589 | +0.083 |
| 10 | 1.981 | 2.128 | +0.147 |
| 20 | 2.700 | 2.965 | +0.264 |

The B3 advantage grows with horizon. At H1 the factor model is better. B3 also has
better 95% coverage (0.938 vs 0.725), so the B5 energy deficit is accompanied by
under-coverage.

Training-only factor variance reaches 54.7%-71.0% at ten components across
sessions. These are statistical common factors, not measured inputs.

## Surrogates

Six draws per surrogate per landmark, with B3 scored on the same landmarks and a
positive value defined as `surrogate energy - B3 energy` (B3 lower is better):

| Surrogate | Mean B3 advantage | Block-5 interval | Fraction better |
| --- | ---: | --- | ---: |
| common-preserved residual shift | +0.5573 | `[+0.3162, +0.7677]` | 0.749 |
| circular full shuffle | +1.2704 | `[+0.8898, +1.5827]` | 0.811 |
| block full shuffle | +1.4379 | `[+1.0098, +1.7991]` | 0.811 |
| residual-preserved factor shift | +1.4707 | `[+1.1974, +1.6763]` | 0.880 |

The common-preserved residual shift is the relevant control: it keeps the common
factor series fixed and permutes only the residual. B3 is better than that
surrogate by `0.5573` energy units, so the observed advantage is not reproduced by
an arbitrary residual arrangement. The stronger full-shuffle surrogates are much
worse, as expected because they destroy both common and residual structure.

These are preservation checks, not a null distribution for a formal p-value. Six
draws describe the sensitivity of the result to the surrogate construction; they
do not turn the repeated-session audit into a population-level test.

## Synthetic specificity

`reports/simulator/kato_common_state/synthetic_summary.csv`:

| Scenario | Mean B3 - B5(3) MSE gain |
| --- | ---: |
| independent | -0.00031 |
| common | -0.00068 |
| coupled | +0.00219 |
| coupled_common | +0.00182 |
| time-varying common | -0.00090 |

Positive means B3 is better. The direct-coupling scenarios are positive in all
three seeds; the common-only scenario is negative. G5 PASS. The saturated
`B5-10` diagnostic is weaker but follows the same direction.

## Gates

| Gate | Status |
| --- | --- |
| G1 historical reproduction | PASS (H20 energy) |
| G2 fair matched baselines | PASS |
| G3 common-factor explanation | PASS |
| G4 residual cross-neuron information | PASS |
| G5 synthetic specificity | PASS |
| G6 robustness | PASS |
| G7 structural identifiability | FAIL |
| G8 biological intervention validity | `NOT_EVALUABLE` |

G4 passes because the prespecified positive point estimate, block-5 interval and
fraction-better criteria are all met. G7 remains FAIL: observational prediction
does not identify direct coupling. G8 remains `NOT_EVALUABLE`.

## Figures

Five generated figures are under `reports/simulator/kato_common_state/figures/`:
`model_ladder.png`, `decomposition.png`, `session_effects.png`,
`surrogate_controls.png` and `synthetic_controls.png`. They are regenerated by
`scripts/kato_common_state_figures.py` and hashed in the artifact manifest.

## Claim boundary

Allowed: residual conditional predictive information beyond own-neuron history,
local drift and ten training-only common factors, concentrated at H5-H20 and
positive in four of five repeated recordings.

Not allowed: direct coupling, synapse, command neuron, causal driver or biological
stimulation target.

**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**
