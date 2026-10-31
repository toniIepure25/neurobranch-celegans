# Phase 6 support-constrained perturbation

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

## Protocol as run

- Family: validation-selected stochastic VAR (`stochastic_gaussian`), 5 worms.
- 3 held-out test frames per worm.
- Targets: AVBL, AVBR, AVEL, AVER, RIVL, RIVR; both signs.
- Amplitude: \(q=1\) times the target's training one-step change SD, capped to half the central 90% marginal range.
- Protocols: initial-state additive edit; dynamical additive input for 1 or 5 frames.
- 15 rollouts per baseline/perturbed condition, matched initial history and seeds.
- Control: one deterministic non-candidate neuron with the same signed amplitude and protocol.
- Support: validation-calibrated 99th-percentile threshold.

## Results

| Quantity | Result |
|---|---:|
| Perturbation events | 540 |
| Conditions (target × protocol × direction) | 36 |
| Mean in-support fraction | **1.000** |
| Conditions with ≥90% in support | **36/36** |
| Mean outcome TV (target) | 0.0200 |
| Mean outcome TV (control) | 0.0123 |
| Mean readout support | 1.000 |

Figure: `figures/simulator/phase6/perturbation_support.png`.

Target-level exploratory TV:

| Target | Mean TV | Mean control TV | Worms with positive excess |
|---|---:|---:|---:|
| AVEL | 0.0252 | 0.0081 | 4/5 |
| AVBR | 0.0259 | 0.0096 | 3/5 |
| AVBL | 0.0252 | 0.0193 | 4/5 |
| RIVL | 0.0200 | 0.0119 | 4/5 |
| RIVR | 0.0178 | 0.0133 | 3/5 |
| AVER | 0.0089 | 0.0126 | 1/5 |

## Gate 5

**Pass.** Every condition had 100% of perturbed states inside calibrated support. The threshold is permissive for q=1 edits: a tiny initial edit does not leave the validation envelope. Passing G5 means "the model was not obviously extrapolating", not "the perturbation is biologically valid".

## Gate 6

**Not evaluable.** The protocol requires a consistent target/direction sign in at least four of five worms **and at least two of three model families**, conditional on G5. Only the stochastic VAR family was run, so the family requirement cannot be met. AVEL is the strongest exploratory signal (positive excess in 4/5 worms, mean TV 0.025 vs 0.008 control), but it is one family, one reused test block, and a very small four-category TV.

The control is also weaker than the protocol: it is a deterministic random neuron, not matched on innovation SD and autocorrelation. Therefore even the exploratory target ranking is descriptive.

## Interpretation boundary

The observed TV values are small and quantised by the four behavioural labels. Do not claim:

- that AVEL or any neuron drives reversal;
- a direction-specific biological effect;
- that a 1-frame input is optogenetic stimulation;
- that in-support implies mechanistic validity.

A defensible follow-up would run the same frozen perturbation protocol on at least two more model families with innovation-matched controls and an independent cohort.
