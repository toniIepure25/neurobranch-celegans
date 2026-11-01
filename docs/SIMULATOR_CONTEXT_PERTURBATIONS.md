# Bounded named-neuron perturbations

## 1. Scope

Targets are limited to **AVBL, AVBR, AVEL, AVER, RIVL, RIVR**, and only when the named neuron is verified present in the recording. Missing targets are not silently substituted.

Two protocols:

- `initial_state`: edit the model state at the rollout start;
- `dynamical_input`: edit the decoded model state for five steps.

Amplitudes are bounded by the smaller of:

- one-step standard deviation of the target's standardized state change;
- half the target's central 90% range.

A same-amplitude non-target alphabetic neuron is used as a control. Perturbation effects are measured as state shift, support distance, and first-exit outcome.

## 2. What these are not

- Not optogenetic stimulation.
- Not an experimental dose.
- Not a synaptic weight change.
- Not evidence of a causal circuit role.
- Not a validated intervention target.

The model is observational; the perturbation is a mathematical intervention on that model.

## 3. Gate status

- Gate 7 (named perturbation validation): **NOT_EVALUABLE** without matched experimental interventions.
- Gate 8 (intervention compatibility): **NOT_EVALUABLE** without a compatible intervention dataset.

No target optimization or large search is performed. A named target is not promoted merely because its model shift exceeds a control in this exploratory run.

## 4. Results

All six targets were present in all five recordings. There were 600 events (5 frames × 6 targets × 2 protocols × 2 directions × 5 sessions), each paired with a same-amplitude control.

Largest mean excess final shift over the matched control:

| Target / protocol | Mean excess |
|---|---:|
| AVBL dynamical | 0.075 |
| AVBR dynamical | 0.055 |
| RIVL dynamical | 0.053 |
| RIVR dynamical | 0.034 |
| AVBL initial | 0.016 |
| AVBR initial | 0.011 |

Positive and negative amplitudes produce identical shift norms because the model is linear. Support q95 was 0.981 for all target events. These are model sensitivities; every matched control also moves the model. No target is promoted as a causal or experimental result.

## 5. Reproduction

```powershell
python scripts/context_perturbations.py
```
