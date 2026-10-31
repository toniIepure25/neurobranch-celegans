# Neuronal perturbation methods

> **Standing label:** `MODEL-DEPENDENT VIRTUAL PERTURBATION - NOT CAUSALLY
> VALIDATED.` A simulated effect is not evidence that the same manipulation
> changes a living worm.

## 1. Three perturbation categories

| Category | Meaning | Status here |
|---|---|---|
| A. Initial-state sensitivity | Change `x(0)` and ask how the model's future changes. | Implemented; mathematical probe only. |
| B. Dynamical input | Add/scale/clamp a named neuron during a window of the rollout. | Implemented; model-state edit. |
| C. Experimentally constrained response | Use a matched empirical stimulation-response model. | `not_evaluable`; refused by code. |

Category C is the only category that could support a biological intervention
claim. No optogenetic, histaminergic, mechanical, or thermal response model is
registered for Kato 2015. `PerturbationSpec(protocol="empirical_response")`
therefore raises `PermissionError`. Gate 6 is `not_evaluable`.

## 2. Named neurons

Six literature-motivated predictor neurons were **pre-registered** in
`docs/SIMULATOR_RESEARCH_PROTOCOL.md` and are present in every worm's predictor
set:

```text
AVBL, AVBR, AVEL, AVER, RIVL, RIVR
```

`AVA`, `SMD`, and `RIB` are excluded because they define the behavioural labels;
perturbing them would be circular. The list is a hypothesis set, not a claim
that these are the six most important neurons.

A pilot sweep used AIBL/AIBR before the frozen target list was checked. That
pilot is disclosed in the protocol deviation table and is **not** used for Gate
5. The primary analysis uses AVBL/AVBR.

## 3. Protocols and units

- `dynamical_input`: additive change in training-standard-deviation units,
  applied after each predicted state for `duration` frames beginning at `onset`.
- `initial_state`: additive change to `x(0)` only. It is not stimulation and does
  not test a persistent input.
- Positive amplitude is an activation-like model edit; negative amplitude is an
  inhibition-like edit. These words describe the sign of a model variable, not a
  biological mechanism.

The simulator's `rollout` applies perturbations after each model prediction:

```text
x_hat(t+1) = F_theta(x_hat(t)) ; then x_hat(t+1) <- P(x_hat(t+1), t)
```

This is a state edit in the model, not an intervention on the animal.

## 4. Experimental design

Frozen main sweep:

- five worms, ten held-out reversal states per worm (50 frame-worm starts);
- six neurons × two directions (amplitude ±1);
- 10 rollouts per condition, horizon 15, noise SD 0.05 (100 paired rollouts per
  target, satisfying the G4 minimum);
- model seed 17 for the main sweep;
- baseline rollouts use the same seeds and noise.

Controls and robustness:

- **random-neuron controls**: 30 named predictor neurons outside the six
  candidates, sampled with a fixed RNG seed, same amplitude and frames;
- **dose**: amplitudes −2, −1, −0.5, +0.5, +1, +2;
- **timing**: onset 0, 5, 10 with duration 5;
- **model seed**: seeds 17, 23, 31;
- **initial-state** probes at ±1.

Primary outcome metrics:

- `delta_tv`: total-variation distance between perturbed and baseline
  first-outcome distributions;
- `delta_turn`: change in probability of `TURN` (signed);
- `trajectory_shift_sd`: mean Euclidean state displacement divided by the mean
  training SD;
- `perturbed_ood_fraction`: fraction of perturbed states beyond the 99th
  percentile training support distance.

## 5. Gate 5 result

Frozen criterion: at least one named-neuron direction has mean `|delta TV|`
above the random-neuron 95th percentile, agrees in sign in ≥4/5 worms, and
agrees in ≥2/3 model seeds.

Result: **fail.**

- Random-control 95th percentile of `|delta TV|`: **0.500** (controls include
  zero-effect neurons and a few large flips; this is a deliberately conservative
  descriptive tail).
- Top candidate: **AVEL inhibition**, mean `|delta TV|` 0.146, mean
  `delta P(TURN)` +0.106, same sign in **2/5** worms, **3/3** model seeds.
- Next: RIVL activation 0.102, AVBR activation 0.058, AVBL activation 0.048,
  RIVR activation 0.046.
- No candidate exceeds the control tail. None is robust in ≥4/5 worms.
- Perturbed OOD fraction is **0.98–0.99** for all candidates, and the mean
  trajectory displacement is 7.5–8.0 training SD. The model is extrapolating
  almost everywhere during a perturbation.

| Target | Direction | Mean \|ΔTV\| | Mean ΔP(TURN) | Worms same sign | Seeds same sign | > control q95 |
|---|---|---:|---:|---:|---:|---|
| AVEL | inhibition | 0.146 | +0.106 | 2/5 | 3/3 | no |
| RIVL | activation | 0.102 | +0.102 | 3/5 | 3/3 | no |
| AVBR | activation | 0.058 | +0.038 | 1/5 | 2/3 | no |
| AVBL | activation | 0.048 | 0.000 | 0/5 | 0/3 | no |
| RIVR | activation | 0.046 | +0.026 | 2/5 | 2/3 | no |
| RIVL | inhibition | 0.040 | 0.000 | 0/5 | 0/3 | no |

`delta TV` is non-negative, so sign robustness is assessed on signed
`delta P(TURN)`, not on TV.

## 6. Dose and timing

Dose–response is not monotonic in the expected direction for most candidates.
Spearman ρ of signed amplitude versus `delta TV`:

| Target | ρ (amplitude, TV) | ρ (amplitude, ΔP(TURN)) |
|---|---:|---:|
| AVBL | +0.299 | +0.085 |
| AVBR | +0.229 | +0.174 |
| AVEL | −0.355 | −0.289 |
| AVER | −0.093 | −0.145 |
| RIVL | +0.196 | +0.340 |
| RIVR | +0.158 | +0.130 |

The ρ for AVEL is negative on both metrics: larger activation-like amplitudes
produced *less* turning. A negative TV ρ can also mean the trajectory changed
less in total. AVBL, AVBR, RIVL and RIVR have positive trends, but their absolute
effects are below the random-control tail. No candidate has a clean monotonic
dose–response in the expected direction **and** a specific effect.

Timing: trajectory displacement decreases as onset moves later (onset 0 → 10),
which is expected because there is less remaining horizon for divergence. Outcome
effects also shrink: AVBR activation `delta TV` 0.034 → 0.014, RIVL activation
0.038 → 0.008, RIVR activation 0.046 → 0.004. At onset 10 only AVBR and RIVL
retain a small nonzero `delta P(TURN)` (0.008 each). No timing condition is
specific versus controls.

Initial-state probes move the trajectory by about 2.5 training SD but change the
outcome distribution little on average (mean `delta TV` 0.002–0.020; one AVBL
case reaches 1.0, most are ≤0.2). Perturbed OOD is 0.98–0.99 even for these
small probes. Mathematical sensitivity is therefore much easier to detect than a
behaviourally meaningful change.

## 7. What this does and does not show

**Shows:** the interface can represent a named predictor neuron, apply a
time-windowed model input, run matched baseline/control ensembles, and quantify
trajectory and outcome-distribution changes. The implementation is tested.

**Does not show:** that inhibiting AVEL changes turning in a real worm; that any
ranking is specific; that the effect survives a different model class; or that
the perturbation is experimentally realisable. The random-control tail, the
worm/seed disagreement, and the ~99% OOD fraction are the reasons to reject a
specificity claim.

## 8. Reproduction

```powershell
python scripts/simulator_perturb.py --horizon 15 --rollouts 10 --frames 8
python scripts/simulator_figures.py
```

Outputs: `reports/simulator/perturbation_effects.csv`,
`perturbation_initial_state.csv`, `perturbation_dose.csv`,
`perturbation_timing.csv`, `perturbation_seed.csv`,
`perturbation_controls.csv`, `perturbation_specificity.csv`,
`perturbation_dose_summary.csv`, `perturbation_summary.json`.

## 9. Required next evidence

1. A matched stimulation dataset with the same neuron identities and a known
   response kernel.
2. A held-out animal or cohort for the response model.
3. Prespecified amplitude calibration from the experimental device, not
   standardised model units.
4. A negative control that is physically matched (e.g. off-target illumination)
   and a random-neuron control in the same animals.
5. Independent behavioural measurement (video), not labels decoded from the same
   neural population.
