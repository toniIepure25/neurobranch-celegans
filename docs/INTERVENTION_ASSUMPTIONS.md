# Intervention Assumptions and Causal Status

> **Standing label for every simulation output:**
> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

This document states precisely what the intervention engine does, what it
assumes, and why none of its outputs is a causal effect in a living animal.

---

## 1. What the engine computes

Given a fitted model `P_θ(x_{t+1} | x_{t-h+1:t})`, an initial state `x_t` sampled
from held-out data, and an edit `e`, the engine computes two model rollouts with
common random numbers:

```
baseline:      x_{s+1} ~ P_θ(· | x_{s-h+1:s})
intervention:  x'_{s+1} ~ P_θ(· | e(x'_{s-h+1:s}))
```

It then decodes macro behaviour from each state and reports:

- `P_baseline(b)`, `P_intervention(b)` over the horizon;
- `ΔP(b) = P_intervention(b) − P_baseline(b)` for the target behaviour;
- a cluster-bootstrap interval over initial frames;
- variability over simulation seeds;
- mean trajectory displacement;
- fraction of simulated states outside the model's training support.

The edit is applied to the **model state**, not to a neuron, not to a worm, and
not to a population of worms.

---

## 2. Supported edits

| Mode | Implementation | Intended reading |
|---|---|---|
| `additive` | `x_i ← x_i + a` during `[onset, onset+duration)` | shift a unit's standardized activity |
| `clamp` | `x_i ← a` | hold a unit at a fixed level |
| `zero` | `x_i ← 0` | remove a unit's deviation from the training mean |
| `scale` | `x_i ← a · x_i` | multiply a unit's deviation |

Amplitudes are in **standardized SD units** of the training set. `+2` means two
training standard deviations above the mean, not "two spikes" and not an
optogenetic power.

`onset` and `duration` are in frames at ≈2.9 Hz (1 frame ≈ 0.344 s). A 12-frame
horizon is ≈4.1 s, comparable to a reversal-to-turn sequence but not matched to a
specific behavioural epoch.

---

## 3. What the edit is not

| Not this | Why |
|---|---|
| `do(X_i = x)` in the animal | We have no intervention on the animal; the model is observational. |
| Optogenetic activation/silencing | Light power, opsin expression, cell specificity, and network compensation are absent. |
| A lesion | Zeroing a standardized trace is not ablating a cell. |
| A genetic perturbation | No genotype is modelled. |
| A pharmacological manipulation | No drug, dose, or time course is modelled. |
| A prediction of a real experiment's effect size | The mapping from model units to biological units is unknown. |

A `do()` operator requires that the intervened variable be set independently of
its parents. In a dynamical model, editing `x_i` at time `t` and then letting the
model evolve is a **soft intervention on the model's state**, which is a valid
mathematical object. It is not the same object as an experimental intervention on
the animal, because the model's causal graph is assumed, not identified.

---

## 4. Assumptions required for a causal reading

For the output to be interpretable as a causal effect in the animal, **all** of
the following would need to hold. None is established here.

1. **Causal sufficiency.** The predictor set contains all common causes of the
   target neuron and behaviour. With 31–47 named cells out of 302, and no
   neuromodulatory, mechanical, or environmental variables, this is false.
2. **Correct functional form.** The model class contains the true transition
   kernel. The one-step R² results (no model beats persistence) argue against
   this for the neural state.
3. **Stationarity / transportability.** The fitted relation holds under the
   intervention and in the target context. The strict-forward failure shows it
   does not even hold across time in the same worm.
4. **Support overlap.** The intervened state is in the training distribution. The
   OOD fractions of 8–38% show it often is not.
5. **No unmeasured confounding.** Behaviour and neural activity are both driven
   by unrecorded state (arousal, oxygen, muscle state, stimulus history).
6. **Measurement model.** ΔF/F is a noisy, nonlinear, slow indicator of calcium,
   not a direct measure of spike rate or membrane potential.
7. **Label validity.** The behaviour labels are derived from a subset of the same
   neurons, so the outcome is not independent of the intervention target.
8. **Intervention fidelity.** The model edit corresponds to a physically
   realizable perturbation with a known dose–response mapping. It does not.

A causal claim would require at minimum an actual randomized intervention with
matched controls, plus evidence that the model's predicted effect generalizes to
that intervention. We have neither.

---

## 5. Evidence gate

`interventions/safety.py` defines `EmpiricalEvidence` with required fields:

- dataset path and SHA256;
- intervention description and target neurons;
- number of intervention and control animals;
- provenance URL;
- whether it was performed in a living animal;
- a negative control;
- pre-registration.

`require_empirical_validation(path)` returns a status object. It validates only
if every field is present, the file hash matches, the dataset exists, both animal
counts are positive, and the living-animal, negative-control, and pre-registration
flags are true. Otherwise it returns `empirical_rejected` with explicit reasons.

This gate is intentionally not satisfiable by the Kato observational dataset. It
exists so that a future user cannot accidentally inherit a "validated" label.

---

## 6. Controls and uncertainty

### 6.1 Paired rollouts

Baseline and intervention use the same initial state and the same random seed
(common random numbers). This removes simulation noise from the difference and
makes small effects detectable *within the model*. It does **not** remove model
error.

### 6.2 Initial-state bootstrap

The 95% interval resamples held-out initial frames with replacement, preserving
temporal clustering at the frame level. Adjacent frames are still correlated, so
the interval is optimistic; a block bootstrap over time would be wider.

### 6.3 Simulation seeds

Three seeds (0,1,2) are used for stochastic models. The reported distribution
averages over them; seed-to-seed variability is small relative to initial-state
variability in the reported scan.

### 6.4 Random-neuron controls

The correct null is **not** "no perturbation". It is "a perturbation of the same
magnitude delivered to an arbitrary cell". Controls are sampled uniformly without
replacement from named predictor cells that are **not** in the hypothesis-driven
candidate set, with identical amplitude, timing, horizon, initial states, and
seeds.

An earlier implementation sampled from the full pool including candidates. That
made the "control" overlap the candidates and invalidated the specificity
comparison. It was caught during red-teaming, fixed in
`interventions/search.py`, and the controls were re-run. Only the corrected
distribution is reported.

Corrected result: RIVL +2 SD gives ΔP(TURN)=+0.333, while 4/19 non-candidate
cells give |Δ| ≥ 0.333 and the maximum is +0.667 (OLQVL). Empirical one-sided
p ≈ 0.21. The candidate effect is **not specific**.

### 6.5 Out-of-support flag

Neural models provide an `ood_score`; the threshold is the 99th percentile of
training scores. Latent kNN uses distance to the k-th training neighbour. The
reported OOD fraction is the proportion of simulated states above threshold.

A large effect at high OOD is an extrapolation artifact. The largest control
effect (OLQVL) has OOD 0.377; the top candidate RIVL has OOD 0.106. Neither is
"safe", but the control makes the extrapolation visible.

---

## 7. How to read a result

**Correct:**

> "In the fitted MLP, a +2 SD virtual edit to RIVL increased the model's predicted
> TURN probability by 0.333 (95% frame-bootstrap CI 0.167–0.533), with 10.6% of
> simulated states outside training support. This effect is not larger than those
> produced by arbitrary non-candidate cells (empirical p≈0.21) and is therefore
> not evidence of a specific biological role."

**Incorrect:**

> "RIVL drives turning."
> "We identified a causal switch for reversal termination."
> "The model predicts that silencing RIVL will increase turning in vivo."
> "The 3-D trajectory bifurcates at RIVL."

The second set smuggles in causal identification, biological dose, and mechanism
that the analysis does not provide.

---

## 8. What a real validation would require

1. A preregistered, randomized intervention on the target cells in living animals.
2. Matched sham and random-neuron controls, ideally within-animal.
3. The same behavioural assay and state annotation as the observational data.
4. Enough animals for a powered effect estimate, not a single worm.
5. A prespecified mapping from model units to intervention dose.
6. Independent replication in a second lab or preparation.
7. A model prediction registered before the experiment, so the test is
   confirmatory rather than post hoc.

See [`NEXT_EXPERIMENTS.md`](NEXT_EXPERIMENTS.md) for the concrete proposal.
