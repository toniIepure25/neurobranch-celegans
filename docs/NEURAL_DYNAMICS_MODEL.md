# Neural dynamics model

This document describes the model that turns a held-out initial neural state into
a free-running future. It is the **core** of the research branch; decoding,
AUROC, SAAV and representation benchmarks are supporting evaluations only.

> **Status.** Functioning simulator: **yes**. Demonstrated dynamical advantage
> over persistence or a constant-state baseline: **no** on the frozen Gate 2
> rule. Causally validated: **no**. The output is a model prediction, not an
> experimental intervention.

## 1. Scope and equation

For each worm independently, the fitted model is a discrete-time map

```text
x(t+1) = F_theta(x(t), x(t-1), ..., x(t-h+1), u(t)) + epsilon(t)
```

where

- `x(t) in R^N` is the standardised calcium state of the `N` predictor traces;
- `h` is the model history length (usually 1);
- `u(t)` is an optional **model-state input** supplied by a virtual
  perturbation (normally zero);
- `epsilon(t)` is rollout noise with standard deviation `noise_std`, injected in
  standardised units;
- `F_theta` is fitted on the training blocks only.

The model is **not** a biophysical simulation and does not integrate membrane
potentials, synaptic conductances, gap junctions, or muscles. It is a
phenomenological population map fitted to calcium observations.

### One-step prediction versus free-running rollout

A one-step prediction receives the observed `x(t)` and predicts `x(t+1)`. A
free-running rollout receives only `x(0)` and its own previous predictions. The
simulator API has no argument for future observations:

```python
from neurobranch.simulator import NeuralSimulator

rollout = sim.rollout(x0, steps=20, seed=17, noise_std=0.05)
```

`rollout.states` has shape `(steps + 1, N)` and `rollout.behavior_proba` has shape
`(steps, 4)`. `x0` is preserved exactly at index 0. The tests verify that
mutating future observations does not change a rollout.

## 2. Data and leakage boundary

- Dataset: Kato et al. (2015) whole-brain calcium imaging, five wild-type,
  unstimulated, immobilised worms.
- Predictor traces: 103-127 per worm, depending on the worm; 31-47 named.
- Eight behaviour-label neurons (`AVAL`, `AVAR`, `SMDDR`, `SMDDL`, `SMDVR`,
  `SMDVL`, `RIBR`, `RIBL`) are excluded from predictors to reduce circularity.
- Split: contiguous blocked split, five blocks, test block 2, validation block 4,
  purge 12 frames. Windows never cross split boundaries.
- Standardisation statistics and PCA/BunDLe encoders are fitted on training
  frames only. A held-out frame is never used to fit a scaler, encoder, support
  threshold, or model-selection score.
- This is an **interpolation** protocol, not a forward forecast. The existing
  `forward_stress` suite is the chronological stress test and is not used to
  select this simulator.

The Kato file is observational. No neuron was stimulated, ablated, clamped, or
optogenetically manipulated. Therefore the fitted map cannot identify a causal
effect of changing a named neuron.

## 3. Model families reused from the frozen Phase 2 suite

| Kind | Map | Notes |
|---|---|---|
| `persistence` | `x(t+1)=x(t)` | One-step and free-running null. |
| `var` | linear VAR with ridge | Best descriptive free-running baseline in this audit. |
| `mlp` | feed-forward MLP | Direct next-state prediction. |
| `mlp_residual` | MLP predicts a residual | Highest validation one-step RÂ²; selected by the frozen Gate 2 selection rule. |
| `gru` | recurrent GRU | History-dependent; no explicit latent PCA. |
| `latent_mlp` | PCA(16) + latent MLP + linear decode | Latent dynamics, invertible PCA decoder. |
| `latent_knn` | PCA(16) + kNN transition + decode | Native stochastic kNN transitions. |

Training uses the frozen config
`configs/experiments/blocked_suite.yaml`: 60 epochs, batch 256, Adam at 1e-3,
weight decay 1e-4, patience 15, behaviour weight 0.2. Existing Phase 2
checkpoints are reused; `scripts/simulator_fit.py --fit-missing` only fills
absent checkpoints and never overwrites one.

## 4. Selection and why it is not enough

`scripts/simulator_fit.py` writes one model card per checkpoint under
`reports/simulator/model_cards/`. Selection uses **validation block 4** only:

1. highest mean validation one-step RÂ²;
2. tie-break lower validation 5-step MSE;
3. tie-break simpler model.

The selected kind is **`mlp_residual`** (mean validation RÂ² `0.7134`). Its
free-running test behaviour is much worse than its one-step fit:

| Kind | Mean test free-running RÂ² | Mean MSE âˆ’ constant-state baseline | Mean OOD fraction |
|---|---:|---:|---:|
| `persistence` | 0.4368 | 0.0000 | 0.1600 |
| `var` | **0.5325** | **âˆ’0.0491** | **0.0076** |
| `mlp` | 0.3078 | 0.1639 | 0.0694 |
| `mlp_residual` (selected) | **âˆ’0.5098** | **0.8398** | **0.7903** |
| `gru` | 0.3726 | 0.0925 | 0.0456 |
| `latent_mlp` | 0.1420 | 0.2956 | 0.2854 |
| `latent_knn` | 0.2283 | 0.2368 | 0.3032 |

The constant-state baseline is the initial state held for the whole horizon. It
is the correct free-running null; one-step persistence is a different baseline
and is reported separately in `one_step_metrics.csv`.

**Interpretation.** A high one-step validation RÂ² can coexist with unstable
free-running dynamics. The selected model is functionally usable (finite
rollouts, correct shape and provenance), but it does not beat the null on the
frozen dynamical-advantage rule in any of five worms. `var` is the best
descriptive free-running model here, but it was not selected by the one-step
rule and is not promoted to a causal simulator.

## 5. History and support diagnostics

The support model is a 99th-percentile leave-one-out training-distance envelope
over standardised states. It is descriptive, not a probabilistic guarantee.
High OOD fractions for `mlp_residual` (about 0.79 on test rollouts) mean its
trajectories leave the region where the fit has empirical support. Perturbation
experiments push OOD fractions to about 0.98-0.99, so apparent perturbation
effects are largely extrapolation.

## 6. Interface contract

Located in `src/neurobranch/simulator/`.

```python
from neurobranch.simulator import NeuralSimulator, PerturbationSpec

sim = NeuralSimulator.from_checkpoint(".../model.pt", data)

base = sim.rollout(x0, steps=20, seed=17, noise_std=0.0)
ens = sim.rollout_ensemble(x0, n=30, steps=20, base_seed=17, noise_std=0.05)

spec = PerturbationSpec(
    target="AVEL",
    protocol="dynamical_input",  # or "initial_state"
    mode="additive",
    amplitude=-1.0,
    onset=0,
    duration=10,
)
pert = sim.rollout_ensemble(x0, 30, 20, base_seed=17, noise_std=0.05, perturbation=spec)
```

Semantics:

- `initial_state`: applies the edit to `x0` only. It is a mathematical
  sensitivity probe, not stimulation.
- `dynamical_input`: applies the edit after each predicted state during
  `[onset, onset + duration)`.
- `empirical_response`: refused unless a matched, validated empirical response
  model is registered. None is registered for Kato, so `PermissionError` is the
  correct result. Gate 6 is `not_evaluable`.
- Perturbations are in training-standard-deviation units. They are not
  optogenetic power, light intensity, histamine concentration, or any physical
  dose.

## 7. Reproduction

```powershell
python scripts/simulator_fit.py
python scripts/simulator_evaluate.py
python scripts/simulator_gates.py
python scripts/validate_reports.py --reports reports/simulator --strict
```

Outputs: `reports/simulator/models.csv`, `model_cards/`, `selection.json`,
`rollout_metrics.csv`, `one_step_metrics.csv`, `selected_model_test.csv`,
`summary.json`, `gates.json`, `manifest.json`.

## 8. What would count as evidence for the model

A dynamical advantage would require, on a held-out worm or cohort, all of:

1. lower free-running state error than constant-state and one-step persistence
   baselines over prespecified horizons;
2. the advantage surviving block-bootstrap uncertainty and not depending on a
   single worm;
3. trajectories staying inside a prespecified support region;
4. calibration of the predicted outcome distribution, not only a point forecast;
5. an independent cohort or stimulation dataset for external validation.

None of 1-5 is established here. Gate 2 fails in 0/5 worms.

