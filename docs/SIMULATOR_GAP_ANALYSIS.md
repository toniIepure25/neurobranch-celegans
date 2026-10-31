# Simulator gap analysis

**Status:** current-state audit, 2026-10-31.
**Branch:** `research/core-dynamics-simulator`
**Repository:** `D:\ComputaCenter\NeuroBranch\neurobranch-celegans`

This is an implementation audit, not a literature review. It records what the
repository can already do, what it cannot, and which existing work must be
preserved.

---

## 1. Executive summary

The repository already contains more simulator machinery than the earlier
phase narrative suggests:

- fitted persistence, VAR, MLP, GRU, PCA-latent-MLP and PCA-latent-kNN models;
- a `rollout` method on every dynamics model;
- checkpoint loading and metadata;
- a paired counterfactual engine with common random numbers;
- named-neuron state edits, random controls and selection-adjusted tests;
- empirical branching tests with block and circular-shift nulls;
- a BunDLe-Net reimplementation and official embedding inspection;
- a 14-tab Streamlit/Plotly application.

What is missing is a **single simulator contract**: one API that owns the
initial state, the free-running rollout, the latent projection, the
perturbation semantics, the evaluation suite, the manifests and the UI. The
current pieces are distributed across `dynamics/`, `interventions/`,
`branching/`, `representation/` and `twin/`, and they use different assumptions.

The largest scientific gaps are not code gaps:

- no matched Kato intervention data, so biological validation is unavailable;
- behaviour labels are inferred from neural activity;
- the observed branching audit does not support excess branching;
- existing one-step models generally do not beat persistence;
- five worms do not identify a population distribution.

---

## 2. Reuse classification

### CORE — directly supports simulation and perturbation

| Component | What it provides | Reuse decision |
|---|---|---|
| `data/kato.py` | Five-worm loader, provenance, macro states, behaviour-label exclusion | Use unchanged |
| `preprocessing/__init__.py` | `WormData`, train-only scaling, split masks, `split_starts` | Use unchanged |
| `dynamics/models.py` | Persistence, VAR, MLP, GRU, PCA-latent-MLP, PCA-latent-kNN, rollouts | Wrap behind simulator API |
| `dynamics/checkpointing.py` | Model persistence and metadata | Use unchanged; add model cards |
| `dynamics/training.py` | Suite fitting and worm loading | Reuse for simulator fits |
| `interventions/spec.py` | Named-neuron state edits | Reuse for dynamical-input perturbations |
| `interventions/engine.py` | Paired baseline/intervention rollouts, bootstrap, OOD | Reuse and split semantics |
| `interventions/safety.py` | Empirical evidence gate and warning | Use unchanged |
| `branching/__init__.py` | Local outcome diversity and corrected branching tests | Reuse as observed reference |
| `branching/stats.py` | iid/block/shift nulls | Reuse |
| `representation/bundlenet.py` | BunDLeLite, causal low-pass, official embedding | Reuse architecture; add train-only encoder wrapper |
| `app/streamlit_app.py` | Data/model cache and Plotly shell | Extend, do not rewrite |

### SUPPORTING — useful for evaluation

| Component | Use |
|---|---|
| `evaluation/behavior_benchmark.py` | Behaviour metrics and context; not a gate |
| `branching/synthetic.py` | Ground-truth synthetic branching scenarios |
| `config.py` | `ExperimentConfig`, `git_state`, hashing |
| `reporting/status.py` | Status-report pattern |
| `scripts/validate_reports.py` | Schema/table validation pattern |
| `schemas/` and `sync_schemas.py` | Packaged schema convention |

### SECONDARY — preserve, do not extend

- Phase 3–5 decoding, SAAV, Kramer turn-direction, Randi optogenetic benchmark.
- Phase 6 CRO, Phase 7 active probing, Phase 8 adaptive twin.
- Phase 9 BoundedBench and Phase 10–15 reliability/measurement audits.
- These remain tested and documented. They are not the simulator success
  criterion and must not be deleted or silently invalidated.

### BLOCKED — missing evidence or invalid assumptions

| Item | Why blocked | Handling |
|---|---|---|
| Biological validation (Gate 6) | No matched Kato stimulation with sham controls | `not_evaluable` |
| Independent behaviour ground truth | Kato labels derived from AVA/SMD/RIB | Label all behaviour results conditional |
| Whole-brain simulation | 109–135 traces, 31–47 named, 302 neurons | Call it an observed subset |
| Full connectome-constrained model | Connectome covers named cells only | Optional named-subset extension only |
| Freely moving transport | Different preparation and observation model | Do not claim |
| BunDLe-Net as complete simulator | No published inverse decoder; representation model | Use as encoder, not as neuron-level simulator |
| Randi → Kato stimulation transfer | Different animals, protocol and measurement | Do not substitute for Gate 6 |

---

## 3. Capability matrix

| Capability | Current state | Gap | Target |
|---|---|---|---|
| Load real initial state | Yes: `WormData.X[t]` | No simulator-owned selection API | `rollout_from_data` |
| Free-running rollout | Yes: model `.rollout(x0, steps)` | Scattered contracts; some add noise in neuron space without documentation | Unified `Rollout` object |
| Multiple stochastic futures | Partial: latent kNN and Gaussian noise | No shared ensemble/seed contract | `rollout_ensemble` |
| Save complete trajectories | Partial: ad hoc `.npz` | No manifest or schema | `reports/simulator/demo/*.npz` |
| One-step vs free-running evaluation | Partial: suite metrics | No common multi-step/distribution suite | `simulator/metrics.py` |
| Latent projection | Yes: PCA, BunDLeLite | No real-vs-simulated validation or stability test | `simulator/latent.py` |
| Latent-space dynamics | Yes: PCA latent MLP/kNN | Not compared to neuron-space simulation; BunDLe has no decoder | Documented comparison |
| Branching statistics | Yes: observed local entropy and nulls | No simulator ensemble vs observed comparison | `simulator/branching.py` |
| Named-neuron perturbation | Yes: additive/clamp/zero/scale state edits | Semantics not separated into initial-state / dynamical / empirical | `PerturbationProtocol` |
| Paired baseline/perturbed | Yes: `run_counterfactual` | Only state-input semantics; no initial-state protocol | Extend |
| Control neurons | Yes: random and candidate controls | Not integrated into simulator report | Reuse and extend |
| OOD diagnostics | Partial: kNN distance in engine | Not on every rollout/ensemble | `support_score` |
| Named-neuron availability | Yes: `WormData.neuron_index` | No per-worm validation table | Model card and report |
| Simulator UI | Existing virtual-perturbation tab | Not simulator-centred; no free-running ensemble | New tab |
| Export manifest | Partial: ad hoc JSON | No unified schema | `SimulationManifest` |
| Tests | 241 existing; some counterfactual/branching tests | No simulator contract, gate logic or UI tests | New suite |

---

## 4. Existing evidence that constrains the new work

1. **Persistence is strong.** In the blocked suite, mean one-step \(R^2\) is
   `0.3613` for persistence; VAR is `0.3655`, MLP-residual `0.3723`, and the
   other models are lower. The mean paired delta for MLP-residual is `+0.0110`
   and it wins 7/15 worm–seed pairs. A simulator that cannot beat persistence is
   still a simulator, but it is not evidence of learned dynamical advantage.
2. **Forward stress is harsher.** Under the strict chronological split,
   persistence \(R^2\) is `0.3984`; MLP-residual is `0.4071`; VAR, GRU, latent
   MLP and latent kNN are negative. The blocked result must not be presented as
   forecasting.
3. **Branching is not established.** The corrected audit found no significant
   excess local entropy in any worm. Four of five worms showed a deficit under
   the iid and block-10 nulls. The observed result is projection- and
   worm-dependent.
4. **RIVL is not specific.** Phase 2 found random control neurons with effects
   as large or larger; no model reached `p < 0.05`. The residual MLP produced
   zero effect and 99.4% OOD states. This is an off-support stress test, not a
   dose–response curve.
5. **The Kato intervention is absent.** No optogenetic, thermal, mechanical or
   pharmacological perturbation is present in `WT_NoStim.mat`.

---

## 5. Implementation implications

- Do not build a new model zoo. Wrap the existing models.
- Do not re-run a hyperparameter search. Use fixed, disclosed configurations.
- Do not describe teacher-forced one-step prediction as simulation.
- Do not merge initial-state sensitivity, dynamical input and experimentally
  constrained response into one “causal” object.
- Do not force the simulator to produce branches. If it cannot, report that.
- Do not use the Randi dataset as if it validated Kato interventions.
- Do not call the 3-D latent trajectory a state space with biological
  coordinates.
- Add a simulator-owned API, evaluation suite, report, UI and tests around the
  existing components.

---

## 6. Files expected to change

New:

```text
src/neurobranch/simulator/{__init__,types,adapters,rollout,train,metrics,latent,branching,perturbation}.py
scripts/simulator_{fit,evaluate,branching,perturb,manifest,validate,report}.py
configs/simulator/core.yaml
schemas/simulator_summary.schema.json
src/neurobranch/schemas/simulator_summary.schema.json
app/simulator_ui.py
tests/simulator/*.py
tests/app/test_simulator_ui.py
docs/ORIGINAL_RESEARCH_OBJECTIVE.md
docs/SIMULATOR_GAP_ANALYSIS.md
docs/SIMULATOR_RESEARCH_PROTOCOL.md
docs/NEURAL_DYNAMICS_MODEL.md
docs/LATENT_SIMULATION_VALIDATION.md
docs/BRANCHING_SIMULATION.md
docs/NEURONAL_PERTURBATION_METHODS.md
docs/SIMULATOR_RESULTS.md
docs/SIMULATOR_LIMITATIONS.md
docs/SIMULATOR_SUPERVISOR_BRIEF.md
docs/QUESTIONS_FOR_DENIS.md
research/simulator_protocol.md
reports/simulator/*
```

Modified:

```text
README.md
docs/CURRENT_STATUS.md
docs/EXPERIMENT_REGISTRY.md
docs/LIMITATIONS.md
docs/CLAIMS.md
docs/README.md
mkdocs.yml
CHANGELOG.md
app/streamlit_app.py
tests/integration/test_app_phase3.py
scripts/sync_schemas.py
scripts/validate_reports.py
```

Existing Phase 1–15 reports and history remain untouched.
