# NeuroBranch

**A research-grade, end-to-end pipeline for *C. elegans* whole-brain population dynamics, latent trajectory branching, and model-dependent virtual perturbation — built on real Kato et al. (2015) calcium imaging.**

> **EPISTEMIC STATUS.** Everything produced by the intervention engine is a
> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.** The models
> are fitted to observational recordings only. No perturbation of a living animal
> was performed for this project. A visually attractive 3-D branch is not a
> bifurcation, and good observational prediction is not causal identification.
> The independent empirical evidence gate in `interventions/safety.py` refuses to
> label any output as empirically validated without a matched, provenance-checked
> intervention dataset.

---

## What is in this repository

| Layer | Status |
|---|---|
| Real-data audit (Kato 2015, 5 worms, WT_NoStim) | ✅ completed on the downloaded archive |
| Leakage-safe blocked and strict-forward splits | ✅ implemented, tested |
| Baselines: persistence, VAR, MLP, residual MLP, GRU | ✅ 105 runs (5 worms × 3 seeds × 7 models) |
| Latent models: PCA+MLP, PCA+kNN | ✅ same 105-run suite |
| BunDLe-Net | ✅ official 3-D artifacts evaluated; PyTorch reimplementation trained (partial reproduction) |
| NC-MCM | ✅ distributed artifacts verified; lag-2 Markov test ported to Python; behaviour decoder cross-validated |
| Branching analysis | ✅ empirical + model-based; **corrected two-sided null test** (Phase 2) |
| Virtual perturbation engine | ✅ paired baselines, common random numbers, multi-seed uncertainty, OOD support flags, random-neuron controls |
| Sparse candidate search | ✅ 13 named cells × 2 amplitudes + 19 non-candidate controls; **Phase 2 specificity audit** |
| Randi 2023 optogenetic atlas | ✅ 110 animals, 3,371 events parsed; Phase 3 target/shape/context audit (**negative**) |
| Independent freely-moving behaviour | ✅ Baskoylu & Flavell 2026, 27 animals, 1,162 reversals, SHA256-verified |
| Prospective behaviour benchmark | ✅ animal-disjoint 3/6/12 s; primary neural increment **inconclusive** |
| Frozen neural representation | ✅ population, PCA, identified-neuron, quadratic and causal-delay controls; **not robust** |
| Interactive 3-D app | ✅ Streamlit + Plotly, 8 tabs (7 Phase 2 tabs preserved) |
| CLI | ✅ Typer, documented commands |
| Tests | ✅ 58 passed (unit, integration, scientific; includes Phase 2 + Phase 3) |
| Figures | ✅ 8 static PNGs + 6 Phase 3 figures + interactive HTML |
| Causal validation | ❌ not attempted; evidence gate refuses unsupported claims |

The headline result is deliberately unflattering: on held-out data, **no neural
dynamics model clearly beats a persistence baseline**, behaviour decoding is far
below label persistence and adds no incremental calibration value, apparent
branching is actually *lower* local entropy than a geometry-preserving null in
4/5 worms (predictability, not branching), and the top named-neuron virtual effect
is **within the range of arbitrary non-candidate cells** (4/19 controls match or
exceed it, empirical p≈0.21), so it is not specific. The external Randi 2023
optogenetic benchmark also fails to beat a constant-mean baseline on held-out
animals. These are findings, not bugs; they are preserved in
[`docs/RESULTS.md`](docs/RESULTS.md), [`docs/PHASE2_SCIENTIFIC_LIMITATIONS.md`](docs/PHASE2_SCIENTIFIC_LIMITATIONS.md)
and [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

---

## Quick start

The project targets Python ≥3.10,<3.14. This checkout was validated with Python
3.13.5, PyTorch 2.7.1, NumPy 2.3.1, SciPy 1.16 and scikit-learn 1.9.1.

```bash
# If uv is on PATH:
uv sync
uv run neurobranch version
uv run neurobranch data audit
uv run neurobranch app

# If uv is installed as a Python module but not on PATH:
python -m uv sync
python -m uv run neurobranch version
```

A plain editable install also works:

```bash
python -m pip install -e .
neurobranch version          # may need the user Scripts directory on PATH
python -m neurobranch.cli.main version
```

The raw dataset is **not committed** (see `.gitignore` and
[`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md)). Place
`WT_NoStim.mat` at `data/raw/kato2015/WT_NoStim.mat`, or re-download from OSF
project `2395t` and verify the SHA256.

---

## Commands

```bash
neurobranch data audit
neurobranch data list
neurobranch train dynamics --config configs/experiments/blocked_suite.yaml
neurobranch train dynamics --config configs/experiments/forward_stress.yaml --seeds 17
neurobranch evaluate --run-dir artifacts/runs/blocked_suite
neurobranch branching analyze --worm 0
neurobranch reproduce ncmcm --k 3
neurobranch reproduce bundlenet --worm 0 --epochs 120
neurobranch simulate baseline --worm 0 --model mlp --seed 17
neurobranch simulate intervention --worm 0 --model mlp --seed 17 \
  --spec-file configs/interventions/rivl_plus2.yaml
neurobranch report
neurobranch app
```

Phase 2 audits (require local raw data; outputs are committed under `reports/`):

```bash
python scripts/phase2a_branching_audit.py
python scripts/phase2a_intervention_audit.py
python scripts/phase2a_optogenetic_benchmark.py
python scripts/phase2a_behavior_benchmark.py
```

Phase 3 (Baskoylu 2026 freely-moving data and Randi response audit):

```bash
python scripts/phase3_download_baskoylu.py          # Anubis challenge + SHA256
python scripts/phase3_prospective_benchmark.py     # 27-animal LOAO benchmark
python scripts/phase3_randi_analysis.py            # target/context/shape audit
python scripts/phase3_make_figures.py              # reports/phase3/figures
python scripts/export_neuropal.jl                  # one-time JLD2 -> TSV (Julia)
```

The Phase 3 headline is a **negative/inconclusive** primary result: adding
population neural activity to behavioural history gives Δ log loss -0.0070 at 6 s,
95% animal CI [-0.0148, +0.0005]. A causal delayed-feature control also improves
slightly (Δ -0.0035), so it is not a clean null; a quadratic
behavioural baseline removes the significance, identified neurons do not help,
and within-animal forward prediction is worse. Randi repeated responses remain at
r≈0.09 and repeated-event averaging is worse than a global mean. See
[`docs/PHASE3_RESULTS.md`](docs/PHASE3_RESULTS.md).

The `simulate intervention` command prints the standing warning and writes a
JSON summary containing the baseline and virtual distributions, bootstrap CI,
trajectory displacement, OOD fraction, and the causal-status object.

---

## Repository layout

```text
neurobranch-celegans/
├─ app/streamlit_app.py          interactive explorer
├─ configs/                      data, experiment, and intervention configs
├─ data/raw/kato2015/            immutable raw archive (not committed)
├─ data/raw/baskoylu2026/        Phase 3 raw HDF5/JLD2 (not committed)
├─ docs/                         methodology, provenance, results, limitations,
│                                supervisor brief, paper prospectus, Phase 3 reports
├─ figures/                      generated PNG/HTML figures
├─ reports/                      audit tables and result summaries
├─ references/external/          cloned upstream repositories (not committed)
├─ src/neurobranch/
│  ├─ behavior/                  runs, transitions, entropies
│  ├─ branching/                 local outcome diversity, corrected null tests, synthetic fixtures
│  ├─ cli/                       Typer entry point
│  ├─ data/                      Kato loader, Baskoylu/Randi adapters, provenance
│  ├─ dynamics/                  baselines, neural/latent models, training, cross-animal probe
│  ├─ evaluation/                metrics, bootstrap, rollouts, Randi/behaviour benchmarks
│  ├─ interventions/             specs, paired engine, OOD, evidence gate, specificity audit
│  ├─ preprocessing/             leakage-safe split/scaling/windowing
│  ├─ representation/            BunDLe-Net artifact integration + reimplementation
│  ├─ reproduction/              NC-MCM verification
│  └─ reporting/                 tables and figures from saved artifacts
├─ scripts/                      Phase 2/3 audit and benchmark entry points
└─ tests/                        unit / integration / scientific
```

---

## Core scientific choices

1. **Behaviour labels are not independent ground truth.** Kato's eight states are
   inferred from calcium signals in AVA, SMD and RIB. Those eight neurons are
   excluded from predictors (`BEHAVIOR_LABEL_NEURONS`), but residual circularity
   remains because the labels and the predictors share the same recording.
2. **Two evaluation regimes are reported.** The primary *blocked interpolation*
   benchmark holds out a contiguous middle block and trains on earlier **and
   later** blocks, with no window crossing a boundary. The *strict forward* test
   trains on the first 70% and tests on the final 15%, exposing severe
   non-stationarity and late-recording behavioural degeneracy.
3. **Persistence is a first-class model.** It is not a straw man; it wins.
4. **Interventions are edits of model state, not `do()` on the animal.** The
   engine implements additive/clamp/zero/scale edits, paired baseline rollouts,
   common random numbers, seed and initial-state uncertainty, an out-of-support
   flag, and random-neuron controls. It returns an explicit non-causal status.
5. **Reference artifacts are not our results.** The official BunDLe-Net
   embeddings were fitted upstream on all windows, so even a kNN classifier
   trained on train labels is scored on test embeddings that the encoder already
   saw. This is **embedding-level leakage**, not held-out generalization. Our own
   reimplementation is evaluated on a blocked holdout and performs substantially
   worse. Both facts are reported.

---

## Data and reproducibility

- Dataset: Kato et al. 2015, *Cell* 163:656–669, OSF project
  <https://osf.io/2395t/>. File `WT_NoStim.mat`, SHA256
  `7531811c1718e26a0455f59cb2f8188cf25809278cfea85770dfe03315a0098a`.
- Phase 3 dataset: Baskoylu & Flavell 2026, Dryad
  `doi:10.5061/dryad.w9ghx3g4v`, 27 animals, 30 files, 320,032,093 bytes, CC0-1.0.
  Raw data are untracked; the SHA256 manifest and derived tables are committed.
- Raw data are immutable; every derived artifact records the SHA256 and the git
  state in its `resolved_config.yaml` / metrics JSON.
- Seeds 17, 23, 31 are fixed for the primary suite. Every training run writes a
  checkpoint, train history, test predictions, and metrics.
- Environment manifest: `reports/environment.json` (assembly environment, written
  after the runs; it is not a training-time snapshot).

See [`docs/EXPERIMENT_REGISTRY.md`](docs/EXPERIMENT_REGISTRY.md) for every run,
config, and artifact path.

---

## Documentation

| Document | Purpose |
|---|---|
| [`docs/SUPERVISOR_BRIEF.md`](docs/SUPERVISOR_BRIEF.md) | Two-page status and discussion points for Denis Kovačević |
| [`docs/POTENTIAL_PAPER.md`](docs/POTENTIAL_PAPER.md) | Paper framing, claims, figures, and what is still missing |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Exact quantitative results and negative findings |
| [`docs/PHASE2_BASELINE_AUDIT.md`](docs/PHASE2_BASELINE_AUDIT.md) | Pre-Phase-2 baseline, inherited state, gates |
| [`docs/BRANCHING_STATISTICAL_AUDIT.md`](docs/BRANCHING_STATISTICAL_AUDIT.md) | Corrected branching nulls and synthetic power checks |
| [`docs/BUNDLENET_REPRODUCTION_AUDIT.md`](docs/BUNDLENET_REPRODUCTION_AUDIT.md) | Transductive artifact vs causal partial reimplementation |
| [`docs/NEURON_IDENTITY_MAPPING.md`](docs/NEURON_IDENTITY_MAPPING.md) | Kato↔Randi canonical names and ambiguity |
| [`docs/INTERVENTION_DATASET_AUDIT.md`](docs/INTERVENTION_DATASET_AUDIT.md) | Randi 2023 provenance, schema, coverage |
| [`docs/INTERVENTION_GENERALIZATION_RESULTS.md`](docs/INTERVENTION_GENERALIZATION_RESULTS.md) | Specificity audit + external optogenetic benchmark |
| [`docs/BEHAVIORAL_GENERALIZATION.md`](docs/BEHAVIORAL_GENERALIZATION.md) | Incremental behaviour prediction (negative) |
| [`docs/RELATED_WORK_NEBULA.md`](docs/RELATED_WORK_NEBULA.md) | NEBULA positioning and evaluation gates |
| [`docs/PHASE2_METHODOLOGY.md`](docs/PHASE2_METHODOLOGY.md) | Phase 2 protocols, splits, statistics |
| [`docs/PHASE2_SCIENTIFIC_LIMITATIONS.md`](docs/PHASE2_SCIENTIFIC_LIMITATIONS.md) | Phase 2-specific limitations |
| [`docs/PHASE2_NEXT_EXPERIMENTS.md`](docs/PHASE2_NEXT_EXPERIMENTS.md) | Falsifiable next experiments |
| [`docs/PHASE3_RESEARCH_PLAN.md`](docs/PHASE3_RESEARCH_PLAN.md) | Locked Phase 3 preregistration and decision rule |
| [`docs/PHASE3_DATASET_AUDIT.md`](docs/PHASE3_DATASET_AUDIT.md) | Baskoylu HDF5/JLD2 audit, event convention, leakage finding |
| [`docs/PHASE3_BEHAVIORAL_VALIDATION.md`](docs/PHASE3_BEHAVIORAL_VALIDATION.md) | Prospective behaviour benchmark and controls |
| [`docs/PHASE3_NEURAL_REPRESENTATIONS.md`](docs/PHASE3_NEURAL_REPRESENTATIONS.md) | Population/PCA/identified representations and stop decision |
| [`docs/PHASE3_RANDI_RESPONSE_AUDIT.md`](docs/PHASE3_RANDI_RESPONSE_AUDIT.md) | Randi target, shape, averaging and context audit |
| [`docs/PHASE3_INTERVENTION_LIMITATIONS.md`](docs/PHASE3_INTERVENTION_LIMITATIONS.md) | Evidence required before a control claim |
| [`docs/PHASE3_RESULTS.md`](docs/PHASE3_RESULTS.md) | Phase 3 exact results and RQ answers |
| [`docs/PHASE3_SUPERVISOR_BRIEF.md`](docs/PHASE3_SUPERVISOR_BRIEF.md) | Phase 3 supervisor handoff and 6-ECTS scope |
| [`docs/PHASE3_NEXT_EXPERIMENTS.md`](docs/PHASE3_NEXT_EXPERIMENTS.md) | Prioritised replication/control experiments |
| [`docs/REPRODUCING.md`](docs/REPRODUCING.md) | Operator recipe with expected numbers and known nondeterminism |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Design decisions and rejected alternatives |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | Splits, models, metrics, branching and perturbation definitions |
| [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md) | Source, checksum, licensing, label caveats |
| [`docs/EXPERIMENT_REGISTRY.md`](docs/EXPERIMENT_REGISTRY.md) | Run matrix and artifact locations |
| [`docs/INTERVENTION_ASSUMPTIONS.md`](docs/INTERVENTION_ASSUMPTIONS.md) | What the virtual perturbation does and does not assume |
| [`docs/VALIDATION_PROTOCOL.md`](docs/VALIDATION_PROTOCOL.md) | Pre-specified tests, gates, and pass/fail outcomes |
| [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | Structural, statistical, and biological limitations |
| [`docs/NEXT_EXPERIMENTS.md`](docs/NEXT_EXPERIMENTS.md) | Highest-priority biological validation experiment |
| [`docs/LITERATURE_REVIEW.md`](docs/LITERATURE_REVIEW.md) | 20-source targeted review |
| [`docs/RESEARCH_FOUNDATIONS.md`](docs/RESEARCH_FOUNDATIONS.md) | Formal assumptions, claims ladder, RQ1–RQ7 |

---

## Figure gallery

- `figures/fig01_dataset_coverage.png` — behaviour composition and partial neuron coverage.
- `figures/fig02_model_comparison.png` — one-step R², decoding, transition TV; persistence line.
- `figures/fig03_forward_stress.png` — strict-forward degradation.
- `figures/fig04_bundlenet_official_3d.png` — official 3-D artifact, upstream-fitted (leaky).
- `figures/bundlenet_latent_3d.html` — interactive official artifact.
- `figures/fig05_branching_worm0.png` — local entropy, outcome mixtures, shuffled null.
- `figures/fig06_intervention_forest.png` — named vs random-neuron effects.
- `figures/fig07_ncmcm_verification.png` — distributed vs reimplemented p-values.
- `figures/fig08_observed_vs_rollout.png` — observed trajectory vs model rollout.
- `figures/fig09_phase2_summary.png` — corrected branching, RIVL specificity, Randi MAE, behaviour increment.
- `reports/phase3/figures/prospective_logloss.png` — Phase 3 animal-disjoint log loss by horizon.
- `reports/phase3/figures/neural_delta_ci.png` — primary neural Δ and control CIs.
- `reports/phase3/figures/randi_reliability.png` — Randi target/shape reliability.
- `reports/phase3/figures/randi_repeated_average.png` — repeated averaging vs global mean.
- `reports/phase3/figures/dataset_coverage.png` — Baskoylu event rate vs reversal occupancy.
- `reports/phase3/figures/cross_animal_dynamics.png` — one-step cross-animal dynamics.

---

## Licence and attribution

Source code is provided for research use. The Kato dataset retains its original
licence and attribution; upstream BunDLe-Net and NC-MCM code retain their
licences. See [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md).
