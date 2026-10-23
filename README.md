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
| Branching analysis | ✅ empirical + model-based; null test and projection-stability test |
| Virtual perturbation engine | ✅ paired baselines, common random numbers, multi-seed uncertainty, OOD support flags, random-neuron controls |
| Sparse candidate search | ✅ 13 named cells × 2 amplitudes + 19 non-candidate random controls |
| Interactive 3-D app | ✅ Streamlit + Plotly, 6 tabs |
| CLI | ✅ Typer, documented commands |
| Tests | ✅ 23 passed (unit, integration, scientific) |
| Figures | ✅ 8 static PNGs + interactive HTML |
| Causal validation | ❌ not attempted; evidence gate refuses unsupported claims |

The headline result is deliberately unflattering: on held-out data, **no neural
dynamics model clearly beats a persistence baseline**, behaviour decoding is far
below label persistence, apparent branching shows *less* local outcome diversity
than a shuffled null, and the top named-neuron virtual effect is **within the
range of arbitrary non-candidate cells** (4/19 controls match or exceed it,
empirical p≈0.21), so it is not specific. These are findings, not bugs; they are preserved in
[`docs/RESULTS.md`](docs/RESULTS.md) and [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

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
├─ docs/                         methodology, provenance, results, limitations,
│                                supervisor brief, paper prospectus
├─ figures/                      generated PNG/HTML figures
├─ reports/                      audit tables and result summaries
├─ references/external/          cloned upstream repositories (not committed)
├─ src/neurobranch/
│  ├─ behavior/                  runs, transitions, entropies
│  ├─ branching/                 local outcome diversity, null test, dimension stability
│  ├─ cli/                       Typer entry point
│  ├─ data/                      Kato HDF5 loader, provenance, synthetic generator
│  ├─ dynamics/                  baselines, neural/latent models, training, checkpoints
│  ├─ evaluation/                metrics, bootstrap, transition distances, rollouts
│  ├─ interventions/             specs, paired engine, OOD, evidence gate, search
│  ├─ preprocessing/             leakage-safe split/scaling/windowing
│  ├─ representation/            BunDLe-Net artifact integration + reimplementation
│  ├─ reproduction/              NC-MCM verification
│  └─ reporting/                 tables and figures from saved artifacts
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

---

## Licence and attribution

Source code is provided for research use. The Kato dataset retains its original
licence and attribution; upstream BunDLe-Net and NC-MCM code retain their
licences. See [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md).
