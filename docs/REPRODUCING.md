# Reproducing the Results

This is the operator recipe. It is written for a fresh checkout on Windows
(PowerShell); Linux/macOS commands are the same apart from environment activation.

---

## 1. Prerequisites

- Python 3.10–3.13 (validated with 3.13.5).
- ~2 GB disk for the environment, plus ~130 MB for the raw dataset.
- Network access to PyPI and OSF.
- `uv` (recommended) or `pip`.

```powershell
cd D:\ComputaCenter\NeuroBranch\neurobranch-celegans

# If uv is on PATH
uv sync --extra dev

# If uv is only available as a module
python -m uv sync --extra dev

# Plain pip alternative
python -m pip install -e ".[dev]"
```

If the `neurobranch` executable is not on PATH after a user-level pip install,
use `python -m neurobranch.cli.main` in place of `neurobranch`, or add the user
Scripts directory:

```powershell
$env:PATH="$env:APPDATA\Python\Python313\Scripts;$env:PATH"
```

---

## 2. Data

Place the file at:

```text
data/raw/kato2015/WT_NoStim.mat
```

Expected SHA256:

```text
7531811c1718e26a0455f59cb2f8188cf25809278cfea85770dfe03315a0098a
```

The raw file is **not** in Git. Download it from OSF project `2395t`
(<https://osf.io/2395t/>), then verify:

```powershell
neurobranch data audit
```

Expected: 5 worms, 15,662 frames, 192 named traces, SHA256 match. Audit outputs
go to `reports/`.

External reference artifacts are also not in Git. Clone them into:

```text
references/external/BunDLe-Net/
references/external/nc-mcm/
references/external/Celegans_premotor/
```

The NC-MCM precomputed `.npz` files are required for `reproduce ncmcm` and
`reproduce bundlenet`.

---

## 3. Primary suite

```powershell
neurobranch train dynamics `
  --config configs/experiments/blocked_suite.yaml `
  --models persistence,var,mlp,mlp_residual,latent_mlp,latent_knn,gru `
  --seeds 17,23,31 --worms 0,1,2,3,4
```

Expected: 105 runs, 0 failures, ~30–60 min on CPU depending on cores. Outputs:

```text
artifacts/runs/blocked_suite/
  suite_metrics.csv
  suite_summary.json
  run_manifest.json
  {model}__worm{w}__seed{s}/...
```

Sanity check:

```powershell
neurobranch evaluate --run-dir artifacts/runs/blocked_suite
```

The mean one-step R² for persistence should be ≈0.361; residual MLP ≈0.372; MLP
≈0.210.

---

## 4. Strict-forward negative control

```powershell
neurobranch train dynamics `
  --config configs/experiments/forward_stress.yaml `
  --models persistence,var,mlp,mlp_residual,latent_mlp,latent_knn,gru `
  --seeds 17 --worms 0,1,2,3,4
```

Expected: 35 runs. Direct MLP/GRU/latent models should have strongly negative R²
on worms 0–1; residual MLP should track persistence.

> The config's `evaluation.seeds` field is unused when `--seeds` is passed. The
> reported result used seed 17 only.

---

## 5. NC-MCM verification

```powershell
neurobranch reproduce ncmcm --k 3
```

Expected at K=3 (rounded):

| Worm | p pre | p reimpl |
|---:|---:|---:|
| 1 | 0.636 | 0.659 |
| 2 | 0.157 | 0.149 |
| 3 | 0.096 | 0.084 |
| 4 | 0.603 | 0.626 |
| 5 | 0.979 | 0.983 |

The surrogate test is stochastic; with 1000 surrogates p-values can vary by a few
hundredths between runs. The port reuses the stored cluster labels, so k-means
restarts are not a source of disagreement. A difference larger than ~0.10 suggests
a port or artifact mismatch. The CLI default is 1000 surrogates; the artifact was
generated with that default.

---

## 6. BunDLe-Net

Official artifact evaluation (fast):

```powershell
neurobranch reproduce bundlenet --worm 0 --epochs 120
```

This command trains the reimplementation for 120 epochs and writes
`artifacts/reproduction_bundlenet_lite.json`. To reproduce the official-artifact
table without training, call:

```powershell
@'
from neurobranch.representation.bundlenet import official_vs_reimplementation
for w in range(5):
    r = official_vs_reimplementation(w)
    print(w, r.get("official_knn_accuracy"), r.get("official_silhouette_fine"))
'@ | python -
```

Expected official kNN: 0.949–0.984, but this has **embedding-level leakage**
(upstream fitted the encoder on all frames, including test). Reimplementation kNN
≈0.407 on a blocked holdout with a causal filter (macro F1 0.304, silhouette
−0.064). These are not comparable and neither is a generalization estimate for
the official model. If you see ≈0.624, the acausal zero-phase filter has been
reintroduced; that number is leaked.

---

## 7. Branching

```powershell
neurobranch branching analyze --worm 0
```

Expected: 328 candidate frames, mean entropy ≈0.578 bits, iid null ≈0.961,
one-sided excess p=1.0, **deficit p≈0.005**. The Phase 1 command asks only the
excess direction. The corrected all-worm audit is:

```powershell
python scripts/phase2a_branching_audit.py
```

Expected: 4/5 worms show a significant deficit under iid/block-10 nulls; no worm
shows significant excess; worm 3 is not significant under block-10; the
circular-shift null is weaker. See
[`BRANCHING_STATISTICAL_AUDIT.md`](BRANCHING_STATISTICAL_AUDIT.md).

---

## 8. Virtual perturbation

```powershell
neurobranch simulate intervention `
  --worm 0 --model mlp --seed 17 `
  --spec-file configs/interventions/rivl_plus2.yaml `
  --out artifacts/intervention_rivl_demo.json
```

Expected: baseline P(TURN)=0.133, intervention P(TURN)=0.467, Δ=+0.333,
95% CI [0.167,0.533], OOD 10.6%. The output contains the standing warning and
`causal_status.validated = false`.

The candidate scan and corrected random controls were produced by
`interventions/search.py`; to rerun them:

```powershell
@'
import json
from pathlib import Path
import pandas as pd
from neurobranch.config import ExperimentConfig
from neurobranch.dynamics.checkpointing import load_model
from neurobranch.dynamics.training import load_worm_data
from neurobranch.interventions.search import random_control, run_sparse_search

cfg = ExperimentConfig.load("configs/experiments/blocked_suite.yaml")
data = load_worm_data(cfg)[0]
model, _ = load_model("artifacts/runs/blocked_suite/mlp__worm0__seed17/model.pt")

scan, _ = run_sparse_search(model, data, seed=17, n_initial=30, sim_seeds=(0, 1, 2), horizon=12)
scan.to_csv("artifacts/intervention_search_worm0.csv", index=False)

ctrl = random_control(model, data, n_random=20, seed=17, n_initial=30, sim_seeds=(0, 1, 2), horizon=12)
ctrl.to_csv("artifacts/intervention_random_controls_worm0.csv", index=False)
print(scan.head())
print(ctrl.describe())
'@ | python -
```

The random controls **must** exclude candidate cells; the code does this. If you
modify it, preserve that exclusion or the specificity test becomes circular.

Phase 2 scans six models, 26 candidate cells and 19 controls per model:

```powershell
python scripts/phase2a_intervention_audit.py
```

Expected: RIVL +2 ranges from 0.000 (residual MLP) to 0.733 (MLP 23); no model
reaches p<0.05 against matched controls; the maximum control is 0.567–0.767 in
the MLP models; `OLQVL` exceeds RIVL at +2. Runtime is ~20–40 min on CPU.

---

## 9. Reports and figures

```powershell
neurobranch report
@'
from neurobranch.reporting.figures import make_all
print(make_all())
'@ | python -
```

`neurobranch report` writes the CSV/JSON summaries; `make_all` writes the eight
PNGs and the interactive HTML. The figure generator prints `[figure skipped]` with
a reason if an input artifact is missing; it does not silently omit a figure.

---

## 10. App

```powershell
neurobranch app
```

or:

```powershell
python -m streamlit run app/streamlit_app.py --server.port 8501
```

Open <http://localhost:8501>. The app needs at least one model checkpoint and the
raw data. It has seven tabs: Overview, 3-D latent, Branching, Virtual
perturbation, Optogenetic (Randi), Reproduction, Audit.

Smoke test without a browser:

```powershell
@'
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app/streamlit_app.py", default_timeout=180).run()
print("exceptions:", len(at.exception), "tabs:", len(at.tabs))
'@ | python -
```

Expected: 0 exceptions, 7 tabs.

---

## 11. Tests and lint

```powershell
python -m pytest -q
python -m ruff check src tests app
```

Expected: 43 passed; lint clean. Scientific tests skip gracefully if the raw data
or external artifacts are absent; in this checkout all 43 run.

---

## 12. What is not reproducible from this repo alone

- The exact historical Git commit of the long training runs: the manifests were
  written after the runs, before the first commit, and therefore show
  `git.commit = null` and a dirty tree. `reports/posthoc_git_state.json` records
  only the later assembly commit. See
  [`VALIDATION_PROTOCOL.md`](VALIDATION_PROTOCOL.md#6-reproducibility-audit).
- Bitwise-identical results across machines: CPU threading and PyTorch kernels are
  not deterministic. Expect small differences in R² and p-values.
- The upstream BunDLe-Net/NC-MCM training runs: only artifacts and independent
  reimplementations are included.
- Any biological intervention: none was performed.

If you need a publication-grade rerun, start from a tagged commit, keep the
manifest writer enabled during the run, and report the resulting `run_manifest.json`
rather than regenerating it afterwards.

---

## 13. Phase 2 Randi 2023 audit

### 13.1 Download and verify

The raw archive is 523,093,816 bytes and is not committed.

```powershell
New-Item -ItemType Directory -Force data/raw/randi2023 | Out-Null
curl.exe -L --fail https://osf.io/9mecf/download -o data/raw/randi2023/exported_data.tar.gz
Get-FileHash data/raw/randi2023/exported_data.tar.gz -Algorithm SHA256
```

Expected SHA256:

```text
d6e7b3d93175b40b7ae17bde2182835e9c2144388142c522ee9be3832f6ce836
```

Extract:

```powershell
tar -xzf data/raw/randi2023/exported_data.tar.gz -C data/raw/randi2023
```

The archive contains `exported_data/` with 678 files. The adapter expects that
directory. See [`INTERVENTION_DATASET_AUDIT.md`](INTERVENTION_DATASET_AUDIT.md)
for schema and coverage.

### 13.2 Connectome reference

The connectome feature table uses the pumpprobe reference CSV. In this checkout:

```text
references/external/aconnectome_witvliet_2020_8_utf8.csv
```

The original file is UTF-16LE; it was converted to UTF-8. The benchmark degrades
to six models if the file is absent (no connectome ridge).

### 13.3 Run

```powershell
python scripts/phase2a_optogenetic_benchmark.py
```

Expected (on the verified archive):

- trial table: 237,639 rows, 3,371 events, 110 animals, 259 targets, 305
  recorded neurons;
- reliability: 580 groups, 925 pairs, median r ≈0.090;
- held-out events: global mean MAE 6.1113, connectome ridge 6.1112, paired
  Δ≈−0.000086, 95% CI [−0.0177, +0.0165];
- held-out targets: global 5.7251, connectome 5.7467;
- held-out animals: global 5.7249, neuron mean 5.7919.

No learned model beats the constant mean on targets or animals. This is a
negative result; do not tune on the test folds.

### 13.4 Behaviour benchmark

```powershell
python scripts/phase2a_behavior_benchmark.py
```

Expected: blocked and forward tables under
`reports/phase2a_scientific_audit/behavior/`. State+neural has higher transition
accuracy but worse balanced accuracy, log loss and Brier than state-only. See
[`BEHAVIORAL_GENERALIZATION.md`](BEHAVIORAL_GENERALIZATION.md).

### 13.5 Runtime and nondeterminism

- Randi parse + benchmark: ~10–20 min on CPU, ~1.2 GB raw data.
- Intervention audit: ~20–40 min.
- Branching audit: ~5–10 min.
- Bootstrap CIs vary slightly across runs; point estimates should match to the
  printed precision. Model fits use fixed seeds where specified; CPU threading
  can still change the last digits.

---

## 14. Phase 3: Baskoylu 2026 and Randi response audit

Phase 3 uses two external data sources:

- Baskoylu & Flavell (2026), Dryad `doi:10.5061/dryad.w9ghx3g4v`, version 2,
  27 animals, 30 files, 320,032,093 bytes, CC0-1.0;
- Randi et al. (2023), the OSF `e2syt` archive already used in Phase 2.

### 14.1 Download and verify Baskoylu

```powershell
python scripts/phase3_download_baskoylu.py
```

The Dryad API is protected by an Anubis proof-of-work challenge. The script
solves the SHA-256 challenge, keeps a cookie session, downloads all 30 files,
and writes `data/raw/baskoylu2026/manifest.json` with per-file SHA256. Expected
total: 320,032,093 bytes. Raw data stay under `data/raw/` and are not committed.

### 14.2 NeuroPAL labels

The JLD2 label file cannot be decoded by h5py 3.16/HDF5 2.0 (or 3.12/1.14):
the string data raises `ran off end of input buffer`. The one-time workaround is
a portable Julia 1.9.2 + JLD2.jl export:

```powershell
julia scripts/export_neuropal.jl
```

Expected: `data/raw/baskoylu2026/neuropal_labels.tsv`, 2,595 rows, 170 unique
labels, 6 ambiguous duplicate-label traces, SHA256
`7645ad1d112f4a412c84994597a81ad6aae809c275d65befcf68a1d482025d72`.
The Python pipeline reads only this TSV; Julia is not an analysis dependency.
`trace_index` is 0-based. See [`PHASE3_DATASET_AUDIT.md`](PHASE3_DATASET_AUDIT.md).

### 14.3 Prospective benchmark

```powershell
python scripts/phase3_prospective_benchmark.py
```

Expected runtime: ~5–8 min on CPU. It writes:

- `reports/phase3/dataset_summary.json` and `dataset_coverage.csv`;
- `reports/phase3/prospective_metrics_{3,6,12}s.csv`;
- `reports/phase3/prospective_deltas_{3,6,12}s.csv`;
- `reports/phase3/prospective_summary.json`;
- `reports/phase3/prospective_within_animal_metrics.csv`;
- `reports/phase3/prospective_c_sensitivity.csv`;
- `reports/phase3/prospective_run_manifest.json`.

Expected primary 6 s: behaviour log loss 0.5927, +neural 0.5856, Δ −0.00701,
95% animal CI [−0.01483, +0.00053], 20/27 improved. The CI includes zero; this
is the prespecified negative/inconclusive result, not a bug. Calibration slopes
are 1.005 (behaviour) and 0.963 (+neural).

### 14.4 Randi Phase 3 audit

```powershell
python scripts/phase3_randi_analysis.py
```

Expected: 237,639 rows, 3,371 events, reliability median r≈0.090 for peak and
≈0.092 for mean/integral; repeated-event averaging worse than global mean;
context MAE 5.9967→5.8900 with overlapping CI. Files are written under
`reports/phase3/randi_*`.

### 14.5 Cross-animal dynamics probe

```powershell
python scripts/phase3_cross_animal.py
```

Expected: persistence next-velocity MAE 0.02722, behaviour ridge 0.02587,
behaviour+neural 0.02590; next-state log loss 0.11631 vs 0.11655. Neural does not
improve. This is a one-step probe, not a generative simulator.

### 14.6 Figures and app

```powershell
python scripts/phase3_make_figures.py
python -m streamlit run app/streamlit_app.py
```

Figures go to `reports/phase3/figures/`. The app has eight tabs; the eighth,
**Prospective Behavior**, reads `reports/phase3/prospective_summary.json` and
`dataset_summary.json`. All seven Phase 2 tabs are unchanged.

### 14.7 Tests

```powershell
python -m pytest -q
python -m ruff check src tests scripts
```

Expected: 57 passed. The Phase 3 tests cover HDF5 parsing, the 0-based exclusive
event convention, timestamps, identity ambiguity, causal z-scoring, horizon
labels, purged chronological splits, calibration diagnostics, the app tabs and
the Randi target schema.

### 14.8 Nondeterminism

- Logistic regression uses fixed C=0.1, `lbfgs`, `max_iter=1000`, `tol=1e-3`;
  CPU threading can change the last digits.
- Bootstrap CIs use 10,000 animal resamples with seed 17 but a slightly different
  RNG draw when the number of compared modes changes. Point estimates are stable;
  the C sensitivity table is the authoritative check for the primary.
- Julia export is deterministic for a fixed JLD2 file. Do not regenerate the TSV
  with a different JLD2.jl version without recording it.
