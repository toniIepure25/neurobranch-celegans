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

Expected: 328 candidate frames, mean entropy ≈0.578 bits, null mean ≈0.960,
p=1.0. The command runs 100 permutations and takes ~1 min.

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
raw data. It has six tabs: Overview, 3-D latent, Branching, Virtual perturbation,
Reproduction, Audit.

Smoke test without a browser:

```powershell
@'
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app/streamlit_app.py", default_timeout=180).run()
print("exceptions:", len(at.exception), "tabs:", len(at.tabs))
'@ | python -
```

Expected: 0 exceptions, 6 tabs.

---

## 11. Tests and lint

```powershell
python -m pytest -q
python -m ruff check src tests app
```

Expected: 23 passed; lint clean. Scientific tests skip gracefully if the raw data
or external artifacts are absent; in this checkout all 23 run.

---

## 12. What is not reproducible from this repo alone

- The exact historical Git commit of the long training runs: the manifests were
  written after the runs and may show a dirty tree. See
  [`VALIDATION_PROTOCOL.md`](VALIDATION_PROTOCOL.md#6-reproducibility-audit).
- Bitwise-identical results across machines: CPU threading and PyTorch kernels are
  not deterministic. Expect small differences in R² and p-values.
- The upstream BunDLe-Net/NC-MCM training runs: only artifacts and independent
  reimplementations are included.
- Any biological intervention: none was performed.

If you need a publication-grade rerun, start from a tagged commit, keep the
manifest writer enabled during the run, and report the resulting `run_manifest.json`
rather than regenerating it afterwards.
