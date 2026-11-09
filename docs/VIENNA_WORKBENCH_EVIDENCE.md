# Vienna workbench — source-to-claim trail

Checked 2026-11-09. This file separates peer-reviewed papers, preprints, third-party code/data,
and our own computations. It is a scoped engineering/research note, not a systematic review.

## 1. Publication records

| ID | Record | Verification | Status |
|---|---|---|---|
| BunDLe-Net | Kumar, Gilra, Meunier, Grosse-Wentrup, “BunDLe-Net: Learning minimal neural manifolds that represent only behaviourally relevant dynamics”, bioRxiv 2023 | Crossref `10.1101/2023.08.08.551978`: type `posted-content`, publisher openRxiv | **Preprint**, not a peer-reviewed journal article |
| NC-MCM | Grosse-Wentrup, Kumar, Meunier, Zimmer, “Neuro-cognitive multilevel causal modeling…”, PLoS Comput Biol 2024 | Crossref `10.1371/journal.pcbi.1012674`: `journal-article`; PubMed PMID 39680605: `Journal Article`, notices `[]` | Peer-reviewed; no recorded PubMed notice (absence is not proof of a clean record elsewhere) |
| Premotor | Morrison & Young, “A data-driven biology-based network model reproduces C. elegans premotor neural dynamics”, PLoS Comput Biol 2025 | Crossref `10.1371/journal.pcbi.1013818`: `journal-article`; PubMed PMID 41460929: `Journal Article`, notices `[]`; PMC12768384 | Peer-reviewed; no recorded PubMed notice |

No publication by Denis Kovačević was found on the group page/UCris during this work. The
workbench therefore starts from the group's published papers. Do **not** attribute BunDLe-Net or
NC-MCM to Denis.

## 2. Claim ledger

| # | Claim | Evidence | Type | Confidence / caveat |
|---|---|---|---|---|
| C1 | BunDLe-Net learns a 3-D latent manifold preserving behaviourally relevant dynamics | Paper + released embeddings; our loader reproduces five 3-D trajectories | Published method + our inspection | Supported for the released five worms; preprint |
| C2 | A quadratic map on these latents is highly predictable one step ahead | Our leave-one-worm-out run: mean R² 0.998 (0.997–0.998) | Our computation | Strong one-step result; not a generative validation |
| C3 | The same quadratic map is a reliable free-run simulator | Mean skill +0.163 vs constant initial state, but one worm −0.331; behaviour decoding 33.5% vs 29.8% majority | Our computation | **Not supported** as a validated generator |
| C4 | A manifold walk gives the strongest short-horizon open-loop forecast among the three tested models | Held-out-worm benchmark v2, 25-frame horizon, 16 seeds: latent MSE 0.0445, CRPS 0.0969, behaviour accuracy 83.8%, NLL 0.542; quadratic 0.0539/0.1150/82.8%/0.618; switching 0.0628/0.1210/66.8%/0.767 | Our computation | Supported descriptively for these five worms; intervals under-cover; not mechanism, not extrapolation |
| C5 | NC-MCM exposes multilevel cognitive states and Markov tests | Published method + released `c`, `p`, `bprob` arrays; schema mapped from Julia source | Published method + our reader | Supported descriptively |
| C6 | The UI can estimate `do(state)` effects | It propagates a forced state through the fitted transition matrix; no experimental intervention; audit rejects strict first-order Markov and stationarity | Our implementation + audit | **Not supported as causal**; labelled Markov counterfactual |
| C7 | The premotor model is a 205-node fitted synaptic rate model | Published paper + `A_10takes_avg`, `d_10takes_avg`; recovered `W` and names | Published model + our loader/builder | Supported for `A`, `d`, `W`, ODE form |
| C8 | Tier 3 reproduces the published 15-intrinsic + 69-clamped-signal configuration | Canonical `2023-01-23-15`: 84 core / 15 intrinsic / 69 signal; β=10 run stable (max |u| 3.337). Fresh Julia 1.11.2 reference: mean trace r 0.9999997, min 0.9999995, max |Python−Julia| 0.0133; canonical behaviour agreement 60.3%, distribution JS 0.0011 | Our computation + authors' Julia code | **Configuration and port supported**; biological/Digital-twin claim still rejected |
| C9 | Any tier establishes whole-brain mechanism or causality | None does; Tier 1 is latent, Tier 2 is assumption-dependent and fails strict Markov/stationarity, Tier 3 is a reduced replay/sandbox | Inference from above | **Rejected** |

## 3. Reproducibility artifacts

- Git branch: `feature/vienna-v2-clamped-audit` (base `4250072`); commit SHA is recorded in the
  generated reference report.
- Artifact manifest: `data/processed/vienna_artifacts_manifest.json` (SHA-256, bytes, license,
  source path for 13 files, including derived `W.npy` and `neuron_names.txt`).
- Tier 1 model/metrics: `artifacts/bundlenet_sim/model.npz`, `metrics.json`, `manifest.json`.
- Evaluation unit: one worm; leave-one-worm-out. No frame-level random split.
- Seeds: manifold-walk UI seed is user-controlled; training uses deterministic ridge solves.
- Commands:

```bash
python scripts/fetch_vienna_artifacts.py --verify-only
PYTHONPATH=src python -m neurobranch.bundlenet_sim.train --alpha 1.0
PYTHONPATH=src python -m neurobranch.bundlenet_sim.benchmark --horizon 25 --stride 75 --seeds 16
PYTHONPATH=src python -m neurobranch.bundlenet_sim.causal_audit --k 6
PYTHONPATH=src python scripts/build_tier3_validation.py
PYTHONPATH=src python scripts/compare_tier3_reference.py
PYTHONPATH=src python -m pytest tests/test_bundlenet_clamped.py tests/test_causal_audit.py tests/test_bundlenet_workbench.py -q
python -m streamlit run src/neurobranch/bundlenet_sim/app.py
```

- Julia reference: Julia 1.11.2, `OrdinaryDiffEq` `Tsit5`, 84/15/69, canonical recording; the
  standalone script is `scripts/reference/run_julia_reference.jl`. Input and output SHA-256 hashes
  are in `reports/vienna_workbench/tier3_julia_reference.json`.
- Environment used locally: Python 3.13, NumPy/SciPy/scikit-learn/h5py/Streamlit 1.66.1/Plotly 7.1.
  The workbench code does not require TensorFlow; Julia is only needed to regenerate the reference.
- Pod environment: Python 3.12.3, Streamlit 1.66.1, Plotly 7.1.0, h5py 3.16.0, scikit-learn 1.6.1,
  NumPy 1.26.4, NVIDIA H100 NVL. Tests and reports are rerun on the pod after deployment.

## 4. What was not done

- No fresh independent reproduction of the BunDLe-Net training (we used released embeddings).
- No independent Julia execution of NC-MCM; its arrays were read from the released NPZ. The
  premotor model **was** executed in Julia for the canonical reference comparison.
- **Reference comparison is one recording, not a full parameter sweep.** It checks the port on
  `2023-01-23-15`; the other two validation recordings use the same code but were not separately
  run through Julia.
- No experimental intervention data, so no identified causal effect estimate.
- No external validation on a sixth worm or another lab's recordings.
- No full historical test-suite green claim; the Vienna contract tests are run explicitly.

## 5. Search / inspection trail

- Crossref DOI checks for the three papers.
- PubMed status checks for PMIDs 39680605 and 41460929.
- Europe PMC record/abstract check for PMID 41460929.
- Local source inspection: `nc-mcm/script_to_generate_ncmcms.jl`, `nilab/src/mcm.jl`,
  `nilab/src/supervised_models/logreg_model.jl`, `Celegans_premotor/functions/functions_simulation.jl`,
  `Celegans_premotor/code/simulations.ipynb`, `Celegans_premotor/data/White_data_processed/df5.csv`.
- HDF5 inspection of `avg_model_weights.jld2` (keys `A_10takes`, `A_10takes_avg`,
  `d_10takes_avg`; shapes (22,205,205), (205,205), (205,)).
- Premotor name/W recovery: parsed the 205-neuron loop from
  `code/computer_avg_weights.ipynb` cell 16 output, rebuilt `EJ_mat`/`S_mat` from
  `data/White_data_processed/df5.csv`, and accepted only the order for which every non-zero `A`
  entry lies inside the transposed chemical-synapse support (913/913). A wrong order gives ~3%
  support overlap; this one gives 100% containment. The h5py/JLD2 layout is transposed relative
  to Julia's in-memory matrix, so `load_premotor` transposes it back.
- Pod reproduction: extracted the 19 MB deployment bundle to `/workspace/vienna-studio`, ran
  `pytest` 7/7 and the benchmark; values matched local to floating-point tolerance.
