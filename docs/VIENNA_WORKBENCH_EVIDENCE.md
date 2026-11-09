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
| C4 | A manifold walk gives stable interactive rollouts | Walk samples observed successor states; finite, on-manifold by construction | Our implementation | Stable, but descriptive; no extrapolation |
| C5 | NC-MCM exposes multilevel cognitive states and Markov tests | Published method + released `c`, `p`, `bprob` arrays; schema mapped from Julia source | Published method + our reader | Supported descriptively |
| C6 | The UI can estimate `do(state)` effects | It only re-displays conditional behaviour distributions and outgoing transitions | Our implementation | **Not supported**; explicitly labelled observational |
| C7 | The premotor model is a 205-node fitted synaptic rate model | Published paper + `A_10takes_avg`, `d_10takes_avg` in `avg_model_weights.jld2` | Published model + our loader | Supported for `A`, `d`, ODE form |
| C8 | Tier 3 is the full published premotor model / a digital twin | `W` (gap junctions) and neuron-name order are absent from the public export; `W=0`, nodes anonymous | Inspection of export | **Not supported**; sandbox only |
| C9 | Any tier establishes whole-brain mechanism or causality | None does; Tier 1 is latent, Tier 2 is assumption-dependent, Tier 3 is a reduced model | Inference from above | **Rejected** |

## 3. Reproducibility artifacts

- Git branch: `feature/bundlenet-simulator` (base `4250072`).
- Artifact manifest: `data/processed/vienna_artifacts_manifest.json` (SHA-256, bytes, license,
  source path for 11 files).
- Tier 1 model/metrics: `artifacts/bundlenet_sim/model.npz`, `metrics.json`, `manifest.json`.
- Evaluation unit: one worm; leave-one-worm-out. No frame-level random split.
- Seeds: manifold-walk UI seed is user-controlled; training uses deterministic ridge solves.
- Commands:

```bash
python scripts/fetch_vienna_artifacts.py --verify-only
PYTHONPATH=src python -m neurobranch.bundlenet_sim.train --alpha 1.0
PYTHONPATH=src python -m pytest tests/test_bundlenet_workbench.py -q
python -m streamlit run src/neurobranch/bundlenet_sim/app.py
```

- Environment used locally: Python 3.13, NumPy/SciPy/scikit-learn/h5py/Streamlit 1.66.1/Plotly 7.1.
  The workbench code does not require TensorFlow or Julia.

## 4. What was not done

- No fresh independent reproduction of the BunDLe-Net training (we used released embeddings).
- No Julia execution of NC-MCM or the premotor model; no `W` recovery.
- No experimental intervention data, so no causal effect estimate.
- No external validation on a sixth worm or another lab's recordings.
- No full test-suite green claim. The repository has pre-existing failing/skipped historical tests;
  only the new workbench tests (5/5) and app smoke test were run for this change.

## 5. Search / inspection trail

- Crossref DOI checks for the three papers.
- PubMed status checks for PMIDs 39680605 and 41460929.
- Europe PMC record/abstract check for PMID 41460929.
- Local source inspection: `nc-mcm/script_to_generate_ncmcms.jl`, `nilab/src/mcm.jl`,
  `nilab/src/supervised_models/logreg_model.jl`, `Celegans_premotor/functions/functions_simulation.jl`,
  `Celegans_premotor/code/simulations.ipynb`, `Celegans_premotor/data/White_data_processed/df5.csv`.
- HDF5 inspection of `avg_model_weights.jld2` (keys `A_10takes`, `A_10takes_avg`,
  `d_10takes_avg`; shapes (22,205,205), (205,205), (205,)).
