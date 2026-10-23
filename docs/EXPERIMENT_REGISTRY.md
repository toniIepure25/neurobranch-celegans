# Experiment Registry

Every experiment that produced a reported number, with its exact command, config,
seed set, and artifact location. Paths are relative to the repository root.

Environment for all runs: Python 3.13.5, NumPy 2.3.1, SciPy 1.16.0,
scikit-learn 1.9.1, PyTorch 2.7.1, CPU. Run manifests record the Git commit and
dirty flag at execution time; because the final documentation and test changes
were made after the long training runs, the **manifests may show a dirty tree**.
The training code and configs did not change between the suite runs and this
registry; the only later code change affecting an experiment is the corrected
random-control sampling (E6), which was re-run.

---

## E1. Dataset audit

| Field | Value |
|---|---|
| Command | `neurobranch data audit` |
| Input | `data/raw/kato2015/WT_NoStim.mat` |
| Outputs | `reports/dataset_audit.json`, `reports/recording_summary.csv`, `reports/neuron_identity_table.csv`, `reports/macro_transitions.csv`, `reports/fine_state_transitions.csv` |
| Worms | 5 |
| Frames | 15,662 |
| Failures | 0 |

---

## E2. Primary blocked suite

| Field | Value |
|---|---|
| Config | `configs/experiments/blocked_suite.yaml` |
| Command | `neurobranch train dynamics --config configs/experiments/blocked_suite.yaml --models persistence,var,mlp,mlp_residual,latent_mlp,latent_knn,gru --seeds 17,23,31 --worms 0,1,2,3,4` |
| Split | blocked, 5 blocks, test block 2, val block 4, purge 12 |
| Models | persistence, var, mlp, mlp_residual, latent_mlp, latent_knn, gru |
| Seeds | 17, 23, 31 |
| Worms | 0–4 |
| Epochs | 60, patience 15 |
| Runs | 105 |
| Failures | 0 |
| Output root | `artifacts/runs/blocked_suite/` |
| Tables | `suite_metrics.csv` (105 rows), `suite_summary.json`, `run_manifest.json` |
| Per-run | `{model}__worm{w}__seed{s}/model.pt`, `metrics.json`, `train_history.json`, `test_predictions.npz`, `resolved_config.yaml` |
| Aggregates | `reports/results_model_aggregate.csv`, `reports/results_paired_vs_persistence.csv` |

**Use.** Primary model comparison. It is **interpolation**, not forecasting.

---

## E3. Strict-forward stress test

| Field | Value |
|---|---|
| Config | `configs/experiments/forward_stress.yaml` |
| Command | `neurobranch train dynamics --config configs/experiments/forward_stress.yaml --models persistence,var,mlp,mlp_residual,latent_mlp,latent_knn,gru --seeds 17 --worms 0,1,2,3,4` |
| Split | temporal 70/15/15, purge 12 |
| Seeds | 17 only |
| Runs | 35 |
| Failures | 0 |
| Output root | `artifacts/runs/forward_stress/` |
| Tables | `suite_metrics.csv`, `suite_summary.json`, `run_manifest.json` |
| Aggregate | `reports/results_forward_stress.csv` |

**Use.** Negative control for non-stationarity; not used to select the simulator.

> The config lists `evaluation.seeds: [17,23,31]`, but the executed command used
> `--seeds 17`. The registry records what was run, not what the unused config
> field says. Re-running with three seeds is supported but not part of the
> reported result.

---

## E4. NC-MCM artifact verification

| Field | Value |
|---|---|
| Command | `neurobranch reproduce ncmcm --k 3` |
| Inputs | `references/external/nc-mcm/precomputed_results/nc_mcm_model_worm_{1..5}.npz` |
| Outputs | `artifacts/reproduction_ncmcm.json`, `.csv`, `reports/results_ncmcm.csv` |
| K | 3 |
| Surrogates | 100 (reimplementation), distributed curves reused for the precomputed p |
| Decoder | 5-fold blocked logistic regression, 8 fine states |
| Worms | 5 |
| Failures | 0 |

**Claim.** The Python port of the lag-2 Markov test agrees with the distributed
p-values within 0.023 (1000 surrogates); the distributed NC-MCM *training* was not reproduced.

---

## E5. BunDLe-Net

### E5a. Official artifacts

| Field | Value |
|---|---|
| Inputs | `references/external/nc-mcm/precomputed_results/bundlenet_consistent_embedding_worm_{1..5}.npz` |
| Output | `artifacts/reproduction_bundlenet_official.json` |
| Metric | kNN (k=15) on the 3-D embedding, silhouette (8 fine states), PCA-3 baseline |
| Caveat | **embedding-level leakage**: upstream fits all windows; test embeddings were seen by the encoder. Silhouette is fully in-sample. |

### E5b. PyTorch reimplementation

| Field | Value |
|---|---|
| Command | `neurobranch reproduce bundlenet --worm 0 --epochs 120` |
| Output | `artifacts/reproduction_bundlenet_lite.json` |
| Split | blocked holdout on worm 0 |
| Window | 15 |
| Latent | 3 |
| Best epoch | 12 (early stopping) |
| Train/test windows | 1853 / 612 |
| Metrics | kNN 0.407, macro F1 0.304, silhouette −0.064, trustworthiness 0.757 (causal filter; earlier 0.624 was acausal leakage) |
| Deviations | recorded in the JSON `metadata.deviations` |

**Claim.** Partial reproduction / integration only. Not a faithful replication of
the published benchmark.

---

## E6. Virtual perturbation scan and controls

| Field | Value |
|---|---|
| Model | `artifacts/runs/blocked_suite/mlp__worm0__seed17/model.pt` |
| Worm | 0 |
| Spec | additive, onset 0, duration 12, horizon 12, ±2 SD, 30 REV initial states, seeds 0/1/2, noise 0.04 |
| Candidates | 13 available named cells × 2 amplitudes = 26 runs |
| Outputs | `artifacts/intervention_search_worm0.csv`, `.json` |
| Controls (corrected) | 19 randomly sampled **non-candidate** named predictor cells, same +2 SD / timing / seeds, `seed=17` |
| Control outputs | `artifacts/intervention_random_controls_worm0.csv`, `.json` |
| Figures | `figures/fig06_intervention_forest.png` |
| Status | **MODEL-DEPENDENT — NOT CAUSALLY VALIDATED** |

**Correction history.** The first control CSV sampled from the full predictor pool
*including* candidate cells; it was not a valid null. The code was fixed
(`interventions/search.py`) to exclude candidates, and the controls were re-run.
Only the corrected 19-cell control distribution is reported. The earlier values
are superseded and should not be cited.

**Results.** Top candidate RIVL +2 SD: ΔP(TURN)=+0.333, CI [0.167, 0.533], OOD
10.6%. Corrected random controls: mean |Δ|=0.194, max |Δ|=0.667 (OLQVL), 4/19
controls ≥ |0.333|, empirical p≈0.21. Not specific.

---

## E7. Branching analysis

| Field | Value |
|---|---|
| Command | `neurobranch branching analyze --worm 0` |
| Projection | PCA-3 fitted on training frames |
| Candidates | 328 held-out REV frames with 20 future frames |
| k | 40, excluding ±12 frames |
| Horizon | 20 |
| Null | 100 outcome permutations |
| Output | `artifacts/branching_worm0.json` |
| Figure | `figures/fig05_branching_worm0.png` |

**Result.** Mean entropy 0.578 bits vs null 0.960; excess −0.382; p=1.0. No
evidence of excess branching.

---

## E8. Reports and figures

| Field | Value |
|---|---|
| Command | `neurobranch report` (plus `python -c "from neurobranch.reporting.figures import make_all; make_all()"`) |
| Inputs | all artifacts above |
| Outputs | `reports/results_model_aggregate.csv`, `reports/results_paired_vs_persistence.csv`, `reports/results_forward_stress.csv`, `reports/results_ncmcm.csv`, `reports/results_summary.json`, `reports/epistemic_manifest.json`, `figures/*.png`, `figures/bundlenet_latent_3d.html` |
| Failures | 0 (figure generator catches and reports a skipped figure rather than silently omitting) |

> `figures/fig04_bundlenet_official_3d.png` had a scalar-array bug during the
> first figure pass; it was fixed and regenerated. No result depended on it.

---

## E9. Test suite

| Field | Value |
|---|---|
| Command | `python -m pytest -q` |
| Result | **23 passed** in ~5 s |
| Categories | 12 unit, 2 integration, 9 scientific |
| Lint | `python -m ruff check src tests app` → all checks passed |

The tests include a real-data scientific check that behaviour-label neurons are
excluded, a real-data split-leakage check, a real-data NC-MCM p-value agreement
check, and a real-data official BunDLe shape check. They skip gracefully if the
external artifacts are absent.

---

## E10. App

| Field | Value |
|---|---|
| Command | `neurobranch app` or `python -m streamlit run app/streamlit_app.py` |
| URL | <http://localhost:8501> |
| Smoke test | Streamlit health `ok`; `AppTest` executed 7 tabs with **0 exceptions** |
| Tabs | Overview, 3-D latent, Branching, Virtual perturbation, Optogenetic (Randi), Reproduction, Audit |

---

## E11. Phase 2 audits

| Field | Value |
|---|---|
| Branch | `research/phase2-intervention-validation` |
| Branching command | `python scripts/phase2a_branching_audit.py` |
| Branching outputs | `reports/phase2a_scientific_audit/branching/` |
| Intervention command | `python scripts/phase2a_intervention_audit.py` |
| Intervention outputs | `reports/phase2a_scientific_audit/intervention/` |
| Behaviour command | `python scripts/phase2a_behavior_benchmark.py` |
| Behaviour outputs | `reports/phase2a_scientific_audit/behavior/` |
| Randi command | `python scripts/phase2a_optogenetic_benchmark.py` |
| Randi outputs | `reports/phase2b_optogenetic_atlas/` |
| Randi raw data | `data/raw/randi2023/exported_data/` (not tracked) |
| Randi hash | `d6e7b3d93175b40b7ae17bde2182835e9c2144388142c522ee9be3832f6ce836` |
| Negative result | pre-stimulus/connectome models do not beat constant mean on held-out targets/animals |
| Tests | `43 passed`; ruff clean |

Phase 2 reports are listed in `README.md` and `docs/PHASE2_METHODOLOGY.md`.

---

## Artifact inventory (quick check)

```text
artifacts/
├─ runs/blocked_suite/          105 runs + suite_metrics.csv + suite_summary.json + run_manifest.json
├─ runs/forward_stress/         35 runs + suite_metrics.csv + suite_summary.json + run_manifest.json
├─ reproduction_ncmcm.json/.csv
├─ reproduction_bundlenet_official.json
├─ reproduction_bundlenet_lite.json
├─ intervention_search_worm0.csv/.json
├─ intervention_random_controls_worm0.csv/.json
└─ branching_worm0.json
reports/
├─ dataset_audit.json
├─ recording_summary.csv
├─ neuron_identity_table.csv
├─ macro_transitions.csv
├─ fine_state_transitions.csv
├─ results_*.csv/json
└─ epistemic_manifest.json
figures/
├─ fig01…fig08 .png
└─ bundlenet_latent_3d.html
```
