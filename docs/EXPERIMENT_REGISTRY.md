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

## E-P3. Phase 3 prospective benchmark and external audits

| Field | Value |
|---|---|
| Branch | `research/phase3-neural-generalization` |
| Download command | `python scripts/phase3_download_baskoylu.py` |
| Raw data | `data/raw/baskoylu2026/files/` (not tracked; 30 files, 320,032,093 bytes) |
| Dataset manifest | `data/raw/baskoylu2026/manifest.json` (per-file SHA256) |
| Identity export | `python scripts/export_neuropal.jl` → `data/raw/baskoylu2026/neuropal_labels.tsv` |
| Prospective command | `python scripts/phase3_prospective_benchmark.py` |
| Prospective outputs | `reports/phase3/prospective_*`, `dataset_summary.json`, `dataset_coverage.csv`, `neuron_identity_coverage.csv`, `prospective_run_manifest.json` |
| Primary target | reversal onset in `(t, t+6 s]`, animal-disjoint LOAO |
| Primary model | L2 logistic, C=0.1, causal features, 10,000 animal bootstrap (seed 17) |
| Primary result | Δ log loss −0.00701, 95% CI [−0.01483, +0.00053], 20/27 improved; **criterion not met** |
| Randi command | `python scripts/phase3_randi_analysis.py` |
| Randi outputs | `reports/phase3/randi_*` |
| Cross-animal command | `python scripts/phase3_cross_animal.py` |
| Cross-animal outputs | `reports/phase3/cross_animal_dynamics.csv/.json` |
| Figures | `python scripts/phase3_make_figures.py` → `reports/phase3/figures/` |
| Manifest | `python scripts/phase3_write_manifest.py` |
| Tests | `58 passed` at the Phase 3 checkpoint; current suite `62 passed`; ruff clean |

Phase 3 does not perform an intervention and does not establish causal control.

### Phase 4: Kramer 2026 turn direction

| Field | Value |
|---|---|
| Branch | `research/phase4-turn-direction-generalization` |
| Base | Phase 3 commit `66cf7bbd2aa21887ad5716e6bae1e0768eaa1c86` |
| Download command | `python scripts/phase4_download_kramer.py` |
| Raw data | `data/raw/kramer2026/files/` (not tracked; 80 files, 1,444,081,660 bytes) |
| Dataset manifest | `data/raw/kramer2026/manifest.json` (82-file Dryad manifest) |
| Audit command | `python scripts/phase4_audit_data.py` |
| Audit outputs | `dataset_inventory.csv`, `cohort_inventory.csv`, `neuron_coverage.csv`, `behavior_event_inventory.csv`, `marker_sanity.csv`, `data_checksums.csv` |
| Replication command | `python scripts/phase4_independent_replication.py` |
| Replication result | 6 s Δ −0.00570, 95% CI [−0.01133, +0.00047], 22/32 improved; **criterion not met** |
| Primary command | `python scripts/phase4_turn_direction.py` |
| Primary target | post-reversal D/V direction from a four-frame mid-reversal segment, animal-disjoint LOAO |
| Primary model | StandardScaler + L2 logistic, C=0.1, 2,000 animal bootstrap (seed 17) |
| Primary result | Δ AUROC +0.0123, 95% CI [−0.0166, +0.0425]; **no incremental evidence** |
| Dynamics/controls | `phase4_dynamics.py`, `phase4_controls.py`, `phase4_ablation.py`, `phase4_nonlinear.py` |
| Figures | `python scripts/phase4_make_figures.py` → `reports/phase4/figures/` |
| Manifest | `python scripts/phase4_write_manifest.py` → `reports/phase4/phase4_manifest.json` |
| Tests | Current suite `62 passed`; ruff clean |

Phase 4 does not perform an intervention; tdc1 is a distribution-shift cohort.
The archive lacks trial-level optogenetic data.


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


## Phase 10 (2026-10-29)

| Artifact | Command | Purpose |
| --- | --- | --- |
| `reports/phase10/reliability_summary.json` | `python scripts/phase10_reliability.py` | Schema v2 single-animal and pooled ceilings |
| `reports/phase10/ceiling_curve.csv` | same | Reliability vs animals pooled |
| `reports/phase10/profile_components.csv` | same | Per-target signal, noise, interaction |
| `reports/phase10/summary_sensitivity.csv` | `python scripts/phase10_robustness.py` | Response-summary sensitivity |
| `reports/phase10/half_replication.json` | same | Animal-half rank stability |
| `reports/phase10/cluster_ci.json` | same | Animal jackknife diagnostic |
| `reports/phase10/design_budget.csv` | `python scripts/phase10_design.py` | Animals needed per reliability goal |
| `reports/phase10/coverage_study.json` | same | Synthetic interval coverage (negative result) |
| `reports/phase10/claims.json` | `python scripts/phase10_claims.py` | Schema v2 claims ledger |
| `reports/phase10/manifest.json` | reliability script | Input hash, artifact hashes, versions |


## Phase 11 (2026-10-29)

| Artifact | Command | Purpose |
| --- | --- | --- |
| `reports/phase11/summary.json` | `python scripts/phase11_sensitivity.py` | Coverage-aware pooled ceiling (0.560) and gate failures |
| `reports/phase11/coverage_sensitivity.csv` | same | Reliability and ceiling across 0.0--1.0 responder coverage |
| `reports/phase11/benchmark_card.json` | `python scripts/phase11_benchmark.py` | Historical headline ceiling and claim statuses |

## Phase 15 (2026-11-01): three-estimand reliability audit

| Artifact | Command | Purpose |
| --- | --- | --- |
| `research/phase15_protocol.md` | frozen before run | Three estimands, uncertainty, 1,000-rep calibration and gates |
| `reports/phase15/summary.json` | `python scripts/phase15_reliability.py` | Full audit, gates and fail-closed decision |
| `reports/phase15/estimands.csv` | same | Per-target `R_obs`, `R_22` and identification flags |
| `reports/phase15/jackknife.csv` | same | Delete-one-animal replicate values |
| `reports/phase15/bayesian_bootstrap.csv` | same | 2,000-draw animal Dirichlet cross-check |
| `reports/phase15/calibration.json` | same | Primary 1,000-rep coverage study |
| `reports/phase15/calibration_heavy.json` | `python scripts/phase15_calibrate.py --heavy --reps 500` | Heavy-tailed misspecification check |
| `reports/phase15/manifest.json` | `python scripts/phase15_manifest.py` | Artifact hashes, input/protocol hashes, git state |

## Phase 12 (2026-10-30, historical calibration evidence)

| Artifact | Command | Purpose |
| --- | --- | --- |
| `reports/phase12/summary.json` | `python scripts/phase12_hierarchy.py` | Original hierarchical fit and gate decision |
| `reports/phase12/sbc.json` | same | Original SBC (unaligned truth; superseded) |
| `reports/phase12/benchmark_cluster.json` | `python scripts/phase12_benchmark.py` | Withdrawn cluster comparison (C11) |

## Phase 12R (2026-10-30, correction)

| Artifact | Command | Purpose |
| --- | --- | --- |
| `research/phase12r_protocol.md` | frozen before run | Preregistered alignment correction and gates |
| `research/phase12r_deviations.md` | recorded before run | False variance-bug diagnosis and deterministic truth deviation |
| `reports/phase12r/summary.json` | `python scripts/phase12r_hierarchy.py` | Aligned rerun; not adopted |
| `reports/phase12r/sbc.json` | same | 3,000-fit SBC; coverage 0.933, rank p 1.43e-06 |
| `reports/phase12r/heldout_ppc.json` | same | 20-split animal-holdout check |
| `reports/phase12r/negative_control.json` | same | 100 requested permuted-profile targets |
| `reports/phase12r/min_events_sensitivity.csv` | same | Primary `min_events=1` vs sensitivity `min_events=2` |
| `reports/phase12r/inference_alignment_audit.json` | `python scripts/phase12r_inference_audit.py` | Baseline/model `y_true` mismatch audit |
| `reports/phase12r/manifest.json` | `python scripts/phase12r_manifest.py` | Artifact hashes, input/protocol hashes, git state |
