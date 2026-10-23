# Phase 2 baseline audit

**Recorded:** 2026-10-23  
**Repository:** `D:\ComputaCenter\NeuroBranch\neurobranch-celegans`  
**Branch created for Phase 2:** `research/phase2-intervention-validation`

This document records the state that Phase 2 inherited. It is intentionally
descriptive, not a claim that the inherited analyses were correct.

## Git state at Phase 2 start

| Item | Value |
|---|---|
| Branch before Phase 2 | `master` |
| Branch created | `research/phase2-intervention-validation` |
| Starting commit | `5b32c9cbe2c8ee9593193253983128afbb23125f` (`5b32c9c`) |
| Parent history | `bfa175e`, `45c71e8`, `3101f2a`, `f64e761` |
| Working tree before Phase 2 | clean |
| Remote | none configured (`git remote -v` empty) |
| Tracked files | 561 |
| Python version | 3.13.5 |

The initial five commits were preserved. No reset, squash, rebase, or force-push
was used. The Phase 2 work is additive.

## Inherited verification

Commands run from the repository root before any Phase 2 edits:

```text
python -m pytest -q
....................... [100%]
23 passed in 5.36s

python -m ruff check src tests app
All checks passed!
```

The previously reported 23 passing tests and clean lint state were therefore
reproduced independently. The starting test suite has 6 test files: three unit
modules, one integration module, and two scientific/sanity modules.

## Inherited data and artifacts actually present

| Asset | Observed state |
|---|---|
| Kato 2015 `WT_NoStim.mat` | present at `data/raw/kato2015/WT_NoStim.mat`; SHA256 `7531811c1718e26a0455f59cb2f8188cf25809278cfea85770dfe03315a0098a` |
| Blocked suite | `artifacts/runs/blocked_suite/suite_metrics.csv`, 105 data rows |
| Forward-stress suite | `artifacts/runs/forward_stress/suite_metrics.csv`, 35 data rows |
| NC-MCM reproduction | `artifacts/reproduction_ncmcm.json`, `.csv` |
| BunDLe official evaluation | `artifacts/reproduction_bundlenet_official.json` |
| BunDLeLite | `artifacts/reproduction_bundlenet_lite.json`, `artifacts/reproduction/bundlenet_lite_worm0.npz` |
| Branching | `artifacts/branching_worm0.json` |
| Intervention scan | `artifacts/intervention_search_worm0.csv/.json` |
| Random controls | `artifacts/intervention_random_controls_worm0.csv/.json` |
| Figures | 8 PNG files plus `figures/bundlenet_latent_3d.html` |
| Reports | `reports/results_summary.json`, `reports/epistemic_manifest.json`, `reports/posthoc_git_state.json`, aggregate tables |
| Documentation | 14 Markdown documents before Phase 2 |

Raw Kato data and external repositories are not tracked. Per-run `model.pt` and
`test_predictions.npz` files are ignored. The result tables and small JSON/NPZ
artifacts are tracked.

## What was not accepted at face value

The Phase 2 work treats the following inherited claims as hypotheses requiring
independent checks:

1. **Branching p = 1.0.** The implementation permutes outcome labels across
   candidate frames and uses a one-sided `observed >= null` test. That is a
   valid directional test for *excess* local diversity, but it does not rule out
   meaningful temporal structure, nor does it test the opposite direction or
   preserve autocorrelation. It must be separated from a claim that "no
   branching exists".
2. **Intervention specificity.** The inherited control design samples 19
   non-candidate cells, which is an improvement over candidate-only controls,
   but the candidate set was literature-motivated after inspection of the same
   worm. Winner's-curse, model-seed, amplitude, architecture, and selection-
   adjusted uncertainty were not jointly evaluated.
3. **BunDLe-Net.** The official kNN is transductive/leaky, while BunDLeLite is a
   partial reimplementation with a causal filter. Neither is a faithful
   held-out reproduction of the published method. The two numbers must not be
   presented as "official vs our better method" without qualification.
4. **Blocked vs prospective evaluation.** The blocked suite is interpolation.
   The forward suite is the only forecast-like negative control and it fails on
   worms 0–1. This distinction is preserved.
5. **NC-MCM.** Only the Markov surrogate test and a decoder were re-run; the
   clustering/training pipeline was not reproduced. Agreement of p-values does
   not validate the upstream model-selection procedure.

## Phase 2 gates

| Gate | Status at start |
|---|---|
| A. Scientific audit | not yet performed |
| B. Experimental intervention data | not present |
| C. Scientific evaluation | inherited only |
| D. Simulation | inherited, model-dependent |
| E. Application | inherited 6-tab app, AppTest previously passed |
| F. Git | clean at starting commit; branch created |

## Next actions recorded at baseline

1. Audit branching test direction and null construction; add synthetic
   ground-truth tests and a corrected report.
2. Audit intervention specificity across model seeds/architectures and
   amplitudes; add selection-aware summaries.
3. Acquire and parse the Randi et al. (2023) optogenetic propagation data.
4. Build an experimental-response benchmark with held-out animals/stimulated
   neurons and simple baselines.
5. Extend the app and reports only after the above evidence exists.

## Phase 2 execution status (appended after completion)

| Gate | Final status | Evidence |
|---|---|---|
| A. Scientific audit | ✅ completed | corrected branching, intervention, behaviour, BunDLe reports |
| B. Experimental intervention data | ✅ acquired and parsed | Randi 2023, 110 animals, 3,371 valid events, verified hash |
| C. Scientific evaluation | ✅ external benchmark run | no positive transfer; see `INTERVENTION_GENERALIZATION_RESULTS.md` |
| D. Simulation | ✅ specificity audited | 6 models, 26 cells, 19 controls/model; nonspecific |
| E. Application | ✅ 7 tabs, AppTest passes | optogenetic tab added |
| F. Git | ✅ additive commits on Phase 2 branch | no reset/squash/force-push |

The baseline itself is unchanged. The Phase 2 branch does not rewrite the
starting commit or the Phase 1 reports.
