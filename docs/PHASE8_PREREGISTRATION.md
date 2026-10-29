# Phase 8 preregistration — Adaptive Perturbation Twin (APT)

**Status:** frozen before the full test run.  
**Frozen:** 2026-10-26 (Europe/Bucharest).  
**Data:** Randi et al. 2023 *C. elegans* whole-brain calcium imaging, Phase 3 trial table
`reports/phase3/randi_trial_table_phase3.csv.gz` (3,371 events, 110 animals).  
**Code:** `src/neurobranch/twin/`.  
**Warning:** **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**
No wet-lab loop, no causal graph, and no validated digital twin is claimed.

## Question

Phase 6/7 asked whether a response operator generalises to unseen **targets**.
Phase 8 asks a different, animal-level question:

> Given a new animal and a small, target-disjoint support set of observed
> optogenetic events, can a history-aware, time-resolved twin predict that
> animal's responses to held-out targets better than a static response operator,
> and can active support selection improve that calibration?

The experimental unit is the **animal × support-set episode**, not the target.

## Primary split and leakage control

- Primary split: `unseen_animals`.
- Split seed: **41**. Development smoke runs used 17, 29 and 43; an aborted
  partial full run used 37 and is reported as an aborted attempt. Seed 41 was
  never used for development, model selection, or any completed test evaluation.
- Within each held-out animal, targets are split into disjoint support and test
  sets (target-disjoint and event-disjoint).
- Model fitting uses training animals only. Support outcomes of the held-out
  animal are observed at inference; test outcomes are never used for fitting,
  feature scaling, calibration, early stopping, or support selection.
- Scalers, target/responder priors, support-profile scalers, and interval
  calibration are fitted on training/validation animals only.
- The CRO baseline is trained on the same training event roles and evaluated on
  the same test event/responder cells.

The runner writes `audit.json` with per-fold animal disjointness and
support/test overlap checks. Any overlap is a failed run, not a result.

## Hypotheses and frozen gates

All deltas are paired at the same event/responder cells. Uncertainty is a
1,000-replicate animal-cluster bootstrap. The primary metric is pair-level
Spearman correlation (the Phase 6/7 metric family).

| ID | Comparison | Gate |
|---|---|---|
| **H1 primary** | APT(k=0) vs static Phase 6 CRO | Δ pair Spearman ≥ **+0.010**, 95% CI low > 0, **and** trajectory Pearson vs a per-responder training-template baseline ≥ **+0.050** |
| **H2 secondary** | APT(k=8, random support) vs APT(k=0) | Δ ≥ **+0.010**, CI low > 0 |
| **H3 secondary** | APT(k=8, `eig` support) vs APT(k=8, random support) | Δ ≥ **+0.010**, CI low > 0; confirmatory only if H2 is supported |
| **H4 exploratory** | 3-event pseudo-prospective sequence rollout | trajectory Pearson > persistence by **+0.050**, and 90% coverage in **0.80–0.95** |
| **H5 exploratory** | history features vs no-history | Δ ≥ **+0.005**, CI low > 0 (reported only if run) |

H2/H3 are Holm-corrected for two secondary tests when interpreted
confirmatorily. H1 is the single primary gate and is not rescued by secondary
results. A null result is reported with its confidence interval and the
observed variance components; it is not interpreted as evidence of absence.

## Design

- 5 folds × 3 model seeds `[17, 23, 31]` × 2 acquisition seeds `[101, 202]`.
- Support budgets `[0, 2, 4, 8, 16]`; primary budget **k=8**.
- Acquisition strategies: `random`, `uncertainty`, `diversity`, `eig` (deep-
  ensemble disagreement proxy), `hybrid`.
- Full budget curves are evaluated for the random arm; all strategies are
  evaluated at k=8, which keeps the row-level audit tractable.
- Twin: rank 16, hidden 64, 8 temporal basis functions, 16-dimensional support
  profile, StudentT likelihood (df=5), 30 epochs, AdamW, early stopping on
  validation animals.
- Baselines: static CRO (Phase 6), global train mean, per-responder trajectory
  template, and persistence in the sequence readout.

## Selection and stopping

- Model/hyperparameter choices were frozen before seed 37 was evaluated.
- Early stopping uses validation animals only.
- Support selection only sees the animal's support pool, never test events.
- No optional stopping: all five folds and all three model seeds are run.
- Failed folds or NaNs are reported, not silently dropped.

## Known limitations fixed in advance

1. The animal-effect variance in Phase 3 pair means is small (η² ≈ 0.00054),
   so H2 may be underpowered; a null H2 is a planned, interpretable outcome.
2. The atlas has no same-task external optogenetic cohort, so this is not a
   cross-dataset transfer claim.
3. Previous-event responses in H4 are observed sequentially, not imagined; H4
   is a pseudo-prospective rollout, not a closed-loop experiment.
4. `eig` is an ensemble-disagreement acquisition proxy, not a formal posterior
   expected-information-gain guarantee.
5. The model is a virtual perturbation model. It does not establish causality,
   and predictions outside the Randi atlas support are unsupported.

## Reproduction

```bash
python scripts/phase8_run.py --config configs/phase8/apt.yaml
python scripts/phase8_readout.py
```

Outputs: `reports/phase8/summary.json`, `metrics.csv`, `predictions.parquet`,
`selection_log.jsonl`, `audit.json`, `models/`, `manifest.json`.
