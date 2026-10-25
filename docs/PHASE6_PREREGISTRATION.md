# Phase 6 preregistration: Causal Response Operator on the Randi atlas

**Status:** frozen before any Phase 6 model was fitted.
**Date:** 2026-10-25.
**Branch:** `research/phase6-causal-atlas`.
**Parent:** `cd5b6f4e496c1014e5941df40638a47c9171a7e2` (Phase 5).
**Author:** NeuroBranch contributors.

This document fixes the primary endpoint, splits, baselines, metrics, uncertainty
procedure and decision rules for the Phase 6 Causal Response Operator (CRO). It is
written before the model is run so that no threshold can be chosen after seeing
the held-out result. Deviations must be logged in `docs/PHASE6_DEVIATIONS.md` and
reported alongside the primary result.

---

## 1. Question

> Can a robust, feature-conditioned, hierarchical low-rank operator predict the
> downstream response to direct optogenetic stimulation of a neuron it has not
> been trained on, in an animal it has not seen, better than anatomy, global
> means or context-only models?

This is a question about **generalisation of measured signal propagation**, not a
claim that calcium responses are synapses, and not a claim of behavioural
causality.

---

## 2. Data and audit

### 2.1 Source

- Randi, Sharma, Dvali & Leifer (2023), *Nature*, DOI
  [10.1038/s41586-023-06683-4](https://doi.org/10.1038/s41586-023-06683-4).
- OSF project `e2syt`, processed `pumpprobe` text exports. The project declares
  no licence (`node_license: null` as of 2026-10-25); raw and row-level data are
  therefore **not redistributed**. Download instructions and hashes are in
  `docs/DATA_PROVENANCE.md` and the Phase 6 manifest.
- Local derived table:
  `reports/phase3/randi_trial_table_phase3.csv.gz` (SHA256 recorded in the
  Phase 6 manifest).

### 2.2 Local coverage (verified)

| Quantity | Value |
|---|---:|
| Source datasets | 110 with usable trials |
| Events | 3,371 |
| Long-form rows | 237,639 |
| Stimulated neurons (targets) | 259 |
| Recorded neurons (responders) | 305 |
| Target–responder pairs | 40,341 |
| Pairs with ≥2 events | 28,000 |
| Pairs with ≥5 events | 15,269 |
| Median events per pair | 3 |
| Median animals per target | 5 |

The paper reports 23,433 measured pairs after its own filtering; the local table
is a **candidate superset** because it retains repeated stimulations, ambiguous
labels and additional measured responders. Phase 6 does not claim exact
reproduction of the paper's pair count. A reconciliation table is a required
output, not a post-hoc excuse.

### 2.3 Endpoint

Primary: `response_mean`, the mean post-stimulus deviation from the pre-stimulus
median, in processed GCaMP units.

Secondary: `response`, the signed peak deviation.

Rationale: both have similar split-half reliability in the local audit (median
pairwise profile correlation ≈ 0.09), but the mean is less sensitive to a single
noisy frame and is the quantity a downstream model should predict. `response` is
reported to show that the result is not an artefact of this choice.

The stimulated neuron's own trace is excluded. Duplicate/ambiguous canonical
labels are excluded, never imputed.

### 2.4 Measurement-error fact

The median split-half correlation of response profiles across repeated
stimulations of the same target in the same animal is **0.09** (Phase 3 audit).
This is the dominant reason a single-trial benchmark can look hopeless. The CRO
therefore models a latent pair effect and treats single-trial deviations as
noise; the primary evaluation is at the **pair level** after aggregating repeated
measurements, with event-level metrics secondary.

---

## 3. Splits

All splits are built from the long-form table before fitting. No test pair,
target or animal contributes to preprocessing statistics, early stopping or
hyperparameter selection.

| Split | Unit held out | Folds | Primary? |
|---|---|---:|---|
| `unseen_targets` | 20% of targets; all their events | 5 | **yes** |
| `unseen_pairs` | 20% of target–responder pairs, stratified by event count | 5 | no |
| `unseen_animals` | 20% of animals; all their events | 5 | no |
| `unseen_targets_and_animals` | both | 5 | no |

- Target folds are stratified by target coverage so that rare targets are not all
  in one fold.
- Pair folds are stratified within animal by `log1p(n_events)`.
- Animal folds are deterministic after sorting animal IDs and hashing the fold
  seed.
- A test pair is excluded from training even if the same target or responder
  appears in training.
- The split manifest stores the exact IDs and a SHA256 of the manifest.

---

## 4. Models

### 4.1 Primary: CRO

For event `e` with target `t`, responder `r`, animal `a`:

```
y_e ~ StudentT(nu=4, mu_e, sigma_e)
mu_e = mu0 + alpha_a + beta' c_e + m_{t,r}
m_{t,r} = b_t + c_r + u_t' v_r + gamma' z_{t,r}
u_t = U_emb[t] * seen_t + MLP_t(feat_t)
v_r = V_emb[r] + MLP_r(feat_r)
```

- `c_e`: pre-activity, pre-SD, log1p(previous interval), previous same target,
  event order. Missing context is median-imputed using train data and flagged.
  `global_delta` is **excluded** because it is a post-stimulus population
  summary; see `docs/PHASE6_DEVIATIONS.md` D1.
- `z_{t,r}`: log1p direct chemical synapses, log1p direct electrical synapses,
  log1p two-hop chemical/electrical counts, target degree, responder degree.
- `feat_t`, `feat_r`: 32-dimensional unsupervised SVD of the connectome
  adjacency (fit on the connectome only, never on responses), plus degree and
  identity-availability indicators.
- `seen_t` is 0 for a held-out target. Its embedding is not trained; its factor
  comes only from anatomical features. This is what makes `unseen_targets` a
  real inductive test rather than a lookup.
- Low rank `K ∈ {4, 8, 16, 32}`, weight decay and MLP width selected on the
  validation animals/targets of the training fold only.
- The likelihood is StudentT to limit the influence of the extreme response
  tails (observed range roughly −5,000 to +11,600 before aggregation).
- Five seeds; reported metric is the mean across seeds, uncertainty is the
  animal-cluster bootstrap.

### 4.2 Baselines (all fitted on train only)

1. `global_mean`.
2. `target_mean` (fallback to global for an unseen target).
3. `responder_mean`.
4. `additive` (`target_mean + responder_mean - global_mean`).
5. `context_only` (ridge on `c_e`).
6. `connectome_ridge` (ridge on `z_{t,r}`).
7. `svd_mf` (rank-16 matrix factorisation of observed pair means; fallback to
   global for unseen targets).
8. `cro_no_features` (embeddings only; diagnostic, not a fair unseen-target
   baseline).
9. `cro_no_connectome` (feature MLP and embeddings but no pair connectome
   features).
10. `oracle_pair_mean` (train pair mean where available; diagnostic ceiling, not
    a deployable model).

`target_mean` and `oracle_pair_mean` are intentionally strong. A model that only
beats `global_mean` has not demonstrated useful structure.

### 4.3 What is not allowed

- No test data in SVD, scaling, imputation or early stopping.
- No target-specific calibration fitted on test targets.
- No hyperparameter search scored on the test fold.
- No pooling of test animal effects into the training animal embedding.
- No use of `response` to select features when the endpoint is `response_mean`.

---

## 5. Metrics

### 5.1 Primary

**Pair-level Spearman correlation** between predicted and observed pair means on
held-out pairs with at least two test events. Spearman is used because the
response scale is heavy-tailed and monotonic prediction is the scientific claim.

Minimum meaningful effect: **Δ Spearman ≥ +0.05** over the strongest deployable
baseline, with a 95% animal-cluster bootstrap CI strictly above zero.

### 5.2 Secondary

- Pair-level MAE and sign accuracy.
- Event-level MAE, RMSE, Spearman and sign accuracy.
- Event-level StudentT negative log-likelihood for probabilistic models; CRPS
  where a predictive distribution is available.
- Calibration: empirical coverage of the 90% predictive interval on held-out
  events, target 0.85–0.95.
- Per-animal and per-target breakdown; report the number of animals/targets
  improved, not only the mean.

### 5.3 Uncertainty

- 1,000 animal-cluster bootstrap resamples for every headline metric.
- For paired model comparisons, resample animals and recompute the difference on
  the same resample.
- Five model seeds are averaged before bootstrapping; seed variance is reported
  separately.
- No p-value is used as the primary evidence; the effect size and CI are.

---

## 6. Decision rules

| Outcome | Interpretation |
|---|---|
| Δ Spearman ≥ +0.05, CI > 0, calibration 0.85–0.95 | **SUPPORTED**: CRO generalises to unseen targets |
| Δ Spearman > 0 but CI includes 0 | **INCONCLUSIVE**: report, do not claim |
| Δ Spearman ≤ 0 or CI < 0 | **NOT SUPPORTED**: report the negative result and the measurement-error analysis |
| CRO wins only on `unseen_pairs`, not `unseen_targets` | **PARTIAL**: matrix completion works, inductive target generalisation does not |
| CRO wins only on event-level, not pair-level | **NOT SUPPORTED** for the latent-operator claim; report as noise-fitting risk |

If the primary fails, Phase 6 still delivers the benchmark, the reliability
analysis and the negative result. It does not tune the endpoint until something
becomes significant.

---

## 7. Secondary Phase 6 analyses (not gate for primary)

1. **Causal-discovery benchmark**: can correlation/Granger/SBTG-style directed
   scores from spontaneous Kato activity recover the Randi pair-effect ranking?
   Evaluated on shared identified neurons with animal-disjoint splits. A negative
   result is expected and reportable.
2. **Cross-dataset transfer**: does the CRO effect matrix improve Kato one-step
   forecasting over persistence/VAR/PCA/GRU and anatomy-only? Only shared,
   high-confidence labels; coverage reported.
3. **Kramer transfer**: does CRO-derived influence improve animal-disjoint,
   event-level turn-direction prediction beyond behaviour/sensory baselines?
   Prior Phase 5 evidence says no; this is a falsification test.
4. **Experiment designer**: rank under-sampled targets by expected information
   gain over competing models. Outputs are hypotheses, not observed effects.

These analyses have their own reported metrics and are not used to rescue a
failed primary.

---

## 8. Epistemic status

- L0: published Randi measurements.
- L1: this reproduction on the local candidate table.
- L2: observational/statistical inference from the benchmark.
- L3: model-dependent prediction for an unmeasured pair.
- L4: wet-lab hypothesis.

No Phase 6 output is L4. Any virtual perturbation remains:

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

---

## 9. Artefacts and reproducibility

Required outputs:

- `reports/phase6/preregistration.json` — machine-readable copy of the frozen
  choices and SHA256 of this document.
- `reports/phase6/cro/splits/*.json` — exact split manifests.
- `reports/phase6/cro/metrics.csv`, `predictions.parquet`, `operator.npz`,
  `uncertainty.npz`.
- `reports/phase6/cro/audit.json` — coverage, reliability, label exclusions.
- `reports/phase6/manifest.json` — git commit, command, environment, input and
  output hashes.

Run:

```bash
uv run python scripts/phase6_randi_cro.py --config configs/phase6/cro.yaml
uv run pytest -q
uv run ruff check .
```

No raw data are committed. No remote push is performed.
