> **Ceiling numbers in this report are superseded.** Phase 9 and Phase 10
> corrected the reliability estimator and aligned the ceiling to the metric
> (see `docs/CORRECTIONS.md` C6-C7 and `docs/PHASE10_RESULTS.md`). The model
> results themselves are unchanged; only the measurement ceiling they are
> compared against has changed.

> **Superseded ceiling (Phase 9).** The split-half reliability ceiling used in
> this document (about 0.152) is biased downward. The variance-corrected
> single-animal ceiling is 0.473 and the repeat ceiling is 0.493. Percentages
> of the ceiling quoted below are therefore too high. See
> [`CORRECTIONS.md`](CORRECTIONS.md) and [`PHASE9_RESULTS.md`](PHASE9_RESULTS.md).
> The historical numbers are retained for audit.

# Phase 6 results: Causal Response Operator on the Randi atlas

**Date:** 2026-10-25
**Branch:** `research/phase6-causal-atlas`
**Primary run:** `reports/phase6/cro/` (5 folds × 5 seeds)
**Preregistration:** [PHASE6_PREREGISTRATION.md](PHASE6_PREREGISTRATION.md)
**Deviations:** [PHASE6_DEVIATIONS.md](PHASE6_DEVIATIONS.md)

---

## 1. One-paragraph answer

A feature-conditioned hierarchical low-rank operator predicts the downstream
response to stimulation of a **held-out target** substantially better than every
deployable baseline: pair-level Spearman **0.098** versus **0.054** for the
strongest additive/responder-mean baseline. The improvement is positive and its
95% animal-cluster interval excludes zero (Δ = **+0.044**, CI **+0.028 to
+0.059**), but it falls short of the preregistered **+0.05** minimum. Under the
frozen decision rule this is therefore **not a clean SUPPORTED result**. The more
informative result is that the CRO recovers **64% of the split-half reliability
ceiling** for within-target response profiles, versus **11%** for the additive
baseline. It does **not** recover direct anatomical connectivity, and its raw 90%
intervals are too wide (coverage 0.984). These are reported as negative and
limiting findings, not hidden.

---

## 2. Primary benchmark

### 2.1 Setup

- 237,639 event-level rows, 3,371 stimulation events, 110 animals, 259 targets,
  305 responders, 28,000 target-responder pairs with ≥2 events.
- Primary split: `unseen_targets`, 5 folds; all events of a held-out target are
  excluded from training.
- Primary endpoint: `response_mean`; pair-level Spearman after averaging repeated
  events per pair.
- Five seeds per fold; seed predictions averaged before evaluation; uncertainty
  is an animal-cluster bootstrap (300 replicates for the primary comparison).
- All preprocessing, early stopping, scaling and calibration use only training
  and validation rows.

### 2.2 Headline table

| Model | Pair Spearman | 95% CI | Pair MAE | Event MAE | 90% coverage |
|---|---:|---:|---:|---:|---:|
| **CRO** | **0.0984** | **0.0817–0.1114** | 1.716 | 2.769 | 0.984 |
| additive | 0.0545 | 0.0403–0.0671 | 1.789 | 2.823 | — |
| responder_mean | 0.0545 | 0.0403–0.0671 | 1.789 | 2.823 | — |
| context_only | 0.0352 | 0.0219–0.0467 | 1.870 | 2.950 | — |
| global_mean | −0.0039 | −0.0165–0.0082 | 1.710 | 2.759 | — |
| target_mean | −0.0039 | −0.0165–0.0082 | 1.710 | 2.759 | — |
| SVD matrix factorisation | −0.0039 | −0.0165–0.0082 | 1.710 | 2.759 | — |
| connectome ridge | −0.0066 | −0.0195–0.0030 | 1.720 | 2.766 | — |

`target_mean`, `oracle_pair_mean` and `svd_mf` collapse to the global mean on
unseen targets because no pair-level information for those targets exists in
training. They are shown to make that limitation explicit, not as competitive
models.

### 2.3 Paired comparison against baselines

| Baseline | Δ Spearman | 95% CI | Verdict |
|---|---:|---:|---|
| additive / responder_mean | +0.0439 | +0.0282–+0.0592 | positive, below +0.05 gate |
| context_only | +0.0632 | +0.0465–+0.0799 | positive |
| target_mean / global_mean | +0.1023 | +0.0823–+0.1216 | positive |
| connectome_ridge | +0.1050 | +0.0824–+0.1229 | positive |

The pre-registered criterion was **Δ ≥ +0.05 over the strongest deployable
baseline**. The observed +0.0439 does not meet it. The correct decision is
**NOT SUPPORTED under the strict gate**, with a clear positive direction and a
profile-level result that is much stronger than the raw headline suggests.

---

## 3. Why the profile analysis matters

The raw pair-level correlation mixes two quantities: overall response magnitude
and the relative response profile across responders. The scientifically
interesting quantity is whether the CRO predicts **which responders** a new target
will drive.

Split-half reliability of target profiles was estimated by repeatedly splitting
events per pair, correlating the two half-profiles across responders, and applying
Spearman–Brown. On this dataset:

- mean half-profile reliability: **0.063**;
- mean full-profile reliability: **0.152**;
- mean attenuation ceiling (maximum expected model–observation correlation):
  **0.299**.

| Model | Mean profile Spearman | Fraction of ceiling |
|---|---:|---:|
| **CRO** | **0.0944** | **0.637** |
| additive / responder_mean | 0.0438 | 0.107 |
| context_only | 0.0391 | 0.291 |
| connectome ridge | −0.0063 | −0.094 |

The CRO is not “98% accurate.” It captures about **64% of the reliably measurable
target-profile structure**, while the strongest conventional baseline captures
about **11%**. That is the main scientific result of Phase 6.

Caveat: the ceiling is estimated from the same local candidate table; it is a
measurement-error correction, not ground truth. Negative half-split correlations
are truncated to zero before Spearman–Brown, which makes the ceiling conservative
for noisy targets.

---

## 4. Calibration

Raw 90% predictive interval coverage on held-out targets is **0.984**, above the
preregistered 0.85–0.95 band. The intervals are therefore **too wide**, not too
narrow. A validation-only multiplicative calibration function was implemented
(`neurobranch.causal.response_operator.calibration_scale`) and wired into the
runner after the five-seed primary was launched, so the archived five-seed run
does **not** include calibrated intervals. This is logged as D4 in
`PHASE6_DEVIATIONS.md`. No test-set calibration was applied.

The point predictions and rank metrics are unaffected by this interval-width
issue. A future run should report both raw and validation-calibrated coverage.

---

## 5. Anatomical association: a clear negative

Using predicted response magnitude to rank directly connected target–responder
pairs above unconnected pairs, averaged over targets:

| Model | Mean AUROC | Median AUROC |
|---|---:|---:|
| connectome ridge | **0.762** | 0.852 |
| CRO | 0.509 | 0.504 |
| additive | 0.487 | 0.486 |
| context_only | 0.485 | 0.482 |

The CRO is essentially at chance for anatomical edge recovery. A model that
directly uses pair connectivity features is much better. This dissociation is
important: **good response-profile prediction does not imply connectivity
recovery**. Phase 6 therefore does **not** claim causal discovery from the CRO
matrix, and the SBTG-style “causal discovery” comparison is left as an explicit
negative/next experiment rather than a rescued claim.

The label-shuffled null is saved in
`reports/phase6/cro/connectivity/edge_summary.json`.

---

## 6. What was tested and rejected

- **Higher capacity is not better.** Rank 32/48 and hidden 128 lowered one-fold
  pair Spearman relative to rank 16/hidden 64 (0.06 vs 0.089 on fold 0), so the
  smaller model was retained.
- **Responder-specific pair gating overfit.** A gate that let anatomy modulate
  each pair feature per responder reduced one-fold Spearman to 0.053; it was
  removed.
- **Post-stimulus `global_delta` was leakage.** It is measured after stimulation
  and was removed before the full run (D1). The context-only baseline fell from
  0.113 to 0.035, confirming that the earlier apparent context signal was largely
  post-stimulus information.
- **Exact paper reproduction is not claimed.** The local table has 40,341
  candidate pairs versus the paper's 23,433; see
  `reports/phase6/randi_reconciliation.csv`.
- **No cross-dataset transfer is claimed.** No external trial-level optogenetic
  cohort is present; see [PHASE6_EXTERNAL_DATA.md](PHASE6_EXTERNAL_DATA.md).
- **Secondary splits were not completed.** `unseen_pairs`, `unseen_animals` and
  the combined split are implemented and runnable but are not reported here;
  only the preregistered primary split was run to completion (D7).

---

## 7. Decision

| Gate | Result | Status |
|---|---|---|
| Δ ≥ +0.05 over strongest deployable baseline | +0.0439 | **not met** |
| 95% CI strictly above zero | +0.0282 to +0.0592 | met |
| Calibration 0.85–0.95 | 0.984 raw | **not met** |
| Beats additive/responder mean | +0.0439 | met |
| Profile recovery above baseline | 0.637 vs 0.107 of ceiling | strong positive |
| Anatomical edge recovery | 0.509 AUROC | negative |

**Overall:** the strict preregistered claim is **not supported**, because the
effect-size gate and calibration gate were missed. The result is nevertheless a
substantive positive finding about **response-profile generalisation to unseen
targets**, with an explicit negative on connectivity recovery and no causal or
behavioural claim.

Any virtual perturbation derived from this operator remains:

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

---

## 8. Artefacts

| Artefact | Path |
|---|---|
| Split manifests | `reports/phase6/cro/splits/unseen_targets.json` |
| Metrics (primary table) | `reports/phase6/cro/metrics.csv` |
| Metrics by fold | `reports/phase6/cro/metrics_by_fold.csv` |
| Predictions | `reports/phase6/cro/predictions.parquet` |
| Operator matrix | `reports/phase6/cro/operator.npz` |
| Uncertainty | `reports/phase6/cro/uncertainty.npz` |
| Audit | `reports/phase6/cro/audit.json` |
| Reliability | `reports/phase6/cro/reliability_by_target.csv` |
| Profile summary | `reports/phase6/cro/profile_summary.csv` |
| Connectivity | `reports/phase6/cro/connectivity/edge_metrics.csv` |
| Experiment candidates | `reports/phase6/design/target_candidates.csv` |
| Reconciliation | `reports/phase6/randi_reconciliation.csv` |
| Manifest | `reports/phase6/manifest.json` |

Reproduce:

```bash
python scripts/phase6_randi_cro.py --config configs/phase6/cro.yaml
python scripts/phase6_design.py
python scripts/phase6_manifest.py
python -m pytest -q
python -m ruff check .
```

The checked-in `configs/phase6/cro.yaml` requests 5 seeds and 12 epochs, matching
the archived run. The 12-epoch budget is a compute deviation from the
preregistered 45-epoch ceiling; it was fixed before the run and is recorded in
`audit.json`, not selected on the test fold.
