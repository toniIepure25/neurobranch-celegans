# Phase 7 preregistration — Active Discrepancy Atlas

**Frozen:** 2026-10-25, before the full active run.  
**Branch:** `research/phase7-active-probing`, based on `7532ea1`.  
**Status:** confirmatory benchmark with an exploratory discrepancy readout.

**Warning:** MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.

## 1. Data and split

- Input trial table: `reports/phase3/randi_trial_table_phase3.csv.gz` (237,639 rows, 3,371 events, 110 animals, 259 targets).
- Response: `response_mean`; pairs require at least 2 events.
- Outer split: `unseen_targets`, 5 folds, 60% train / 20% validation / 20% test, whole targets held out.
- Outer test targets are frozen before any acquisition decision and are never in the candidate pool.
- Outer validation targets are used for early stopping and conformal calibration; they are never acquired.
- Seed set: 20% of all targets, stratified across target animal-coverage bins, identical for every strategy within a block.
- Candidate pool: remaining targets in the outer-training split.
- Budget points: 20%, 30%, 40%, 50% of all targets. Primary budget: **40%**.
- Reference: all outer-training targets (60% of all targets) with 5 model seeds; this is the full-atlas reference, not an acquisition strategy.
- Acquisition seeds: 101, 202. Model seeds for acquisition: 17, 23. Final reference seeds: 17, 23, 31, 41, 53.
- Target acquisition is **batched**: up to 8 targets are selected per batch; the model is refit once per budget point, not after every batch. This is a compute-honest batch design and is identical across strategies.

## 2. Model

Base CRO is unchanged from Phase 6: rank 16, hidden 64, StudentT likelihood, target/responder feature MLPs, low-rank interaction, context coefficients. No hyperparameter search on outer test.

An optional state-conditioned ablation (`use_state=True`) appends 16 PCA components of median pre-stimulus population activity, with the PCA fitted only on outer-training events. It is not part of the primary H1 test; it is a separate H3 ablation.

## 3. Acquisition strategies

All strategies receive the same seed set, candidate pool, model seeds and batch budget.

1. `random`: uniform sampling without replacement.
2. `uncertainty`: mean ensemble epistemic standard deviation of the predicted response profile.
3. `effect`: mean absolute predicted response.
4. `diversity`: farthest-point / max-min distance in standardised target anatomy features.
5. `discrepancy`: predicted mean absolute residual of a simple additive anatomical prior, extrapolated from acquired targets with a ridge on target anatomy features.
6. `hybrid`: `z(uncertainty) + 0.5·z(discrepancy) + 0.25·z(diversity)`, with robust standardisation over the candidate pool and greedy batch diversity updating.

The anatomical prior for discrepancy is the target+responder mean model, not the CRO. This avoids using the model’s own residual as both score and target.

## 4. Hypotheses and gates

### H1 (primary)
At the 40% budget, `hybrid` minus `random` on outer-test pair Spearman is **≥ +0.02 absolute**, with a 95% paired block-bootstrap CI strictly above 0.

Decision: if the gate is met, active sample-efficiency is supported. If not, H1 is **NOT SUPPORTED** regardless of other strategies.

### H2 (secondary, exploratory)
`discrepancy` ranks anatomy-unexplained test pairs better than `uncertainty` by **≥ +0.05 AUROC**. Falsified by a smaller difference or AUROC ≤ 0.5.

### H3 (conditional)
A state-conditioned CRO at full outer-training data beats the base CRO by **≥ +0.01** pair Spearman with CI above 0. If not, the base CRO remains canonical and the ablation is reported as negative.

### H4 (secondary)
Active acquisition improves direct-edge AUROC over random. Expected to be weak or negative on the basis of Phase 6.

## 5. Endpoints

Primary:
- pair-level Spearman on outer test, aggregated over pairs with ≥ 2 events;
- target-profile Spearman, normalised by the split-half attenuation ceiling.

Secondary:
- 90% interval coverage, raw and conformal-calibrated;
- interval score;
- median targets to 90% of the full-atlas pair Spearman;
- direct-edge AUROC and anatomy-unexplained AUROC;
- acquisition cost in animals and events.

## 6. Statistics

- Paired block bootstrap over fold × acquisition seed for strategy deltas.
- Animal-cluster bootstrap for the final prediction intervals when prediction rows are available.
- Primary H1 is one preregistered contrast; secondary strategies use Benjamini–Hochberg where a family is declared.
- No test residual is used for calibration or model selection.
- Missing or non-finite metrics are reported, not silently dropped.

## 7. Calibration

Split conformal calibration is fitted on outer-validation targets at 90%, with the empirical quantile of absolute normalised residuals. The same quantile is applied to outer test. Raw intervals use a StudentT nominal width for comparison. If validation is too small, the scale falls back to 1.0 and the fallback is logged.

## 8. Negative-result policy

A negative H1 is a valid Phase 7 result. The project will not move the gate, change the primary budget after seeing results, or promote a secondary strategy to primary without labelling it exploratory.

## 9. Non-claims

No wet-lab closed loop, no causal graph, no cross-dataset transfer claim, no foundation-model claim, no exact reproduction of OPhELIA or SBTG.
