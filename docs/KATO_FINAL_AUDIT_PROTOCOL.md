# Kato final reproducibility and scientific audit: protocol

**Branch:** `research/kato-final-reproducibility-audit`
**Starting HEAD:** `358107a7a55470d48f8092b2a2dc7f704dbd37c1`
**Protocol status:** written after the clean rerun was launched but before its
comparison was inspected. It does not claim preregistration.

## 1. Question

This is an audit, not a new modelling phase. It asks whether the latest positive
cross-neuron predictive result is:

1. computationally reproducible from the committed entry point;
2. methodologically comparable to its own-history and ten-factor controls;
3. statistically described without treating overlapping landmarks as independent
   animals;
4. represented in the supervisor handoff without moving from prediction to
   biological mechanism.

The primary numerical object is unchanged: paired H20 PCA-10 energy,

```text
B5_common_k10 energy - B3_population energy
```

where energy is lower-is-better, so a positive value means B3 is better. The
historical protocol's sign notation is audited separately in
[KATO_FINAL_AUDIT_SIGN_CONVENTION.md](KATO_FINAL_AUDIT_SIGN_CONVENTION.md); no
score or threshold is changed in this phase.

## 2. Data and fixed design

- Dataset: Kato et al. (2015), `WT_NoStim`, five recordings only.
- File: `data/raw/kato2015/WT_NoStim.mat`.
- Expected SHA256: `7531811c1718e26a0455f59cb2f8188cf25809278cfea85770dfe03315a0098a`.
- Total frames: 15,662.
- Native dimensions: 103, 127, 123, 118, 121.
- Split: chronological 70/15/15 with a 12-frame purge; train, validation and
  test boundaries are recorded in `preflight.json`.
- H20 landmarks: 175, stride 10, fixed before the audit.
- Representation: one 10-component PCA fitted on training frames only,
  `random_state=17`; the same transform defines the evaluation space and the B5
  common-factor scores.
- Models: B0 persistence, B1m matched own AR (200), B2 own + drift, B4m matched
  own AR (500), B5-k own + `k` common factors (`k=1,3,5,10`; primary `k=10`), B3
  full population VAR, B6 coefficient-ablation diagnostic.
- Estimation: 500-frame causal window for B3/B4m/B5; ridge `1.0` on standardized
  features; unpenalized intercept; common J5 innovations; 100 ensemble members;
  100-step rollout with H20 evaluation.
- Uncertainty: within-session moving-block bootstrap, equal session pooling,
  1,000 draws; primary block 5, sensitivity blocks 3, 10 and 20. Official seed
  for block `b` is `1700 + b`.

No WT_Stim, Nichols, Dunn, Randi, Baskoylu or Atanas data is loaded. No new
dataset, architecture search, hyperparameter optimization on test outcomes, or
neuron-ranking screen is permitted.

## 3. Clean end-to-end reproduction

A separate output directory is used:
`reports/simulator/kato_final_audit/clean_rerun/`. The committed historical
directory is not overwritten.

The only code change to the official entry point is an optional `--out-dir`
argument, which changes the destination only. The analysis code, data, split,
seeds and model configuration are unchanged.

Command:

```powershell
python -X utf8 scripts/kato_common_state_main.py `
  --out-dir reports/simulator/kato_final_audit/clean_rerun
```

The run is detached, with stdout and stderr stored under
`reports/simulator/kato_final_audit/logs/`. It must complete the model stage,
surrogates and synthetic controls; using `kato_common_state_postprocess.py` as a
substitute is not a reproduction.

Deterministic comparison tolerance is `1e-8` absolute for matching float64/CPU
outputs; `1e-6` is retained as the historical G1 tolerance and is not sufficient
to call a clean reproduction. A status is assigned only after the full run exits:

- `FULL_END_TO_END_REPRODUCTION_PASSED`
- `FULL_END_TO_END_REPRODUCTION_PASSED_WITH_TOLERANCE`
- `FULL_END_TO_END_REPRODUCTION_FAILED`
- `FULL_END_TO_END_REPRODUCTION_INCOMPLETE`

## 4. Audit questions and decision rules

### A. Sign convention

Review the committed protocol, code, gate evaluation and commit provenance. Do
not rewrite history. Decide whether the original rule was unambiguous despite a
notation error, or genuinely ambiguous. If the success direction was not frozen
unambiguously before evaluation, G4 may retain its numerical PASS but cannot be
called a fully prespecified confirmatory pass.

### B. Decomposition

Verify the arithmetic `B4m - B3 = (B4m - B5) + (B5 - B3)`. Recompute the shares
under alternative baselines. Record that the shares depend on ordering,
representation, estimation and regularization and are not unique variance
components.

### C. Ten-factor control

Establish whether the evaluation PCA and common-factor PCA are the same transform,
whether both are training-only, how much variance they explain, and whether B5
has access to the same permissible history as B3. Use fitted operators and causal
windows to determine whether B3's residual-source rows contain structure that B5's
diagonal-plus-factor form cannot represent. Do not fit a new model to distinguish
residual information from parameter count; mark that ambiguity unresolved.

### D. Uncertainty

Use already generated paired landmark results. Report blocks 3, 5, 10 and 20,
per-session estimates, leave-one-session-out means, serial dependence, landmark
subsampling, and horizon-specific intervals. Do not select the most favorable
interval. Do not describe 175 overlapping landmarks as independent replicates.

### E. Claims

Audit the five propositions separately:

1. full population history improves an H20 probabilistic score;
2. it contains predictive information beyond the common-factor baseline;
3. there are stable pairwise dynamical interactions;
4. the learned operator represents biological connectivity;
5. mathematical perturbation predicts stimulation outcome.

Only the first two may be positive. Preserve:

```text
PROBABILISTIC CALIBRATION NOT ESTABLISHED
MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.
Biological intervention validation: NOT_EVALUABLE.
```

## 5. Deliverables and stop rule

Eight versioned audit documents, a machine-readable claims ledger, a decision
record, reproduction comparison, bootstrap sensitivity, ten-factor diagnostic,
figures, manifest and validator. Previous handoff artifacts remain unchanged.

The phase stops after the evidence is consolidated for supervisor discussion.
It does not start another modelling phase, rank neurons, or make a biological
causal claim.
