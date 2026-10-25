# Phase 6 deviations and leakage corrections

This file records every change to the preregistered Phase 6 protocol. A
deviation is not automatically a problem; hiding one would be.

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.** No deviation
> below upgrades a model prediction into an interventional claim.

## D1 — `global_delta` excluded from deployable context

**Date:** 2026-10-25, after a 20,000-row smoke test and before the full
preregistered run.

**Observed:** `global_delta` is the median post-stimulus peak across the other
recorded neurons in the same event, computed by
`neurobranch.data.randi.extract_trials`. It is measured **after** stimulation.
A context-only ridge using it reached pair-level Spearman 0.113 on the smoke
subset, while the same model without it is not a prospective baseline.

**Decision:** remove `global_delta` from `CONTEXT_COLUMNS` and from the CRO's
deployable inputs. It is not a pre-stimulus covariate. The Phase 6 benchmark now
uses only pre-activity, pre-SD, previous-stimulation interval, previous-target
flag and event order.

**Impact:** this makes the benchmark harder and removes a likely leakage path.
No primary threshold or split was changed. The smoke result is not reported as a
Phase 6 result.

**Follow-up:** if a post-stimulus population summary is scientifically useful, it
must be labelled an **oracle context** and reported separately, never as a
prospective predictor.

## D2 — Target folds use validation targets as unseen too

**Date:** 2026-10-25, before the full run.

The first implementation marked only test targets as unseen when constructing
target embeddings. Validation targets were excluded from training rows but their
lookup embeddings were still used as if trained. The corrected code marks any
target absent from the training rows (validation or test) as unseen, so early
stopping evaluates the same inductive regime as the test fold. This is a
correctness fix, not a threshold change.

## D3 — CRO interval scale is a calibration target, not a claim

The smoke run's 90% interval coverage was 0.983, i.e. conservative. Phase 6
therefore reports calibration as a secondary criterion. A validation-only
multiplicative calibration (`calibration_scale`) was implemented and wired into
the runner, but the archived five-seed run started before that change; its
coverage is **raw** (0.984). No test-set calibration was applied.

## D4 — Five-seed run used 12 epochs, not the 45-epoch ceiling

The preregistered ceiling was 45 epochs. The archived five-seed run and the
checked-in `configs/phase6/cro.yaml` use 12 epochs per fit for tractability,
decided before the run and recorded in `reports/phase6/cro/audit.json`. Early stopping was not
triggered in most folds; the epoch budget is therefore a compute deviation, not a
test-set hyperparameter search. It is reported in the results and must not be
described as the exact preregistered schedule.

## D5 — Some preregistered diagnostics were not run

`cro_no_features` and `cro_no_connectome` were not run as separate five-fold
models. A capacity/architecture ablation was run on fold 0 (rank 4/8/16/32,
hidden 32/64/128, a responder-specific pair gate); rank 16/hidden 64 was retained
and the gate was removed. This is a partial ablation, not the full preregistered
baseline set, and is reported as such.

## D7 — Secondary split runs were not completed

The split builder implements `unseen_targets`, `unseen_pairs`, `unseen_animals`
and `unseen_targets_and_animals`. Only the preregistered **primary**
`unseen_targets` split was run to completion in this checkout. The secondary
splits remain available through `--splits`; no result is reported for them, and
none is used to qualify the primary claim. This is a scope limitation, not a
post-hoc exclusion.

## D6 — Causal-discovery and cross-dataset secondaries

The connectome-association benchmark was run and is negative for the CRO. A
literal correlation/Granger/SBTG comparison on spontaneous Kato activity was not
run because the endpoint and timescale differ; it is not silently substituted
with a weaker analysis. No external optogenetic cohort is present, so no transfer
claim is made. The contract and audit are implemented in
`neurobranch.benchmark.external`.
