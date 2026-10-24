# Phase 4 supervisor brief

## Bottom line

Phase 4 does **not** rescue the missing neural signal.  The independent 6 s
replication fails its prespecified criterion, and an animal-disjoint
turn-direction benchmark shows no uncertainty-bounded neural increment beyond
behaviour.  This is a clean negative result, not a failed analysis.

## Key numbers

- Replication, WT octanol, 32 animals: 6 s Δ log loss **−0.0057**
  ([−0.0113, +0.0005]), **22/32** animals improved; criterion not met.
- Turn direction, 56 WT animals / 945 events: behaviour AUROC **0.595**;
  +population **0.608**; Δ **+0.0123** ([−0.0166, +0.0425]).
- SAAV: Δ **−0.0042** ([−0.0117, +0.0029]).
- Circuit: Δ **−0.0097** ([−0.0380, +0.0191]).
- Identified: Δ **−0.0176** ([−0.0991, +0.0574]).
- Nonlinear boosting: Δ **+0.0121** ([−0.0275, +0.0481]).
- Largest secondary point estimate: WT diacetyl population **+0.0624**
  ([−0.0040, +0.1241]), exploratory and including zero.
- Calibration is poor (slopes 0.43–0.47); probabilities are not usable as
  absolute turn probabilities.

## Interpretation

- Changing the target from reversal state to turn direction did not reveal a
  neural signal.
- Population, SAAV, circuit, identified and nonlinear representations all
  failed in the primary cohort; “linear underfit” is not a sufficient
  explanation, though the paper's exact RNN was not reproduced.
- Cohort heterogeneity is real but does not produce a robust cross-cohort
  signal: the diacetyl point estimate (+0.062) has a CI including zero, and
  octanol is null.
- The archive lacks trial-level optogenetics; tdc1 is a distribution-shift
  cohort, not an intervention.
- We cannot claim “no neural information.”  We can claim no evidence of
  incremental prospective decodability at this resolution.

## Recommendation

Do not scale this into a 6-ECTS causal/whole-brain intervention claim.  Keep
the 6-ECTS scope to a methods-and-null-result package plus a focused
follow-up: balanced animal-disjoint replication, exact paper alignment/RNN if
the derived CSVs become available, and trial-level perturbation data.

## Deliverables

- `docs/PHASE4_RESULTS.md` and the specialist Phase 4 documents.
- 9-tab dashboard including the turn-direction tab and intervention warning.
- Compact reports and figures under `reports/phase4/`.
- Tests and manifest; no raw data committed.

## What would change the decision

A prespecified animal-disjoint positive increment with a CI excluding zero in
an independent cohort, or a successful exact reproduction of the published RNN
on its intended inputs.  Neither exists in this phase.
