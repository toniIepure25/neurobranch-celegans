# Phase 15 deviations

Protocol frozen 2026-11-01 before the full run. Deviations are recorded here,
including failed diagnostics that were not used.

## D1. BCa diagnostic not reported

A BCa-style diagnostic on the delete-one-animal values was implemented during
the run and then removed. The delete-one-animal values are not a bootstrap
distribution, so percentile calibration on them is not valid; trial values were
narrower than the primary interval and would have been misleading. The primary
delete-one-animal normal interval is unchanged. No gate or decision used BCa.
This is recorded rather than silently dropping the diagnostic.

## D2. Null table written as a derived artifact

The audit computes the 1,000-draw permutation and sign-flip nulls inside
`summary.json`. `scripts/phase15_nulls.py` only re-exposes them as `nulls.json`
for the manifest and manuscript. It does not recompute or alter the draws.

## D3. Heavy-tailed check is a stress test

The 500-replicate Student-t(6) / lognormal-heteroskedastic calibration
(`calibration_heavy.json`) was run after the primary study. It is not a second
primary analysis, is not used to select an interval, and is reported as a
misspecification stress test. Its coverage is 1.000 for all three estimands.

## D4. No external transport evaluation

No independent repeated-stimulation dataset is available locally. Transport
remains `not_evaluable`; the protocol's external-validation gate is not silently
marked as passed.

## D5. Phase 14A result is not re-run

Phase 14A's V1 point estimate is reproduced by the Phase 15 V1 variant. The M4
model is not retuned or re-fit; Phase 14B remains rejected.
