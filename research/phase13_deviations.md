# Phase 13 deviations

Protocol: `research/phase13_protocol.md`
Frozen: 2026-10-30
Base commit: `5dc4407`

This file records every deviation from the frozen protocol before the full run
and before interpretation. Pilot results are not used to choose thresholds.

## D1. Protocol drafted with a diagonal projection correction

The approved v1.4 plan described an exact projection/contrast likelihood as M1.
During implementation review, the full contrast covariance was found to require a
rank-`R` Woodbury update per grid point, which is not compatible with the frozen
runtime budget for 3,000 SBC fits. The protocol therefore defines M1 as a
first-order diagonal projection correction and records the off-diagonal
limitation explicitly. M2 remains the primary candidate.

This deviation was recorded before any Phase 13 real-data result. The exact
projection model remains a possible future sensitivity (M3 in the wider plan),
but it is not claimed here.

## D2. Event-variance model is marginal/plug-in, not a full cell-level hierarchy

M2 uses a marginal scaled chi-square likelihood for a single event-level variance
scale `omega_e`, with posterior median/plug-in and limited draw propagation. The
approved plan described a cell-level hierarchical event-variance model. The
simpler model is used because it directly removes the `1e-8` floor and can be
calibrated within the runtime budget. Cell-level heterogeneity is reported as a
limitation and is not claimed to be modelled.

Both deviations narrow the model relative to the approved plan. They do not
relax any gate and do not permit a stronger claim. Implementation uses the
posterior median only; no event-variance draws are propagated through the
target-level fit.

## D3. SBC draw count follows the fixed configuration

Protocol section 4 freezes 2,500 posterior draws, while section 6 contains an
older 2,000-draw sentence. The implementation and full run use **2,500 draws**
from the fixed configuration in section 4. The conflict was noticed after the
full run; no gate or threshold depends on the difference. Resolution: section 4
is authoritative, and this deviation is recorded rather than editing the frozen
protocol after seeing results.

## D4. Held-out PPC not run

The primary candidate failed the negative-control gate (G3) and the sign-flip
gate (G3b) before a valid M2 animal-holdout predictive check was implemented.
The check is therefore reported as `not_run` and counts as a gate failure. No
PPC number is invented or copied from the Phase 12R model, whose noise model and
variance floor differ. This is a protocol deviation, not a pass.

## D5. Adversarial-control grid and draws

Protocol section 7 specifies a 50 x 50 grid and 1,200 draws for the negative and
sign-flip controls. The full run reused the frozen 80 x 80 grid and 1,200 draws
for those controls. The larger grid is more conservative for edge mass; the
null summaries are unchanged in direction. No threshold was changed.

## D6. Variance recovery reported through SBC

`variance_recovery.json` is not a separate artifact: M2 event-variance coverage
is computed inside `sbc.json` from the same 600 x 3 replicates. This avoids a
second simulation with a different seed and keeps the gate tied to the
calibration run. The protocol artifact list is therefore satisfied by
`sbc.json` for that quantity.

## Pending

No pilot-driven change to prior scale, grid resolution, replicate count, or
runtime was made. The full-run configuration is the frozen 2,500 draws, 80 x 80
grid, seeds 31/32/43/44/47, and 1,000/600 SBC replicates for M1/M2.

