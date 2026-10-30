# Phase 12R deviations

## D1. Pre-run correction of the protocol rationale (2026-10-30)

Before any Phase 12R analysis result was produced, direct verification showed
that the suspected sampling-variance loss was a false alarm:
`fit_hierarchical_cells()` consumes per-cell summaries directly and preserves
`var / n`. The protocol's Section 1 was corrected to remove that claim. The
known-variance regression test remains as a guard.

No gate, parameter, seed, or decision rule was changed. The real alignment
target remains the double centering/standardization of simulated templates and
the mismatch between that transformation and the uncentered SBC truth formula.
The protocol hash was updated in the same commit as this note.

This deviation is recorded before inspecting any Phase 12R SBC, PPC, negative
control, or gate result.
