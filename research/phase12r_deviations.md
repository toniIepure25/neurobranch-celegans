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

## D2. Deterministic centered truth instead of per-replicate Monte Carlo (2026-10-30)

The protocol described a Monte Carlo truth for the SBC rank. The implementation
uses the equivalent deterministic Gaussian expression with the centering factor
`1 - 1/R` for both variance components. This avoids a second nested simulation
at every replicate while keeping the same estimand and centering as the fit.
The approximation is checked separately against a small direct simulation in
the test suite. No gate or decision rule changed, and this note was written
before inspecting a Phase 12R gate result.
