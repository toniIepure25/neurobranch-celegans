# Phase 14 deviations

This file records deviations from `research/phase14_protocol.md` before the full
run and before interpretation.

## D1 - robust event variance for the reference audit

**Recorded:** 2026-10-31, before the full Phase 14 run.

**Protocol version amended:** section 3 of `research/phase14_protocol.md`.

**Reason:** A numerical pilot of the pre-registered scaled-chi-square /
HalfNormal(1) plug-in on the raw response scale returned `omega^2` at the grid
upper bound (`100`). The raw cell-variance distribution is extremely
heavy-tailed (99th percentile `> 80`, maximum `> 10^6`), so the plug-in is
dominated by a handful of outlier cells and is not a valid central estimate of
the typical event variance. The pilot was used only to detect a numerical
failure, not to choose a result.

**Change:** the primary reference audit uses the median of observed within-cell
sample variances for cells with `n >= 2` and `s^2 > 0`. V1h (homogeneous event
variance) and V2 (`min_events=2`) remain pre-registered sensitivities. The
scaled-chi-square plug-in and a 99% winsorized-mean variant are reported as
outlier-sensitivity diagnostics only and do not enter the A2 decision.

**Integrity:** V0 must still reproduce `0.3136077301426884` and
`0.5600069018705827` within `1e-6`. No model prior, threshold, seed, filter, or
sample size was changed.

## Rules

- A pilot may detect runtime, numerical, or implementation bugs only.
- A pilot result may not choose a prior, threshold, seed, filter, or grid.
- If a pre-registered computation is infeasible, amend and re-freeze the protocol
  before the full run, with the exact change and reason.
- A missing gate is a failure, not an omission.
