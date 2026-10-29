# Phase 11 deviations

The analysis plan was written before the sensitivity run. These deviations are
recorded because the results changed the implementation, not the gates.

## D1 — Bootstrap diagnostic added after the interval failed

The plan said to report the two-way percentile interval. The first real-data run
produced a point estimate (0.314) outside its own interval ([0.497, 0.583]).
A diagnostic (`bootstrap_mean`, `bias_vs_point`, `interval_status`) was added so
the failure is machine-readable rather than buried in a table. The interval gate
itself was not relaxed; the interval is marked failed.

## D2 — Synthetic coverage gate treated as conservative, not calibrated

The pre-declared synthetic gate was coverage in [0.93, 0.97]. The observed
coverage is 1.000 with a mean width of 0.185. This is over-coverage, not
anti-conservative, so it is reported as **conservative** and the width is
reported. It is not used to certify the real-data interval, which failed its own
diagnostic.

## D3 — Benchmark-card ceiling is the square root

`coverage_sensitivity.csv` reports reliability; the card reports the ceiling
(`sqrt(reliability)`). This is a unit clarification, not a change of estimand.

## D4 — Shrinkage gate undefined rather than numerically negative

At coverage 0.8 the shrunk values are constant and the half-split correlation is
undefined. The gate is recorded as failed; no alternative shrinkage strength was
tried after seeing this.

## D5 — No new literature search

The coverage audit is a re-analysis of the existing Randi atlas. No new
literature search was performed, and the Phase 10 evidence table is reused
without upgrading its review type.
