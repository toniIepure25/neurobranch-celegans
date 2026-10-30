# Phase 14A reference audit

**Question:** is the Phase 11 descriptive ceiling `0.560` an artifact of setting
one-event cell variance to zero?

**Answer:** no, under a robust event-variance treatment. The reference is robust
to replacing the one-event zero (`|V1 - V0| = 0.0131 < 0.05`), but it is not a
universal bound: filtering to targets with at least two events changes both the
target population and the estimate.

## Treatments

| Treatment | One-event variance | `n >= 2` variance | Targets | R(22) | Ceiling |
|---|---|---|---:|---:|---:|
| V0 historical | 0 | sample | 126 | 0.3136077 | 0.5600069 |
| V1 primary | `omega^2 / n` | sample | 126 | 0.3005503 | 0.5482247 |
| V1h homogeneous | `omega^2 / n` | `omega^2 / n` | 126 | 0.3138911 | 0.5602599 |
| V2 filter | dropped | sample | 36 | 0.1753713 | 0.4187736 |

`omega^2 = 1.7629` is the median observed within-cell sample variance for cells
with at least two events. The value is on the raw `response_mean` scale.

## Uncertainty

Animal-cluster bootstrap, 2,000 resamples, seed 61:

| Treatment | 2.5% | 97.5% |
|---|---:|---:|
| V0 | 0.5044 | 0.6500 |
| V1 | 0.4902 | 0.6393 |

The bootstrap is conditional on the estimated `omega^2`; it describes animal
sampling variability, not event-variance estimation uncertainty.

## What this changes

- Phase 13's G6 mismatch did not inflate `0.560` by a large amount. The
  one-event zero is a real assumption, but the descriptive reference is stable
  under a robust replacement.
- The `0.560` claim must still name its population: 126 targets at 80%
  responder coverage, Phase 11 estimator, `fold_size=22`.
- The two-event filter is a different estimand. `0.1754` on 36 targets is not a
  corrected version of `0.560`; it is a sparse-target-subset estimate.
- The raw-scale scaled-chi-square plug-in saturated at the grid edge because of
  extreme outliers. It is recorded as a numerical failure (D1), not as a
  competing estimate.

## Evidence

- `reports/phase14/reference_audit.json`
- `reports/phase14/reference_audit_by_target.csv`
- `reports/phase14/reference_audit_bootstrap.csv`
- `research/phase14_protocol.md`, section 3
- `research/phase14_deviations.md`, D1
