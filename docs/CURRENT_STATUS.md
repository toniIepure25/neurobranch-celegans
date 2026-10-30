# Current status

**As of the Phase 12R correction (2026-10-30).** This page is the concise
entry point; the detailed records are linked at the bottom.

## Headline

The only defensible coverage-aware reliability ceiling remains **0.560**. The
hierarchical measurement model gives a pooled point estimate of **0.304** at 22
animals and 80% responder coverage, within 0.05 of the Phase 11 estimate, but
its interval is **not calibrated** and is not published as a replacement.

The model is a measurement/reliability model, not a causal or generative model
of the animal. No target ranking, animal-budget guarantee, virtual-perturbation
claim, or digital-twin claim is supported.

## Phase 12R gate result

| Gate | Requirement | Observed | Status |
|---|---|---:|---|
| SBC coverage | 0.93--0.97 | 0.933 | pass |
| SBC rank uniformity | KS p > 0.01 | 1.43e-06 | **fail** |
| Grid edge mass | < 0.05 | 0.039 | pass |
| Fit integrity | <= 5% failures | 0.000 | pass |
| PPC coverage | >= 0.90 | 0.904 | pass |
| PPC median absolute error | <= 0.10 | 0.207 | **fail** |
| Negative control median | <= 0.05 | 0.120 | **fail** |
| Negative control q95 | <= 0.15 | 0.594 | **fail** |
| Prior sensitivity range | <= 0.05 | 0.0031 | pass |

The decision is `not_adopted_gate_failures`. No threshold, prior, grid, seed, or
filter was tuned after results.

## Two limitations that change interpretation

1. **Variance floor.** 32,427 of 41,086 observed cells (79%) have zero
   within-cell variance and use a `1e-8` floor. The floor is an approximation,
   not a measured variance.
2. **Event-filter sensitivity.** The primary `min_events=1` filter gives 126
   targets and 0.304; the preregistered `min_events=2` sensitivity gives 36
   targets and **0.199**. The point estimate is therefore filter-sensitive.

## Benchmark inference withdrawn

The old Phase 12 animal-cluster benchmark comparison is invalid: baseline and
model `y_true` differ for **1,334 of 1,334** paired rows because rows were
merged by animal/target/neuron aggregates rather than events. There is no
benchmark superiority claim and no “underpowered positive” framing. Leaderboard
values remain descriptive. `paired_cluster_inference` now fails closed on
unaligned truth by default.

A synthetic check of the procedure also found interval coverage of 0.900--0.917
at nominal 0.95 and null rejection of 0.075--0.083, so the inference procedure is
slightly undercovering even before the alignment problem. No positive benchmark
claim is available.

## What remains supported

- Single-animal reliability is below 0.5 (bounded claim).
- Within-animal repeat-stimulation reliability is below 0.5 (bounded claim).
- The coverage-aware pooled ceiling is 0.560 under the Phase 11 estimator.
- Phase 8 model comparisons retain their original statuses; none is a
  generalisation claim.

## Evidence

- [Phase 12R results](PHASE12R_RESULTS.md)
- [Phase 12 results](PHASE12_RESULTS.md) (historical; C10/C11 corrections)
- [Phase 11 results](PHASE11_RESULTS.md)
- [Corrections](CORRECTIONS.md)
- [Claims](CLAIMS.md)
- [Limitations](LIMITATIONS.md)
- `reports/phase12r/summary.json` and the manifest

## Not done

- No new data acquisition or external validation cohort.
- No calibrated interval, target ranking, or animal-budget recommendation.
- No generative model prototype; Phase 12R failed, so the conditional
  generative branch remains a future experiment.
