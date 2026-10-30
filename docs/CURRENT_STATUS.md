# Current status

**As of the Phase 13 measurement-model result (2026-10-30).** This page is the
concise entry point; detailed records are linked at the bottom.

## Headline

The only defensible coverage-aware reliability ceiling remains **0.560**. Phase
13 tested whether two known defects in the Phase 12R hierarchical model could be
repaired: animal centering that left the noise variance unchanged, and a `1e-8`
floor applied to 79% of observed cells. The answer is **no**.

- Correcting centering only (M1) gives pooled **R(22) = 0.3138** (ceiling
  0.5602), but M1 still fails SBC rank and retains the floor.
- Replacing the floor with an event-level variance model (M2) gives pooled
  **R(22) = 0.2003** (ceiling 0.4475). This shows the old 0.304 was inflated by
  the floor, but M2 fails six frozen gates, including both null controls.
- Decision: **`not_adopted_gate_failures`**. No calibrated interval, target
  ranking, animal-budget guarantee, causal effect, or digital-twin claim is
  supported.

The model is a measurement/reliability model, not a causal or generative model
of the animal.

## Phase 13 gate result

| Gate | Requirement | M2 observed | Status |
|---|---|---:|---|
| G1a SBC coverage | 0.93--0.97 | 0.9244 | **fail** |
| G1b SBC rank | KS p > 0.01 | 0.00306 | **fail** |
| G1c grid edge | < 0.05 | 0.0121 | pass |
| G1d fit integrity | <= 5% failures | 0 / 1,800 | pass |
| G1e one-event stratum | 0.90--0.98 | 0.9231 | pass |
| G2 held-out PPC | coverage >= 0.90; MAE <= 0.10 | not run | **fail** |
| G3 negative control | median <= 0.05; q95 <= 0.15 | 0.1485; 0.4680 | **fail** |
| G3b sign-flip | median <= 0.05; q95 <= 0.15 | 0.1341; 0.5796 | **fail** |
| G4 prior sensitivity | range <= 0.05 | 0.00661 | pass |
| G5 event-variance recovery | 0.90--0.98 | 0.9356 | pass |
| G6 Phase 11 agreement | within 0.05 of 0.314 | 0.2003 (0.1137 away) | **fail** |

The PPC was not run after the null controls failed; it is reported as `not_run`
and counted as a failure, not silently omitted. No threshold, prior, grid, seed,
or filter was tuned after seeing the full-run results.

## The main diagnosis

The Phase 12R estimate was sensitive to the one-event variance treatment, not
mainly to the centering algebra. M1 changes the pooled estimate by less than
0.01; M2 changes it by 0.11. Under within-animal responder permutation, M2 still
reports median R(22) = 0.1485, so the model manufactures apparent between-target
reliability from sampling noise. That positive null bias means the real-data
value 0.2003 cannot be interpreted as a calibrated reliability, even though it is
lower than 0.304.

## Benchmark inference withdrawn

The old Phase 12 animal-cluster benchmark comparison remains invalid: baseline
and model `y_true` differed for **1,334 of 1,334** paired rows because rows were
merged by animal/target/neuron aggregates rather than events. There is no
benchmark superiority claim and no “underpowered positive” framing. Leaderboard
values remain descriptive. `paired_cluster_inference` now fails closed on
unaligned truth by default, with event-keyed pairing and BCa intervals.

A synthetic check of the repaired procedure found interval coverage of
0.900--0.917 at nominal 0.95 and null rejection of 0.075--0.083, so it is still
slightly undercovering before the alignment problem. No positive benchmark claim
is available.

## What remains supported

- Single-animal reliability is below 0.5 (bounded claim).
- Within-animal repeat-stimulation reliability is below 0.5 (bounded claim).
- The coverage-aware pooled ceiling is 0.560 under the Phase 11 estimator.
- Phase 8 model comparisons retain their original statuses; none is a
  generalisation claim.

## Evidence

- [Phase 13 results](PHASE13_RESULTS.md)
- [Phase 12R results](PHASE12R_RESULTS.md)
- [Phase 12 results](PHASE12_RESULTS.md) (historical; C10/C11 corrections)
- [Phase 11 results](PHASE11_RESULTS.md)
- [Corrections](CORRECTIONS.md)
- [Claims](CLAIMS.md)
- [Limitations](LIMITATIONS.md)
- `reports/phase13/summary.json`, `sbc.json`, and the manifest

## Not done

- No new data acquisition or external validation cohort.
- No calibrated interval, target ranking, or animal-budget recommendation.
- No valid held-out PPC for M2 (protocol deviation D4).
- No spike-and-slab/zero-inflated between-target model; that is the next
  experiment, not a Phase 13 retry.
- No generative model prototype; the conditional generative branch remains a
  future experiment.
