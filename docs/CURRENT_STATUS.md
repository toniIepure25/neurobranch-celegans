# Current status

**As of the Phase 14 measurement-model result (2026-10-31).** This page is the
concise entry point; detailed records are linked at the bottom.

## Headline

The only defensible coverage-aware reliability ceiling remains **0.560** for the
126-target, 80%-coverage population. Phase 14 audited the reference and tested an
exact zero/spike-and-slab model. The audit supports the reference under a robust
one-event variance treatment; the model is **rejected**.

- Reference audit: V1 `R(22) = 0.3006` versus V0 `0.3136`, a difference of
  0.0131. The `0.560` ceiling is robust to the one-event zero under the robust
  estimator. V1h gives `0.3139 / 0.5603`; the `min_events=2` filter gives
  `0.1754 / 0.4188` on a different 36-target population.
- Model M4: real-data median `R(22) = 0.0543`, median `P(z=0) = 0.3235`.
- **Eight of twelve frozen gates failed**: SBC rank, grid edge, held-out PPC,
  both null q95 tails, prior sensitivity, reference agreement, and spike
  calibration.
- Decision: **`not_adopted_gate_failures`**. No calibrated interval, target
  ranking, animal-budget guarantee, causal effect, or digital-twin claim is
  supported.

The model is a measurement/reliability model, not a causal or generative model of
the animal.

## Phase 14 gate result

| Gate | Requirement | Observed | Status |
|---|---|---|---:|---|
| G1a SBC coverage | 0.93--0.97 | 0.9673 | pass |
| G1b SBC rank | KS p > 0.01 | 3.35e-192 | **fail** |
| G1c grid edge | < 0.05 | 0.0810 | **fail** |
| G1d fit integrity | <= 5% failures | 0 / 3,000 | pass |
| G1e one-event stratum | 0.90--0.98 | 0.9670 | pass |
| G2 held-out PPC | coverage >= 0.90; MAE <= 0.10 | 0.7697; 0.3998 | **fail** |
| G3 negative control | median <= 0.05; q95 <= 0.15 | 0.0000; 0.3535 | **fail** |
| G3b sign-flip | median <= 0.05; q95 <= 0.15 | 0.0000; 0.4509 | **fail** |
| G4 prior sensitivity | range <= 0.05 | 0.1220 | **fail** |
| G5 event-variance recovery | 0.90--0.98 | 0.9357 | pass |
| G6 reference agreement | within 0.05 of V1 | 0.2463 away | **fail** |
| G7 spike calibration | median P(z=0 \| z=0) >= 0.80 | 0.3561 | **fail** |

No threshold, prior, grid, seed, or filter was tuned after the full run. The
negative result is published as a result, not as a failed attempt to hide.

## The main diagnosis

A point mass at zero gives a zero **median** null estimate, but it does not fix
the upper tail: the negative-control q95 is 0.3535 and the sign-flip q95 is
0.4509. The spike is also poorly calibrated when the truth is zero (median
`P(z=0)=0.3561`). The posterior is prior-dominated: moving `pi0` from 0.25 to 0.75
moves real-data median `R(22)` from 0.1220 to 0.0. Reporting only the median would
have hidden the false positives.

The Phase 14A audit also shows that the old Phase 13 G6 comparison was not
like-for-like, but the reference gap was only 0.0131 under a robust estimator.
The Phase 13 conclusion stands independently because its SBC rank and null
controls failed.

## Benchmark inference withdrawn

The old Phase 12 animal-cluster benchmark comparison remains invalid: baseline
and model `y_true` differed for **1,334 of 1,334** paired rows because rows were
merged by animal/target/neuron aggregates rather than events. There is no
benchmark superiority claim and no "underpowered positive" framing. Leaderboard
values remain descriptive. `paired_cluster_inference` fails closed on unaligned
truth by default, with event-keyed pairing and BCa intervals.

A synthetic check of the repaired procedure found interval coverage of
0.900--0.917 at nominal 0.95 and null rejection of 0.075--0.083, so it is still
slightly undercovering before the alignment problem. No positive benchmark claim
is available.

## What remains supported

- Single-animal reliability is below 0.5 (bounded descriptive claim).
- Within-animal repeat-stimulation reliability is below 0.5 (bounded claim).
- The coverage-aware pooled ceiling is 0.560 under the Phase 11 estimator for the
  126-target population; Phase 14A supports its robustness to the one-event
  variance treatment.
- Phase 8 model comparisons retain their original statuses; none is a
  generalisation claim.

## Evidence

- [Phase 14 results](PHASE14_RESULTS.md)
- [Phase 14 methods](METHODS_PHASE14.md)
- [Reference audit](REFERENCE_AUDIT.md)
- [Phase 13 results](PHASE13_RESULTS.md)
- [Phase 12R results](PHASE12R_RESULTS.md)
- [Phase 12 results](PHASE12_RESULTS.md) (historical; C10/C11 corrections)
- [Phase 11 results](PHASE11_RESULTS.md)
- [Corrections](CORRECTIONS.md)
- [Claims](CLAIMS.md)
- [Limitations](LIMITATIONS.md)
- `reports/phase14/summary.json`, `sbc.json`, `heldout_ppc.json`, and the manifest

## Not done

- No new data acquisition or external validation cohort.
- No calibrated interval, target ranking, or animal-budget recommendation.
- No adopted zero-inflated model; M4 failed 8/12 gates.
- No external repeated-stimulation transport; status is `not_evaluable`.
- No generative model prototype; the conditional generative branch remains a
  future experiment.
