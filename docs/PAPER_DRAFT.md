# Measurement-bounded reliability in a whole-brain *C. elegans* perturbation atlas

> **Superseded.** This Phase 14 draft is retained as a historical record. The
> current Phase 15 manuscript is `manuscript/phase15_reliability.md`; it
> replaces the 0.560 headline with `R_obs = 0.2606`, `R_22 = 0.3006`
> (counterfactual for 87/126 targets), and `R_id = 0.3501`. See
> [`PHASE15_RESULTS.md`](PHASE15_RESULTS.md) and C15.

**Draft status:** internal manuscript draft. Not submitted, not peer reviewed.
Numbers are generated from `reports/phase14/`; the claims ledger is authoritative.

## Abstract

Whole-brain optogenetic atlases are increasingly used to rank neurons and to
design interventions, but the reliability of the underlying response profiles is
rarely reported with a calibrated uncertainty procedure. We audit a widely used
*C. elegans* atlas and test three measurement models. The historical descriptive
ceiling is 0.560 for a 126-target, 80%-coverage population. A reference audit
shows that this bound is robust to replacing the zero variance of one-event cells
with a robust event-variance estimate (V1 0.3006 versus V0 0.3136), but it is
specific to that population. An exact zero/spike-and-slab model removes the
median null bias yet fails eight of twelve pre-registered calibration gates,
including rank uniformity, held-out prediction, both null upper tails, prior
sensitivity, reference agreement, and spike calibration. We report the negative
result and argue that a zero median is insufficient evidence for a valid null
model. No calibrated interval, target ranking, animal-budget guarantee, causal
effect, or digital-twin claim is supported.

## 1. Introduction

Perturbation atlases promise to connect neuron identity to response and to guide
experiments. A ranking is only as useful as the reliability of the profiles it
ranks. Two failure modes are common: treating sparse measurements as noiseless,
and reporting a point estimate whose interval has not been calibrated.

We study the Randi et al. (2023) optogenetic atlas through a derived event table.
The analysis is deliberately measurement-first: before asking which neuron is
"most important", we ask how repeatable a target's response profile is.

## 2. Data and estimand

- 237,639 stimulation events, 110 animals, 259 targets, 305 neurons.
- Animal-disjoint evaluation; 126 targets at 80% responder coverage, at least 6
  animals and 3 responders.
- Estimand: reliability of a target profile pooled over `n` animals,
  `R(n) = sigma^2 / (sigma^2 + (tau^2 + nu)/n)`.
- The ceiling is `sqrt(R)`; it bounds correlation with an error-free predictor.

The input is a derived table; raw recordings are not redistributed.

## 3. Reference audit

Phase 11 set the variance of one-event cells to zero. We reproduced that
estimator exactly and compared four treatments:

| Treatment | R(22) | Ceiling |
|---|---:|---:|
| V0 historical | 0.3136 | 0.5600 |
| V1 robust one-event variance | 0.3006 | 0.5482 |
| V1h homogeneous event variance | 0.3139 | 0.5603 |
| V2 `min_events=2` | 0.1754 | 0.4188 |

V1 uses the median observed within-cell variance (`omega^2 = 1.7629`) for cells
with one event. The difference from V0 is 0.0131, below the pre-registered 0.05
audit threshold. The `min_events=2` result is a different 36-target population,
not a corrected 0.560.

## 4. Models

- **M0:** Phase 12R reference.
- **M1:** projection-scaled known variance.
- **M2:** marginal event-variance model.
- **M4:** exact zero/spike-and-slab, with `z ~ Bernoulli(0.5)`,
  `mu | z=1 ~ N(0, sigma^2)`, `mu | z=0 = 0`, `u ~ N(0, tau^2)`.

Phase 13 rejected M1 and M2. M1 was nearly neutral for the point estimate but
failed SBC rank; M2 lowered the estimate to 0.2003 but failed both null controls.
M4 was designed to repair the positive null bias with an exact point mass.

## 5. Results

M4 real-data median `R(22) = 0.0543`; median posterior `P(z=0) = 0.3235`. The
model is rejected:

| Gate | Observed | Status |
|---|---:|---|
| SBC coverage | 0.9673 | pass |
| SBC rank KS p | 3.35e-192 | fail |
| Grid edge | 0.0810 | fail |
| Fit integrity | 0 / 3,000 | pass |
| One-event coverage | 0.9670 | pass |
| Held-out PPC coverage / MAE | 0.7697 / 0.3998 | fail |
| Negative-control median / q95 | 0.0000 / 0.3535 | fail |
| Sign-flip median / q95 | 0.0000 / 0.4509 | fail |
| Prior range | 0.1220 | fail |
| Event-variance coverage | 0.9357 | pass |
| Reference gap | 0.2463 | fail |
| Spike calibration | 0.3561 | fail |

The zero median is not a pass: the null upper tails remain too heavy, and the
spike is poorly identified when the truth is zero. The result is prior-dominated:
moving `pi0` from 0.25 to 0.75 moves the real-data median from 0.1220 to 0.0.

## 6. Discussion

A point mass at zero is necessary but not sufficient for a credible null model.
Calibration must assess the full null distribution, spike identification, and
prior sensitivity, not just the median. The reference audit shows that the
historical 0.560 bound is not an artifact of the one-event zero, but it is not a
universal constant either: the target population and event filter matter.

The practical recommendation is to report reliability with its population,
variance treatment, and null tail, and to avoid ranking targets until a
ranking-stability gate has been pre-registered and passed.

## 7. Limitations

- No independent repeated-stimulation dataset; transport is `not_evaluable`.
- The reference bootstrap is conditional on the estimated event variance.
- The SBC templates preserve real masks but simulate truths; they are not an
  external validation cohort.
- The model is a measurement model, not a causal or generative model.
- The literature scan is rapid and non-systematic.

## 8. Reproducibility

`reports/phase14/` contains the real fit, reference audit, SBC, PPC, nulls, prior
sensitivity, manifest, and validation. The run is CPU-only and seeded. The
validator fails on missing PPC, insufficient SBC, gate/decision mismatch, hash
mismatch, stale checksum, or a dirty code worktree.

## References

Reference management is in `research/phase14_evidence.csv`; no citation is added
without a verified DOI/PMID. The manuscript draft does not yet contain a
reference list and must not be circulated as a submission.
