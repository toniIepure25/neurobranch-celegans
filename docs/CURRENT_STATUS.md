# Current status

**As of the Phase 15 reliability audit (2026-11-01).** This page is the concise
entry point; detailed records are linked at the bottom.

## Headline

The Phase 11 **0.560** headline is withdrawn as a bound. It was
`sqrt(R(22))` with `R(22) = 0.3136`; only **39 of 126** targets have at least 22
animals, so 87 values were extrapolated. Phase 15 reports three estimands:

| Estimand | Estimate | 95% delete-one-animal CI | Targets | Extrapolated |
| --- | ---: | --- | ---: | ---: |
| `R_obs` (observed counts) | **0.2606** | [0.1782, 0.3430] | 126 | 0 |
| `R_22` (counterfactual) | **0.3006** | [0.2087, 0.3924] | 126 | 87 |
| `R_id` (`n >= 22`) | **0.3501** | [0.2020, 0.4982] | 39 | 0 |

`R_obs` is the descriptive observed-data quantity; `R_22` is a design
counterfactual; `R_id` is a different population.

- Variance robustness passes: V0 `0.2709 / 0.3136 / 0.3579`, V1
  `0.2606 / 0.3006 / 0.3501`, V1h `0.2725 / 0.3139 / 0.3635`; all V1-V0 deltas
  are below 0.05. The V2 filter changes the population to 36 targets.
- **Extrapolation fails**: median absolute error 0.204 and bias +0.038 against
  thresholds 0.10 / 0.05.
- **Ranking fails**: median Spearman 0.126 (95% interval -0.021 to 0.247),
  top-10 Jaccard 0.176. No ranking is published.
- The audit decision and calibration/null gate statuses are in
  `reports/phase15/summary.json`; this page does not promote a descriptive
  estimate to a calibrated interval.
- Phase 14 remains rejected: M4 median 0.0543, 8/12 gates failed, and 0.0543 is
  prior-dominated. No calibrated interval, target ranking, animal-budget
  guarantee, causal effect, or digital-twin claim is supported.

This is a measurement/reliability audit, not a causal or generative model of the
animal.

## Core dynamics simulator research branch

A separate branch, `research/core-dynamics-simulator`, adds a free-running
population simulator over the existing Kato checkpoints. Its scientific result
is negative and is reported separately from the Phase 15 measurement audit:

| Gate | Question | Status |
| --- | --- | --- |
| G1 | Valid free-running rollout from `x(0)` | pass |
| G2 | Advantage over constant-state / persistence | fail (0/5 worms) |
| G3 | Reversal branching better than marginal null | fail (0/5) |
| G4 | Named-neuron perturbation interface | pass (interface only) |
| G5 | Specific, robust perturbation | fail |
| G6 | Biological validation | not_evaluable |

The selected model (`mlp_residual`) has the best validation one-step R²
(0.7134) but the worst free-running test R² (−0.5098) and an OOD fraction of
0.79. VAR is the best descriptive free-running model (R² 0.5325, ΔMSE −0.0491,
OOD 0.0076) but was not selected and is not a causal simulator. No
named-neuron effect exceeds the random-control 95th percentile. Every
perturbation output is a **model-dependent virtual perturbation, not causally
validated**.

Start at [SIMULATOR_RESULTS.md](SIMULATOR_RESULTS.md),
[SIMULATOR_LIMITATIONS.md](SIMULATOR_LIMITATIONS.md), and
[SIMULATOR_SUPERVISOR_BRIEF.md](SIMULATOR_SUPERVISOR_BRIEF.md).

### Phase 6 dynamical validation

Phase 6 corrects the history/evaluation contract, selects all models on validation
block 4, and treats test block 2 as exploratory because Phase 5 had already
inspected it.

| Gate | Phase 6 status |
| --- | --- |
| G1 functionality | pass |
| G2 autonomous skill | fail (3/5 vs frozen Phase 5 VAR) |
| G3 distributional fidelity | fail (0/5) |
| G4 stochastic branching | fail |
| G5 support-constrained perturbation | pass |
| G6 robust target specificity | not_evaluable (one family only) |
| G7 biological validation | not_evaluable |

No learned family beat persistence by median validation skill. The validation-selected VAR
(p2, ridge 10) reaches exploratory H=20 test skill 0.093 versus 0.101 for the
frozen Phase 5 VAR; the Phase 5 MLP diverges (H=20 −1.356, H=100 −18.0). The
stochastic VAR improves autocorrelation but is worse than persistence on energy
distance and covariance; branching TV is 0.513 versus 0.264 for a Markov chain.

Start at [SIMULATOR_PHASE6_RESULTS.md](SIMULATOR_PHASE6_RESULTS.md) and
[PHASE6_SIMULATOR_SUPERVISOR_BRIEF.md](PHASE6_SIMULATOR_SUPERVISOR_BRIEF.md).
The older Phase 5 tables above are retained as run history.

### External-validation readiness

The external audit keeps the Phase 6 result frozen and asks whether any public
recording can test it without reusing Kato test block 2. Frozen transfer is
blocked: each checkpoint is tied to a worm-specific ordered trace panel, and the
external files have different panels and multi-neuron trace labels. Kato Stim
supports only a limited per-animal refit on its constant-21%-oxygen first 360 s;
Nichols 2017 is a different developmental/preparation regime and was not run.

The Kato Stim pilot is negative: **0/7 sessions** beat persistence at H=20
(median skill −0.602; mean −0.593; bootstrap 95% CI −1.024 to −0.241). Oxygen is
treated as measured environmental input, not targeted neuronal stimulation.

The Phase 6 audit also records that the official support threshold is the 99th
percentile, while the original protocol named the 95th; the q=0.95
condition-level gate status cannot be reconstructed from the published event
table. The nominal VAR spectral cap is not exactly realized (realized radii
0.996–1.050), and the Phase 6 manifest verifies only under LF-normalized hashes
(40/46 files on Windows). These are implementation/provenance limitations, not a
retroactive change to the official result.

Start at [EXTERNAL_DATASET_COMPATIBILITY.md](EXTERNAL_DATASET_COMPATIBILITY.md),
[EXTERNAL_VALIDATION_PROTOCOL.md](EXTERNAL_VALIDATION_PROTOCOL.md),
[PHASE6_FINAL_SCIENTIFIC_AUDIT.md](PHASE6_FINAL_SCIENTIFIC_AUDIT.md) and
[RESEARCH_DIRECTION_FOR_DENIS.md](RESEARCH_DIRECTION_FOR_DENIS.md).

### Context-conditioned generative dynamics branch

The separate branch `research/context-conditioned-generative-dynamics` asks whether
stimulus-aware, compositional, low-dimensional dynamics improve free-running
simulation. The answer on the available recordings is **no**:

- selected latent LDS median H20/H100 skill: −0.911 / −0.164;
- corrected stable VAR median H20/H100 skill: −0.770 / −0.061;
- compositional 2-operator test H20 mean skill −1.771, with coefficient-policy
  collapse and no session better than the latent LDS;
- measured oxygen did not improve H20 prediction (B−A paired MSE difference +0.008);
- shared zero-shot dynamics H20 mean skill −2.822; calibration adaptation made it
  worse;
- q99 maximum-distance support coverage is 0.86 for the Phase 6 switching VAR and
  0.093 for context latent session 1.

Gates: G1 PASS, G2 PASS, G3 PASS, G4 CONDITIONAL, G5 FAIL, G6 FAIL, G7
NOT_EVALUABLE, G8 NOT_EVALUABLE. The new dashboard tab is
“CONTEXT-AWARE NEURAL SIMULATOR”; it preserves all existing tabs and labels
measured oxygen, hypothetical oxygen, model perturbation, and actual stimulation
data separately.

Start at [SIMULATOR_CONTEXT_RESEARCH_PROTOCOL.md](SIMULATOR_CONTEXT_RESEARCH_PROTOCOL.md),
[SIMULATOR_CONTEXT_RESULTS.md](SIMULATOR_CONTEXT_RESULTS.md),
[SIMULATOR_CONTEXT_LIMITATIONS.md](SIMULATOR_CONTEXT_LIMITATIONS.md), and
[SIMULATOR_CONTEXT_SUPERVISOR_BRIEF.md](SIMULATOR_CONTEXT_SUPERVISOR_BRIEF.md).

## Phase 15 gate result

The final statuses are produced by `scripts/phase15_reliability.py` and checked
by `scripts/phase15_validate.py`. The decision is fail-closed: `G3` calibration
and `G4` null control must both pass before any interval is called calibrated,
and `G5` extrapolation must pass before `R_22` is called supported.

| Gate | Observed | Status |
| --- | --- | --- |
| G1 estimand completeness | 126 targets; 87 extrapolated at 22 | pass |
| G2 jackknife integrity | valid fraction 1.0 | pass |
| G3 calibration | coverage 0.990 / 0.976 / 0.999 | **fail** |
| G4 null control | null medians 0.096--0.137; fold sign-flip q95 0.1641 | **fail** |
| G5 extrapolation | MAE 0.204; bias +0.038 | **fail** |
| G6 ranking | Spearman 0.126; Jaccard 0.176 | **fail** |
| G7 variance robustness | deltas 0.0103 / 0.0131 / 0.0078 | pass |
| G8 reproducibility | manifest/validator | checked at release |

Decision: **`not_calibrated`** (`G3`, `G4`, `G5`, `G6`). The intervals
over-cover in simulation and are too wide, but the null controls still produce
0.10--0.14 apparent reliability. No calibrated interval, ceiling, ranking,
animal budget or causal claim is published.

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
- The observed-count reliability is 0.2606 (interval 0.1782--0.3430) for the
  126-target population; this is descriptive and its calibration status is
  reported separately. The historical 0.560 value is extrapolated for 87 targets
  and is not a bound.
- Phase 8 model comparisons retain their original statuses; none is a
  generalisation claim.

## Evidence

- [Phase 15 results](PHASE15_RESULTS.md)
- [Phase 15 methods](METHODS_PHASE15.md)
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
- `reports/phase15/summary.json`, `estimands.csv`, `calibration.json`,
  `manifest.json`, and `validation.json`
- `reports/phase14/summary.json`, `sbc.json`, `heldout_ppc.json`, and the manifest

## Not done

- No new data acquisition or external validation cohort.
- No calibrated interval, target ranking, or animal-budget recommendation.
- No adopted zero-inflated model; M4 failed 8/12 gates.
- No external repeated-stimulation transport; status is `not_evaluable`.
- No generative model prototype; the conditional generative branch remains a
  future experiment.
