# Phase 4 turn-direction benchmark

## Question and design

The primary question is whether pre-offset neural activity adds animal-disjoint
prospective information about the post-reversal dorsal/ventral turn beyond a
strong behavioural/context baseline.

- **Cohorts:** WT octanol and WT diacetyl; 56 animals.
- **Events:** reversals ≥12 frames with a valid two-frame post-offset mean;
  945 events, 186 dorsal (19.7%) and 759 ventral.
- **Label:** positive mean `dv` over the first two post-offset frames = dorsal.
- **Primary window:** four middle frames of the reversal.  The segment ends at
  least two frames before the offset (`seg_end <= end - 2`), so there is a
  two-frame guard band between the last feature and the outcome window.  No
  outcome-window frame is a feature.  Behaviour features exclude
  completed-reversal duration and time-to-offset, and odour context is
  truncated at the feature-segment end.
- **Selection caveat:** segment location is event-aligned (defined from the
  completed reversal, as in the paper's offline task), not selected online at
  reversal onset.  Every feature value is pre-offset, but this is not an
  onset-triggered real-time decoder.
- **Split:** leave-one-animal-out.
- **Model:** StandardScaler + logistic regression, fixed `C=0.1`, no class
  weighting, no test-time calibration.
- **Primary metric:** pooled out-of-fold AUROC; paired animal-cluster
  bootstrap (2,000 resamples) for the increment over behaviour.
- **Event metrics:** AUPRC, log loss, Brier, balanced accuracy at 0.5,
  accuracy, calibration slope/intercept, per-animal AUROC.
- **Artifacts:** `turn_direction_predictions.parquet` contains every held-out
  event prediction for every mode.

## Primary results

| mode | AUROC | AUPRC | log loss | Brier | balanced accuracy | calibration slope |
|---|---:|---:|---:|---:|---:|---:|
| behaviour | 0.595 | 0.259 | 0.4987 | 0.1566 | 0.499 | 0.428 |
| behaviour + population | **0.608** | **0.293** | **0.4971** | **0.1552** | 0.510 | 0.470 |
| behaviour + SAAV | 0.591 | 0.258 | 0.5000 | 0.1569 | 0.499 | 0.404 |
| behaviour + circuit | 0.586 | 0.263 | 0.5064 | 0.1580 | 0.504 | 0.318 |
| behaviour + identified | 0.577 | 0.272 | 0.5848 | 0.1777 | 0.537 | 0.176 |
| neural population only | 0.556 | 0.244 | 0.4963 | 0.1578 | 0.500 | 0.480 |
| neural circuit only | 0.503 | 0.207 | 0.5038 | 0.1606 | 0.500 | 0.020 |
| neural SAAV only | 0.379 | 0.154 | 0.4994 | 0.1591 | 0.500 | −4.94 |

The primary contrast is **+0.0123 AUROC** for behaviour + population over
behaviour, 95% CI **[−0.0166, +0.0425]**, bootstrap `p(Δ≤0)=0.210`.  The CI
includes zero.  The prespecified classification is **no incremental evidence**.

Per-animal AUROC is available for the 50/56 animals with both classes.  The
population model improves AUROC in 25/50 animals, worsens it in 20/50, and
ties in 5/50; median Δ +0.017, mean +0.006.  The animal-level distribution is
broad and does not support a consistent effect.

Calibration is poor for every mode: slope 0.43 for behaviour and 0.47 for
behaviour + population, with negative intercepts.  Probabilities are
overconfident and should not be read as absolute turn probabilities.

## Prespecified neural representations

- **SAAV:** Δ AUROC **−0.0042**, 95% CI **[−0.0117, +0.0029]**.  SAAV is
  present in only 27/56 WT animals, so this is also tested in the
  SAAV-available subset below.
- **Eight-neuron circuit** (SAAV, RMDV, RMDD, SMDV, SMDD, RIV, SMBV, SMBD):
  Δ **−0.0097**, CI **[−0.0380, +0.0191]**.
- **Full identified vocabulary:** Δ **−0.0176**, CI
  **[−0.0991, +0.0574]**; log loss is worse.  Sparse coverage makes this a
  weak representation, not a strong test of identified-neuron sequences.

None of the prespecified neural representations shows a positive,
uncertainty-bounded increment.

## Cohort and window sensitivity

| analysis | behaviour AUROC | +population AUROC | Δ | 95% CI |
|---|---:|---:|---:|---|
| WT octanol (32 animals, 596 events) | 0.596 | 0.596 | −0.0002 | [−0.0429, +0.0394] |
| WT diacetyl (24 animals, 349 events) | 0.519 | 0.582 | +0.0624 | [−0.0040, +0.1241] |
| early window (all WT) | 0.593 | 0.569 | −0.0243 | [−0.0520, +0.0009] |
| late window (all WT) | 0.618 | 0.619 | +0.0013 | [−0.0322, +0.0348] |

No cohort or window has a CI excluding zero.  The diacetyl cohort is the
largest point estimate but remains **secondary, exploratory**: the behaviour
baseline is near chance (0.519), the cohort has only 349 events and 24
animals, and the CI includes zero.  It is a hypothesis for a follow-up, not
evidence of a general neural turn-direction signal.  The earlier feature set
that included completed-reversal duration gave a larger diacetyl estimate;
removing that future information attenuated it, which is exactly why the final
benchmark excludes it.

## SAAV-available subset

In the 27 animals with an SAAV trace (445 events):

| mode | AUROC | log loss |
|---|---:|---:|
| behaviour | 0.475 | 0.5645 |
| behaviour + SAAV | 0.479 | 0.5656 |
| SAAV only | 0.482 | 0.5347 |

The SAAV increment is **+0.0045 AUROC**, 95% CI **[−0.0127, +0.0210]**.
The behaviour baseline is below chance in this selected subset, which makes
the absolute values difficult to interpret.  There is no evidence that SAAV
carries a prospective turn-direction signal in this subset.

## External cohort shift

Fit on one cohort and score a disjoint cohort; no target-cohort fitting.

| train → test | behaviour | +population | +SAAV | +circuit |
|---|---:|---:|---:|---:|
| octanol → diacetyl | 0.564 | 0.551 | 0.565 | 0.571 |
| WT → tdc1 | 0.621 | 0.614 | 0.629 | 0.609 |
| octanol → tdc1 | 0.612 | 0.626 | 0.621 | 0.576 |

The transfer is modest and inconsistent.  There is no evidence that a neural
representation learned in WT rescues the mutant cohort, and the octanol →
diacetyl transfer does not reproduce the within-diacetyl exploratory gain.
This is consistent with a distribution/cohort-dependent effect or with noise,
not with a robust cross-cohort neural signal.  These external numbers are
descriptive; they were not included in a formal multiplicity correction.

## Within-animal comparison

A five-fold event split (not animal-disjoint) gives behaviour AUROC 0.584,
behaviour + population 0.602 and behaviour + circuit 0.577.  The population
point estimate is larger than in LOAO, but this split reuses animals across
folds and is not evidence of cross-animal generalization.  It neither rescues
nor explains away the primary null.

## Nonlinear sensitivity

A secondary gradient-boosted tree model (120 iterations, depth-15 leaves,
`min_samples_leaf=20`, L2=1.0) gives:

| mode | AUROC | log loss |
|---|---:|---:|
| behaviour | 0.605 | 0.5249 |
| behaviour + population | 0.617 | 0.5239 |

Δ AUROC **+0.0121**, 95% CI **[−0.0275, +0.0481]**.  The nonlinear model gives
a similar null-compatible point estimate to the linear benchmark, and the CI
still includes zero.  This weakens the “insufficient linear representation”
explanation, though it does not reproduce the paper's RNN or its exact
alignment.

## Controls

The shifted-neural population control is worse than the true window
(Δ AUROC −0.0164 on 913 common events), circuit identity permutation is
essentially unchanged (0.5824 vs 0.5857), and within-animal label permutation
(6 draws) gives an observed Δ of +0.0125 against a null SD of 0.0226
(one-sided empirical p = 0.333).  Matched random-neuron ablation (24 draws)
places every prespecified feature effect inside the random-control
distribution (random mean +0.0012, SD 0.0033; all p ≥ 0.12).  These controls do
not establish a specific neural effect.  See
`PHASE4_NEURAL_BRANCH_DYNAMICS.md` and the CSV/JSON outputs for details.

## Interpretation

Under the primary animal-disjoint design, the answer is **no incremental
prospective turn-direction information from the tested neural
representations**.  The result does not prove that the circuit is uninvolved.
It bounds what this archive, preprocessing, target and model can decode.  The
main positive signals are behavioural: head state immediately before the
offset is highly predictive.  The diacetyl population estimate is positive but
its CI includes zero, so it is not a bounded positive finding.

## Outputs

- `reports/phase4/turn_direction_metrics.csv`
- `reports/phase4/turn_direction_per_animal.csv`
- `reports/phase4/turn_direction_predictions.parquet`
- `reports/phase4/turn_direction_calibration.csv`
- `reports/phase4/turn_direction_summary.json`
- `reports/phase4/turn_direction_cohorts.csv`
- `reports/phase4/turn_direction_window_sensitivity.csv`
- `reports/phase4/turn_direction_saav_subset.csv`
- `reports/phase4/turn_direction_external.csv`
- `reports/phase4/turn_direction_nonlinear.csv`
- `reports/phase4/turn_direction_within_animal_cv.csv`
- figures in `reports/phase4/figures/`
