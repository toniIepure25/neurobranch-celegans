# Phase 5 research plan: SAAV mechanistic validation

## Outcome (post-registration)

The plan below was frozen before Phase 5 model evaluation. Results are now in
[`PHASE5_RESULTS.md`](PHASE5_RESULTS.md), with the protocol comparison in
[`PHASE5_PROTOCOL_COMPARISON.md`](PHASE5_PROTOCOL_COMPARISON.md), online test in
[`PHASE5_ONLINE_DECODING.md`](PHASE5_ONLINE_DECODING.md), mechanisms in
[`PHASE5_NEURAL_MECHANISMS.md`](PHASE5_NEURAL_MECHANISMS.md), intervention map
in [`PHASE5_INTERVENTION_EVIDENCE.md`](PHASE5_INTERVENTION_EVIDENCE.md), and
the 6-ECTS follow-up in [`PHASE5_NEXT_EXPERIMENTS.md`](PHASE5_NEXT_EXPERIMENTS.md).

Exact reproduction remains `BLOCKED_MISSING_INPUTS`; the reconstruction is
`PARTIAL_REPRODUCTION` and `METHOD_RECONSTRUCTION`. No regime produced a
positive, properly scored, animal-disjoint SAAV increment. No virtual
intervention was run.

**Branch:** `research/phase5-saav-mechanistic-validation`  
**Starting HEAD:** `14a43710334393c2dd234e848fb445d4317a734b`  
**Data:** Kramer et al. (2026) Dryad v5, CC0-1.0, local raw files only (not committed).  
**Status at plan freeze:** source audit completed; no Phase 5 model had been evaluated
when the hypotheses, splits and primary metrics below were fixed.

## Question

Phase 4 found no robust animal-disjoint neural increment for post-reversal
dorsal/ventral turn direction. Kramer et al. reported a positive SAAV +
head-curvature result. Phase 5 asks which protocol and preprocessing differences
could explain the discrepancy, and whether a prospective neural signal survives
when the prediction time is determined causally.

The goal is not to rescue the hypothesis. A null, a reconstruction that cannot
be completed exactly, or a protocol-dependent result are all acceptable outcomes.

## Status vocabulary

| status | meaning |
|---|---|
| `EXACT_REFERENCE_REPRODUCTION` | original inputs, code path and evaluation reproduced exactly |
| `METHOD_RECONSTRUCTION` | architecture/protocol reimplemented from source, with documented differences |
| `PARTIAL_REPRODUCTION` | some original inputs or preprocessing unavailable; approximate reconstruction evaluated |
| `BLOCKED_MISSING_INPUTS` | a required input is absent and cannot be reconstructed without invention |

No result in this phase is labelled `EXACT_REFERENCE_REPRODUCTION` unless the
four author-local CSVs are obtained and all source-level differences are removed.

## P0. Repository verification

- Verify actual branch, HEAD, clean tree and preserve Phase 1–4 history.
- Re-run the existing test suite and Ruff before adding Phase 5 code.
- Record the baseline test count and any failure.

## P1. Source and method contract

1. Read the Nature Neuroscience article and the bioRxiv preprint.
2. Clone the GPL-3.0 `flavell-lab/SAAV_Decoding` repository outside the tracked
   tree and record commit, file hashes and licence.
3. Reconstruct the exact contract from source, not from the prose:
   - GRUCell with input size 2, hidden size 3;
   - 0.5 dropout on every scanned hidden state;
   - LayerNorm over the three hidden units;
   - linear layer to one logit, sigmoid per timestep;
   - 4-frame input sections;
   - 250 epochs, Adam, learning rate `1e-3`, batch size 16;
   - five outer folds × four inner train/validation rotations;
   - ventral-event balancing in each training epoch;
   - validation-loss epoch selection;
   - test accuracy computed per section/timestep and averaged by class.
4. Record source-level discrepancies between prose and code, including:
   - all sliding 4-frame sections are used, not one sample per event;
   - the network returns a prediction at every timestep;
   - test accuracy is section/timestep accuracy, not an event-level AUROC;
   - the published test set is not explicitly animal-disjoint;
   - the notebook's displayed 12-frame region is not the model's full input.
5. Verify the four CSV paths and their absence from the public repository and
   Dryad. Do not synthesise the files or call a reconstruction exact.

## P2. Reconstructed data and protocol comparison

### Reconstruction assumptions (fixed before model evaluation)

The author-local aligned CSVs are unavailable. Raw Dryad HDF5 recordings are the
only public source. The reconstruction will:

- use raw `gcamp/trace_array_original` for SAAV, not the published whole-recording
  z-score;
- use orientation-corrected `dv` as signed head curvature, with positive dorsal;
- detect head-curvature zero crossings, assign phase `0, π, 2π, ...` at successive
  crossings, and resample to **four samples per full head-swing cycle**;
- build a **48-frame history ending at the last reversal frame** (11.75 cycles),
  which is the closest reproducible analogue of the notebook's 50-frame traces;
- retain wild-type events with duration ≥12 frames, valid two-frame post-turn
  window, a unique SAAV identity and finite phase-resampled values;
- exclude `|post-turn dv| < 0.05` as direction-ambiguous, matching the Phase 4
  outcome rule;
- z-score SAAV and HC within each recording for the original-style
  reconstruction; this is disclosed as retrospective normalisation and is not
  used as an online feature;
- keep all reconstruction choices fixed before inspecting model performance.
  No alternative alignment will be selected because it produces a larger effect.

This is `PARTIAL_REPRODUCTION`, not an exact reproduction. Differences include
the alignment interpolation, trace length, complete-case rules, cohort
composition and the PyTorch implementation of the Equinox GRU.

### Validation regimes

The same reconstructed inputs and architecture-compatible model will be run
under four explicitly labelled regimes:

| regime | split | claim |
|---|---|---|
| A. original-style | ungrouped 5-fold event split, 4 inner rotations | closest to the published protocol; measures event/section interpolation, not new-animal generalization |
| B. animal-disjoint | 5-fold grouped by animal, 4 inner rotations | principal generalization claim |
| C. chronological within-animal | earliest 80% of each animal's events train, latest 20% test | temporal generalization with animal overlap |
| D. external cohort | train WT octanol, test WT diacetyl | domain-shift generalization |

Primary offline metric: **event-level log loss**. Secondary: AUROC, AUPRC, Brier,
balanced accuracy, calibration slope/intercept and section-level accuracy. All
uncertainty is by animal-cluster bootstrap where animals are the dependence unit;
event-level bootstrap is reported only as a secondary, anti-conservative check.

### Protocol diagnostics

- animal overlap between train and test in regime A;
- number of events and animals per class per fold;
- recording-level fluorescence mean/SD distribution;
- repeated neural patterns / nearest-neighbour distances across animals;
- class prevalence and training-balance counts;
- head-curvature alignment validity and excluded-event reasons;
- model-seed variability;
- SAAV identity coverage;
- event-selection sensitivity;
- odor/cohort composition.

A protocol difference is not called leakage unless a concrete train/test
violation is demonstrated. Ungrouped CV is a valid estimator of a different
generalization target.

## P3. True online-prospective benchmark

The prediction landmark is fixed **after** a causally detected reversal onset and
does not use the reversal end, final direction or future neural activity.

Primary onset definition: the rising edge of the archival `reversal_vec` state,
which is available at the current frame. Sensitivity: a two-consecutive-frame
negative-velocity detector. Detection performance is reported against the
archival annotations.

Landmarks: fixed elapsed times after onset, `τ ∈ {0.6, 1.2, 2.4, 3.6, 4.8,
7.2}` s. At each landmark, an event is in the risk set only if the reversal is
still active at the prediction frame and at least 3.6 s of history is available.
The eventual direction is used only as the target, never to select the landmark.

Models:

1. training prevalence prior;
2. behaviour history (HC, velocity, angular velocity, pumping);
3. behaviour + sensory context (current odor state);
4. behaviour + SAAV;
5. behaviour + compact head-steering circuit;
6. behaviour + population neural summary.

Both a small linear model and the architecture-compatible GRU are run. Primary
online metric: **held-out event log loss** under animal-disjoint CV, with
animal-cluster bootstrap. Secondary: AUROC, AUPRC, Brier, calibration, balanced
accuracy, lead time and per-animal results. Risk-set counts by landmark and
direction are mandatory. No online result is called a deployed closed-loop
predictor.

## P4. Neural mechanisms

After the benchmark, use the phase-aligned traces to test:

- direction-dependent SAAV/circuit profiles and cross-animal reproducibility;
- whether neural features add information beyond past motor state;
- whether apparent leads survive temporally shifted neural controls;
- matched-neuron ablations with a global random-neuron null and
  Benjamini–Hochberg correction;
- identity coverage and stable subsets.

Feature importance is not treated as necessity. Encoding is distinguished from
causal contribution.

## P5. Intervention evidence

Build an evidence table from the published optogenetic experiments only:
modality, target, control, measured outcome, reported direction and whether
trial-level observations or extractable effect sizes exist. No pseudo-trials
from group means. No new virtual intervention optimisation without a validated
neural-response model. Any simulator output must retain:

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

## P6. Decision gates

| gate | decision |
|---|---|
| 1. Reproduction fidelity | exact / partial / blocked, with the missing inputs named |
| 2. Protocol sensitivity | does neural Δ differ materially between A and B? |
| 3. Prospective value | does SAAV/circuit improve a prespecified proper score from causal inputs? |
| 4. Mechanistic stability | are associations reproducible across animals and contexts? |
| 5. Intervention support | are trial-level intervention observations available? |

Criteria are not changed after seeing the final test predictions. Power and
sensitivity analyses accompany any claim of absence.
