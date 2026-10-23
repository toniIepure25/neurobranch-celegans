# Phase 3 research plan and preregistered evaluation

**Locked:** 2026-10-23, before any Phase 3 test-set metric was computed.
**Repository:** `D:\ComputaCenter\NeuroBranch\neurobranch-celegans`
**Branch:** `research/phase3-neural-generalization`
**Starting commit:** `c69fd5afe1a084827ce0c64f3f2c96fa0679b077`

This document freezes the primary hypotheses, target definitions, split policy,
metrics and decision rules. Exploratory analyses are allowed, but they are
labelled exploratory and cannot replace the primary test.

## 1. Inherited state

Phase 2 ended with 43 passing tests, clean Ruff, a seven-tab app, real Kato and
Randi data, a corrected branching test, an intervention-specificity audit and
two negative external benchmarks. Those results and warnings are preserved.
The Phase 3 branch is additive; no prior commit is rewritten.

## 2. Datasets

### Primary behavioural dataset

Baskoylu & Flavell (2026), Dryad `doi:10.5061/dryad.w9ghx3g4v`, CC0-1.0.
The archive contains one HDF5 file per animal, 27 animals, 1,600 confocal frames
per animal, raw and z-scored GCaMP traces, velocity, reversal state and reversal
events, head angle, angular velocity, arena distance, and a separate JLD2
NeuroPAL identity file.

The published `gcamp/trace_array` is z-scored and may use whole-recording
statistics. Phase 3 will use `gcamp/trace_array_original` and fit every
normalisation, baseline, scaler and PCA on training data only. If the published
z-score is used at all, it is a clearly labelled sensitivity analysis.

### Secondary optogenetic dataset

Randi et al. (2023), already local at `data/raw/randi2023/exported_data`, archive
SHA256 `d6e7b3d93175b40b7ae17bde2182835e9c2144388142c522ee9be3832f6ce836`.
Phase 3 reuses the Phase 2 parsed trial table and extends the target audit.

### Context

Kato 2015, Randi 2023 and Baskoylu 2026 differ in preparation, stimulus,
behaviour and neuron coverage. They are analysed as separate domains. No shared
training labels or pooled intervention responses are used without an explicit
measurement model. No cross-dataset transfer is claimed.

## 3. Primary prospective behavioural question

**Target.** For animal \(a\) and confocal frame \(t\), predict whether a
reversal **onset** occurs in the half-open future interval \((t, t+H]\) seconds.
Frames with `reversal_vec[t] == 1` are ineligible, because the event has already
started. A complete future window is required. Reversal onsets are derived from
`behavior/reversal_events` and checked against rising edges of
`behavior/reversal_vec` before use.

> **Correction recorded before any test metric was computed.** The export's
> event table is **0-based start, exclusive end**, not 1-based inclusive. The
> audit of all 1,162 events established this; an earlier 1-based assumption
> produced 31–41 mismatches per animal and was discarded. The target definition
> (onset in `(t, t+H]`) and the eligibility rule are unchanged. See
> [`PHASE3_DATASET_AUDIT.md`](PHASE3_DATASET_AUDIT.md) §3.

**Horizons.** 3, 6 and 12 seconds. At the observed median sampling interval of
0.600 s these are 5, 10 and 20 frames. The **primary horizon is 6 s**; 3 s and
12 s are prespecified secondary horizons.

**Primary generalisation level.** Unseen animals. Leave-one-animal-out
cross-validation over all usable animals. Each held-out animal is scored by a
model trained on the other animals only.

**Separate task.** Within-animal forward prediction: chronological
70% train / 15% validation / 15% test with a purge of at least \(H\) frames.
This is reported separately and is never pooled with the animal-disjoint test.

## 4. Models

All models use identical eligible rows, folds and proper scoring code.

1. Constant event-rate baseline (training prevalence).
2. Current behavioural state: velocity, speed, head angle, angular velocity,
   arena distance, reversal state.
3. Behavioural history: current state plus causal windows (3, 6, 12 s) of
   velocity, acceleration, angular velocity, distance, time since last reversal,
   forward-run duration, reversal counts and window means/slopes.
4. Behavioural Markov/hazard baseline: discrete hazard by forward-run age bin.
5. Neural history only: population-level causal neural summaries.
6. Behaviour + neural history (primary incremental comparison).
7. Behaviour + train-only PCA of neural history.
8. One compact nonlinear model only after the linear benchmark is complete.

Neural features are computed from causally normalised raw traces. At each frame,
the normalisation uses only that frame and the preceding 30 s. Population-level
features are permutation-invariant summaries across recorded ROIs (mean, SD,
quantiles, extremes, active fractions and their temporal differences), because
ROI indices are not neuron identities. When the NeuroPAL mapping is verified,
an identified-neuron benchmark is added using only labels present in the
training animals and the held-out animal; missing labels are masked, not
imputed as identity matches.

## 5. Preprocessing and leakage controls

- All event labels are future-looking; no future behaviour enters any feature.
- Normalisation, imputation, scaling, PCA and hyperparameter selection are fit
  inside each training fold.
- The held-out animal is never used to choose a hyperparameter or threshold.
- Inner hyperparameter selection uses animal-grouped cross-validation on the
  training animals only. The primary logistic models use a **prespecified fixed
  L2 strength C=0.1**; this avoids hidden test tuning and makes the number of
  fits auditable. A sensitivity run over C∈{0.03, 0.3} is reported separately.
- No overlapping train/test horizons; the forward protocol has an explicit
  purge of \(H\) frames.
- No random frame-level split.
- Multiple random seeds are used only where a model is stochastic; the primary
  logistic models are deterministic.
- Event prevalence and the behavioural-history baseline strength are reported
  before any neural increment is interpreted.

## 6. Metrics and primary decision rule

Primary metric: held-out **log loss**, lower is better.
Secondary: Brier score, balanced accuracy, AUPRC, event precision/recall at a
training-selected threshold, and calibration (slope/intercept and reliability
curve). AUROC may be reported but is not the primary metric.

Incremental value is paired on identical held-out rows:

\[
\Delta = \text{log loss}(\text{behaviour+neural}) -
         \text{log loss}(\text{behaviour-only})
\]

Negative \(\Delta\) means neural information helps. Uncertainty is a
animal-level bootstrap (10,000 resamples, seed 17) over the per-animal paired
differences. A positive claim requires:

1. mean \(\Delta < 0\);
2. the 95% animal-level CI upper bound \(< 0\);
3. at least 70% of animals have negative \(\Delta\);
4. the result survives a prespecified sensitivity analysis excluding the
   strongest behavioural-history features one family at a time.

If these conditions fail, the result is **negative or unresolved**, not
"no neural information exists". A CI crossing zero is inconclusive, not proof of
absence. Transition accuracy alone is never evidence of improvement.

## 7. Representation benchmark

Compare, under the same animal-disjoint splits:

- raw population summaries;
- train-only PCA of population features;
- linear autoregressive neural representation;
- compact causal temporal encoder;
- BunDLe-inspired representation, without transductive fitting;
- behaviour-supervised representation trained only on training animals.

Every encoder is fit on training animals, frozen before scoring the held-out
animal, and saved with a provenance record (training animals, feature set,
seed, Git commit, input hash). A visually coherent 3-D manifold is not evidence.
The decision is **KEEP**, **ITERATE** or **STOP** based on prospective proper
scoring, not on silhouette or reconstruction alone.

## 8. Randi response audit

The existing signed-peak target is compared with prespecified alternatives:

- mean post-stimulus response;
- integrated response over the post window;
- response sign;
- response probability above the trial noise floor;
- normalised temporal response profile;
- standardised response.

Reliability is decomposed across repeated trials, animals, neuron pairs and
stimulated-neuron classes. Variance components are reported only when the design
supports them. Baselines are nested: global mean, population mean,
stimulated-neuron mean, recorded-neuron mean, pair mean, regularised additive
effects, and connectivity-informed regression, all fit on training folds with
explicit fallback for unseen neurons/pairs.

Context models use pre-stimulation activity only. Held-out protocols are new
events, new targets and new animals, reported separately.

**Causal distinction.** A = predicting a response to an observed intervention.
B = predicting an intervention not represented in training. C = predicting an
intervention effect from spontaneous activity alone. Success at A is not
evidence for B or C.

## 9. Transition-state analysis

Event-aligned neural activity before reversals is compared with matched
non-reversal periods, matching current behavioural state and recent history.
Controls include multiple pre-event horizons, time-shifted events,
behaviour-preserving nulls, animal-disjoint evaluation and uncertainty. A
separation is not a decision or a cause. The earliest predictive horizon is
reported only if it survives the behavioural-history baseline.

## 10. Decision gates

| Gate | Pass condition |
|---|---|
| 1. Independent data | loaded, aligned, audited, tested; usable animals documented |
| 2. Prospective benchmark | leakage-safe animal-disjoint comparison with proper scores |
| 3. Representation value | frozen train-only representation beats raw/behavioural baselines with uncertainty |
| 4. Optogenetic response | prespecified alternative target/context model beats held-out baselines reproducibly |
| 5. Intervention generalisation | only claimed for genuinely held-out intervention conditions |

A sound negative result passes gates 1 and 2 and fails 3/4 honestly.

## 11. Exclusion and quality rules

An animal is excluded only for a documented structural reason (unreadable file,
non-monotonic timestamps, missing behaviour, insufficient events), never because
a model performed poorly on it. Neuron traces that are entirely non-finite are
excluded; individual missing values are skipped, not imputed. Ambiguous
duplicate NeuroPAL labels are excluded from identity-aligned analyses.

## 12. Reporting rule

Every number in the Phase 3 reports must be traceable to a committed CSV/JSON
artifact or a saved prediction file. Synthetic fixtures are labelled synthetic.
No biological intervention was performed. No virtual perturbation is described
as causally validated.

> Any virtual intervention remains a **MODEL-DEPENDENT VIRTUAL PERTURBATION —
> NOT CAUSALLY VALIDATED.**
