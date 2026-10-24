# Phase 4 research plan: turn direction and neural sequences

## Scientific question

Phase 3 found no reproducible animal-disjoint neural gain for 6 s prospective
reversal prediction.  Phase 4 asks a narrower and more interpretable question:

> Is the missing neural signal explained by (a) an inappropriate prediction
> target, (b) insufficient neural representations, (c) distribution shift, or
> (d) a genuine absence of prospectively decodable information?

The four explanations are treated as competing hypotheses, not as an ordered
list of excuses for a null result.  A negative result is a valid outcome.

## Why the published result and this benchmark differ

Kramer et al. (2026, *Nature Neuroscience*,
<https://doi.org/10.1038/s41593-026-02257-5>; preprint
<https://doi.org/10.1101/2024.08.11.607076>) decode the **direction** of the
post-reversal turn (dorsal vs ventral) from four-frame stretches of reversal
activity, using an RNN and five-fold cross-validation.  Their released code
expects pre-extracted CSVs (`SAAV_dorsal.csv`, `SAAV_ventral.csv`,
`HC_dorsal.csv`, `HC_ventral.csv`) that are **not in the repository**.  The
Dryad archive contains per-animal MATLAB/HDF5 recordings, not those derived
tables and not trial-level optogenetic recordings.

Phase 4 therefore does not claim an exact reproduction.  It builds an
independent, prespecified, animal-disjoint benchmark with:

- the archive's orientation-corrected `dv` field as the D/V signal
  (positive = dorsal);
- four-frame reversal segments ending at least two frames before the reversal
  offset;
- a post-offset label from the mean `dv` over the first two frames;
- a behaviour/context baseline that includes head state, velocity, angular
  velocity, pumping, odour context, event duration and time-to-offset;
- neural population, SAAV-only, prespecified circuit and identified-neuron
  representations;
- leave-one-animal-out fitting, train-only scaling and identified-label
  coverage;
- animal-cluster bootstrap uncertainty, calibration, and event-level metrics.

## Hypotheses and decision rules

### H1: The target was inappropriate

If the Phase 3 null was caused by predicting a generic reversal state rather
than a behaviourally meaningful turn direction, then a turn-direction target
should show a neural increment even when behaviour is controlled.

**Falsifier:** the primary animal-disjoint AUROC increment for
`behaviour + population` over `behaviour` has a 95% animal-cluster CI that
includes zero, and the same is true for the prespecified SAAV representation.

### H2: The representation was insufficient

If linear population summaries missed the signal, a nonlinear model or
identified-neuron features should recover it.

**Falsifier:** a secondary gradient-boosted tree model and the identified /
circuit representations also fail to show a positive increment.  A remaining
limitation is that the paper's exact RNN and alignment are not reproduced.

### H3: Distribution shift explains the discrepancy

If the effect exists only in a subset of animals, odour contexts or the mutant
cohort, cohort-specific and external-cohort analyses should reveal it.

**Falsifier:** no cohort or external transfer shows a robust, uncertainty-
bounded increment.  A cohort-specific exploratory signal is reported as
hypothesis-generating, not as confirmation.

### H4: There is no prospectively decodable signal at this resolution

This is not proven by a null.  The correct statement is a bounded negative:
the current archive, representation, target and animal-disjoint design do not
provide evidence for an incremental prospective signal.  The report must state
the achieved sample size and the uncertainty.

## Primary design (P3)

- **Cohorts:** wild-type octanol and wild-type diacetyl; 56 animals.
- **Events:** reversals with at least 12 frames and a valid two-frame
  post-offset mean.  Primary analysis: 945 events, 186 dorsal, 759 ventral.
- **Primary representation:** four-frame middle window.  For a reversal of
  length L, the segment starts at `start + max(1, (L-4)//2)` and satisfies
  `seg_end <= end - 2`, i.e. it ends at least two frames before the offset.
  Features therefore never include the outcome window or the two-frame guard.
  Completed-reversal duration, time-to-offset and post-segment odour are
  excluded from the behaviour features because they are future information.
- **Selection caveat:** the segment is event-aligned, as in the paper's offline
  task, rather than triggered online at reversal onset.  Every feature frame is
  pre-offset, but this is not a real-time decoder.
- **Primary model:** StandardScaler + logistic regression, fixed `C=0.1`,
  `lbfgs`, no class weighting, no test-time calibration.
- **Primary metric:** pooled out-of-fold AUROC, with animal-cluster bootstrap
  (2,000 resamples) for paired increments over the behaviour baseline.
- **Secondary metrics:** AUPRC, log loss, Brier score, balanced accuracy at
  0.5, accuracy, calibration slope/intercept, per-animal AUROC, calibration
  curve, and event-level predictions saved as parquet.
- **Primary contrast:** `behaviour + population` minus `behaviour`.
- **Prespecified neural contrasts:** SAAV-only, eight-neuron circuit, full
  identified-neuron vocabulary.

## Controls

All controls are model-dependent and cannot substitute for experimental
perturbation:

1. Behaviour-preserving neural time shift (50 frames earlier).
2. Circuit identity permutation within each animal.
3. Within-animal label permutation, preserving class counts and behaviour.
4. Matched random circuit-feature ablation and prespecified SAAV/circuit
   ablations.
5. Overlap audit for adjacent reversal events.
6. Multiple-testing accounting for the secondary neural contrasts.

## Secondary and descriptive analyses

- Early and late four-frame windows.
- Cohort-specific models (octanol, diacetyl).
- SAAV-available subset (27 animals).
- Within-animal five-fold event split, reported only as a comparison.
- External transfer: octanol → diacetyl, WT → tdc1, octanol → tdc1.
- Time-resolved decoding from −12 to +12 frames around the offset; frames at
  or after 0 are retrospective controls.
- PCA population-state trajectories and descriptive peak times of the
  prespecified circuit neurons.
- Mutant tdc1 as a distribution-shift cohort, not as a causal perturbation.
- No virtual perturbation is presented as an intervention result.

## What would change the conclusion

- A prespecified, animal-disjoint positive increment with a CI excluding zero
  in an independent cohort.
- Replication of the exact published preprocessing/RNN on the released
  derived CSVs, if those files become available.
- Trial-level optogenetic or mutant recordings with animal-disjoint,
  event-level controls.

## Scope decision for 6-ECTS

Do **not** scale the current result into a 6-ECTS causal or whole-brain
intervention claim.  The defensible 6-ECTS scope is a methods-and-null-result
package plus a focused follow-up: balanced animal-disjoint cohorts, a
sequence model matched to the paper, and trial-level perturbation data.  A
6-ECTS expansion aimed at neuron-specific control would be premature without a
prospectively validated signal.
