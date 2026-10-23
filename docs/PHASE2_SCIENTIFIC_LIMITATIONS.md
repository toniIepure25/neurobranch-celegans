# Phase 2 scientific limitations

**Status:** applies to every Phase 2 result. Read with
[`RESULTS.md`](RESULTS.md) and [`LIMITATIONS.md`](LIMITATIONS.md).

## 1. Causal claims

- No animal was perturbed by this project. Every intervention output is a
  **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**
- The Randi dataset is experimental, but the benchmark is **predictive**. A model
  that predicts a response is not a model that can intervene.
- No necessity, sufficiency, dose–response, or off-target claim is supported.
- The Phase 1 RIVL effect fails specificity. Candidate ranking is unstable and
  arbitrary cells can produce equal or larger effects.

## 2. Branching

- The corrected statistic measures local outcome diversity in a **projection**,
  not a bifurcation of the underlying vector field.
- The test has low power for genuine stochastic branching. Synthetic fixtures
  show that when outcomes are not predictable from geometry, shuffling outcomes
  barely changes expected entropy (see `BRANCHING_STATISTICAL_AUDIT.md`).
- Results depend on projection width: worms 2 and 3 have Spearman 0.15–0.38
  between 3-D and 32-D frame scores.
- Only four macro-states; among REV candidates only REV and TURN occur. Entropy
  is bounded by 1 bit and sensitive to class balance.
- Candidate frames overlap; the effective sample size is far below the frame
  count. Block bootstrap is only a partial correction.
- The blocked test block is interpolation. The forward split is not used for the
  branching claim.
- Outcomes are derived from AVA/SMD/RIB activity, which is also present in the
  predictor geometry (the eight reporter cells are excluded, but the labels are
  not independent of the population).

## 3. Optogenetic benchmark

- The response is a simple event-triggered signed peak, **not** the authors'
  response detector or effect-size model.
- The baseline uses a −8…−1 median. Different windows would change absolute
  values, though the model ranking is likely similar.
- Repeated stimulation of the same target has median pairwise r ≈ 0.09. This is a
  hard reliability ceiling: even a perfect predictor of the mean response cannot
  predict single-trial noise.
- The stimulated neuron's own trace is excluded, but indirect illumination and
  movement artifacts may remain.
- 3,371 of 5,808 stimulation events survive unambiguous target filtering. The
  excluded events are not missing at random; ambiguous labels may correlate with
  preparation or anatomy.
- Duplicate canonical labels are excluded rather than resolved. This loses data
  and may bias toward uniquely labelled cells.
- Same cell name ≠ same individual cell. There is no cross-session registration.
- Connectome features are cell-type-level Witvliet edges, not measured
  connectivity in the recorded animal.
- The benchmark does not fit NeuroBranch dynamics to the optogenetic data. It
  tests transfer from pre-stimulus state/identity, not the Phase 1 simulator.
- No behavioural labels are present in the archive.

## 4. Behaviour

- Labels are decoded from AVA/SMD/RIB. Excluding the eight reporter cells reduces
  but does not eliminate circularity.
- The blocked split is temporal interpolation.
- The strict-forward split is dominated by a few states; worm 1 has zero
  transitions in its forward test block. Transition accuracy is undefined or
  extremely noisy.
- Per-worm models, five worms, no external independent behavioural dataset.
  This is **not** the independent behavioural generalization originally
  requested.
- A weak transition signal (17.9–27.8% transition accuracy) coexists with worse
  calibration and worse balanced accuracy. It is not incremental predictive
  value.

## 5. Reproduction

- BunDLe official artifacts are transductive: the encoder saw all frames. Their
  kNN/silhouette numbers cannot be used as held-out evidence.
- BunDLeLite is a PyTorch reimplementation with a causal filter and blocked
  split, not the upstream TensorFlow pipeline. It does not reproduce the
  artifact.
- NC-MCM work is artifact verification plus a ported Markov test, not a full
  pipeline reproduction.
- Upstream code was not executed end-to-end; external repositories were inspected
  at pinned commits.

## 6. Statistics

- Branching p-values are uncorrected across five worms and five null
  specifications. The per-worm pattern is the evidence, not a single p.
- The intervention scan tests 26 candidate cells; a selection-adjusted
  max-statistic is reported precisely because naive per-cell p-values would be
  anti-conservative.
- Randi CIs resample events, but events within an animal are not independent;
  the animal-level fold variability is also reported.
- Behaviour deltas are paired within worm, with five pairs. No claim of
  population-level generalization is made.
- All CIs are percentile bootstrap intervals, not adjusted for model selection.

## 7. Data and provenance

- Raw Kato and Randi data are not committed. Results depend on local files whose
  hashes are recorded.
- OSF metadata can change; the archive hash was checked on 2026-10-23.
- The Randi OSF node exposes no license field. Raw data are not redistributed.
- The connectome reference CSV was converted from UTF-16 to UTF-8; the conversion
  is recorded, and numeric content was checked.
- No remote Git repository is configured; commit provenance is local.

## 8. Scope

- Five Kato worms, one lab, one preparation, one behavioural label definition.
- One optogenetic atlas, one response detector, one connectome reference.
- Results do **not** establish that *C. elegans* lacks branching, that
  optogenetic propagation is unpredictable in general, or that neural activity
  is irrelevant to behaviour.
- They establish narrower claims: under these protocols, these models, and these
  labels, the positive results do not survive.
