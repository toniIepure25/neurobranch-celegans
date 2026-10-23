# Phase 3 supervisor brief

**To:** supervisor / collaborator
**From:** NeuroBranch Phase 3
**Date:** 2026-10-23
**Branch:** `research/phase3-neural-generalization` (local; not pushed)
**Starting point:** Phase 2 `c69fd5afe1a084827ce0c64f3f2c96fa0679b077`

## 1. One-paragraph summary

Phase 3 tested whether recorded neural activity improves prospective prediction of
a real behavioural transition on an independent, freely-moving dataset
(Baskoylu & Flavell 2026; 27 animals, 1,162 reversals), and whether the Randi
optogenetic atlas supports better response modelling. The data pipeline and
animal-disjoint benchmark are sound. The **primary neural increment is
inconclusive**: adding population neural activity to behavioural history improves
mean log loss by 0.0070 at 6 s, but the 95% animal bootstrap CI includes zero
([-0.0148, +0.0005]). A secondary train-only PCA representation clears the
zero-CI rule, but the effect shrinks and loses significance against a quadratic
behavioural baseline, is absent for identity-aligned neurons, and reverses in the
within-animal forward task. Randi target changes do not rescue a repeated-response
reliability of r≈0.09; repeated-event averaging is worse than a global mean.
There is **no evidence for neuron-specific control**, and no new causal
experiment was performed.

## 2. What was delivered

| Deliverable | Status |
|---|---|
| Phase 3 research plan / preregistration | done before test metrics |
| Baskoylu dataset download, SHA256 manifest, HDF5/JLD2 audit | done |
| NeuroPAL identity export and ambiguity handling | done |
| Prospective animal-disjoint benchmark (3/6/12 s) | done |
| Frozen representation comparison + negative controls | done |
| Randi target / repeated-average / context / shape audit | done |
| Cross-animal one-step dynamics probe | done |
| Intervention-limitations and causal-evidence checklist | done |
| Streamlit Prospective Behavior tab (8 tabs total) | done |
| Tests | Phase 3 tests pass; full suite re-run before handoff |
| README / RESULTS / SUPERVISOR_BRIEF / POTENTIAL_PAPER / REPRODUCING | updated |

## 3. Headline numbers

### Prospective behaviour (animal-disjoint, primary 6 s)

| Model | Log loss | Δ vs behaviour | 95% CI | Animals improved |
|---|---:|---:|---|---:|
| Behaviour history | 0.5927 | — | — | — |
| + population neural | 0.5856 | -0.00701 | [-0.01483, +0.00053] | 20/27 |
| + PCA(10) neural | 0.5842 | -0.00841 | [-0.01475, -0.00220] | 20/27 |
| + identified 52 neurons | 0.5909 | -0.00172 | [-0.01054, +0.00699] | 13/27 |
| Causal delayed neural (211 frames) | 0.5892 | -0.00345 | [-0.00851, +0.00175] | 18/27 |

The prespecified criterion (CI upper < 0, ≥70% animals) is **not met** for the
primary model. The direction is consistent at 3/6/12 s, but the CI is fragile.

### Key controls

- Quadratic behavioural state: Δ shrinks to -0.00524, CI [-0.01334, +0.00301].
- Within-animal forward: Δ +0.00288 (neural worse), 9/27 improved.
- C sensitivity: direction stable, significance only at C=0.3 (not prespecified).
- Causal delayed neural (211 frames): Δ -0.00345, CI includes zero; delayed
  activity still carries slow behavioural/arousal information, so this is not a
  clean null and weakens a transition-specific interpretation.

### Randi

- Reliability: median r 0.090 (peak), 0.092 (mean/integral).
- Temporal shape: per-neuron r 0.048; population time-course r 0.126; sign
  agreement 0.518.
- Repeated averaging: global MAE 5.9008 is better than pair average 6.3238.
- Context: MAE 5.9967 → 5.8900 (1.8%), RMSE 60.78 → 49.91; the large
  correlation gain is response-scale/outlier driven.

### Cross-animal dynamics

Persistence next-velocity MAE 0.02722; behaviour ridge 0.02587; + neural 0.02590.
Next-state log loss behaviour 0.11631; + neural 0.11655. Neural activity does not
add generalisable one-step dynamics. No cross-animal generative simulator is
claimed.

## 4. What is genuinely new

1. A reproducible, event-index-validated adapter for a 2026 freely-moving
   whole-brain dataset, including a documented JLD2 identity workaround.
2. A leakage-safe animal-disjoint prospective benchmark with causal features,
   proper scores, calibration, animal bootstrap and explicit null controls.
3. An honest test of whether the Kato negative pattern reproduces on independent
   behaviour: it does.
4. A target/shape/repeated-average audit of Randi that explains why the Phase 2
   r≈0.09 is not a modelling bug.
5. A causal-evidence checklist that separates observed-response prediction from
   intervention generalisation and control.

## 5. What did not work / what is still uncertain

- The population neural increment is not robust to a strong behavioural baseline.
- Identity-aligned neurons do not provide a better representation at this
  coverage (52 labels in ≥80% of animals).
- Within-animal forward prediction does not show the animal-disjoint direction.
- Randi repeated responses are not reliable enough for per-pair averaging.
- The context model's RMSE gain is not matched by a meaningful MAE gain.
- The 27-animal cohort is one preparation; no cross-lab replication.

## 6. Why this is a useful negative/inconclusive result

The project was designed so that a negative result is publishable as a rigorous
benchmark, not hidden. It rules out several easy explanations (event-index bug,
whole-recording z-score leakage, transductive splits, trivial time alignment) and
shows that the remaining neural signal is small and compatible with nonlinear
behavioural/arousal state. That is more useful to the lab than a fragile positive
claim that would not replicate.

## 7. 6-ECTS scope (realistic)

**Goal for a 6-ECTS project (~150–180 h):** determine whether the small
animal-disjoint neural direction is real and whether it survives a strong
behavioural baseline, using existing data and at most a modest new cohort.

Work packages:

1. **Replication cohort (40–60 h):** acquire/curate a second freely-moving
   dataset with the same target; preregister the 6 s analysis; run the existing
   pipeline unchanged. Decision: does the 95% CI exclude zero?
2. **Strong behavioural baseline (25–35 h):** nonlinear state, arousal proxies,
   session drift, learned behaviour encoder. Decision: does the neural increment
   survive?
3. **Identity reproducibility (20–30 h):** quantify label coverage/confidence
   across animals; test a probabilistic identity representation without
   transductive fitting. Decision: does identity add over population summaries?
4. **Randi window/target sensitivity (15–25 h):** prespecify windows and targets;
   test whether r≈0.09 is a window artifact. No new animals.
5. **Write-up / reproducibility (20–30 h):** manuscript, code, figures, supervisor
   review.

**Out of scope for 6 ECTS:** new optogenetics rig, closed-loop control,
cell-specific manipulation, cross-lab replication. Those belong to a longer
collaboration.

## 8. Longer collaboration (12–24 months)

- Closed-loop, cell-specific activation/inhibition with random-access illumination.
- Bidirectional manipulation, dose–response, off-target and light-only controls.
- Simultaneous imaging of candidate and downstream population; mediation analysis.
- Cross-lab replication and a causal model trained on intervention data.
- A validated cross-animal identity/registration method.

## 9. Decisions requested from supervisor

1. Approve the **negative/inconclusive** framing; do not spin the PCA secondary
   result as a positive.
2. Choose between (a) second observational cohort replication and (b) a stronger
   behavioural baseline as the first 6-ECTS work package. I recommend (a) first,
   because it directly tests the fragile CI.
3. Confirm that no wet-lab intervention is expected within 6 ECTS.
4. Confirm the branch remains local until the supervisor authorises a push.

## 10. Execution caveats

- The matched pre-event analysis in plan §9 was not run as a separate analysis;
  the prospective horizon benchmark is the implemented transition test.
- The temporal control was corrected to a causal delay; it is not a clean null.
- RQ5 is one-step, not a multi-step generative generalisation test.
- No new animals or wet-lab interventions were performed.

## 11. Bottom line

Phase 3 is a rigorous observational benchmark with a small, fragile neural lead
and a clear causal boundary. It does **not** show that a neuron controls a
transition. It does show what would be required to find out.

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**
