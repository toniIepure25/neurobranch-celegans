# Phase 3 next experiments

Prioritised by **decision value per unit effort**. The first block is feasible
within a 6-ECTS project using existing data; later blocks need a wet lab and a
longer collaboration.

## P0 — replicate the fragile primary result

**Question:** does the animal-disjoint neural increment at 6 s replicate in a
second, independent freely-moving cohort?

- Same target: reversal onset in `(t, t+6 s]`; same eligible-frame rule.
- Same primary model: L2 logistic, C=0.1, causal population features, LOAO.
- Preregister the cohort, exclusion rules and primary CI before scoring.
- Minimum useful sample: 25–30 animals with ≥20 events each; report the CI, not
  only the point estimate.
- Decision: CI upper < 0 and ≥70% animals improved = replicated; otherwise stop.

**Effort:** 40–60 h (data curation + run + write-up).
**Risk:** cohort differences in behaviour/event rate; predefine a prevalence
sensitivity analysis.

## P1 — a behavioural baseline that could explain the signal

**Question:** is the neural increment just nonlinear behavioural/arousal state?

- Add a learned behaviour encoder (temporal CNN/GRU) trained animal-disjoint on
  behaviour only.
- Add explicit arousal proxies available in the export (pumping, body angle,
  distance from copper, recent event history).
- Add session-drift features that use only past frames.
- Compare neural increment against this strong baseline, not the linear one.
- Decision: if the increment disappears, the parsimonious explanation is
  behavioural state, and the neural claim is withdrawn.

**Effort:** 25–35 h.
**Risk:** overfitting; use train-only fitting and animal-disjoint evaluation.

## P2 — identity reproducibility

**Question:** are the 52 high-coverage NeuroPAL labels stable enough to support an
identity-level claim?

- Quantify per-label coverage, confidence and duplicate rate across animals.
- Test a probabilistic identity representation (soft assignment with uncertainty)
  without transductive fitting on the held-out animal.
- Compare against permutation-invariant population summaries on identical splits.
- Decision: identity must beat population summaries with a CI excluding zero;
  otherwise keep the population-level conclusion.

**Effort:** 20–30 h.
**Risk:** the JLD2 export is probabilistic; a 52-label intersection may be too
small.

## P3 — Randi window/target sensitivity

**Question:** is the r≈0.09 repeatability a window/aggregation artifact?

- Prespecify a grid of pre/post windows and baseline estimators.
- Report reliability for peak, mean, integral and full shape.
- Use repeated-event averaging only where at least 3 trials exist.
- Test whether a longer window or a deconvolution-based response improves
  reliability.
- Decision: a target with median r≥0.3 and a repeated-average MAE better than
  global mean would justify a response model. Otherwise retain the negative.

**Effort:** 15–25 h (existing data).
**Risk:** multiple comparisons; preregister one primary target/window.

## P4 — cross-animal dynamics with a stronger model

**Question:** does a sequence model (GRU/transformer) generalise across animals
better than the one-step ridge?

- Train on 26 animals, test on 1; causal inputs only.
- Compare persistence, behaviour-only sequence model and behaviour+neural.
- Evaluate short rollouts (5–20 frames), not only one step.
- Report per-animal error growth; a model that is only better at one step is not
  a simulator.
- Decision: neural model must beat behaviour-only on held-out rollout error with
  animal-level uncertainty.

**Effort:** 30–50 h.
**Risk:** small cohort; sequence models overfit animal identity.

## P5 — causal intervention (medium/long collaboration)

**Question:** does activating/inhibiting neuron X change transition T?

Minimum design:

- cell-specific bidirectional opsin, verified expression;
- randomised interleaved mock/light-only/off-target controls;
- dose–response and temporal-window manipulation;
- blinded/automated behaviour scoring;
- animal-level replication and a manipulation check;
- mediation measurement of a downstream population;
- preregistered primary outcome and analysis.

**Effort:** 6–12 months, new rig and animals.
**Risk:** effect may be state-dependent; a null is informative but expensive.

## P6 — closed-loop control (long)

Only after P5 shows a reliable, specific effect:

- real-time inference of the pre-transition state;
- random-access stimulation in the claimed window;
- closed-loop vs open-loop vs random-stimulation controls;
- test whether stimulation **causes** the transition rather than correlating with
  it;
- independent replication.

This is a thesis-scale programme, not a 6-ECTS project.

## What not to do next

- Do not add more secondary representations and select the one with the smallest
  p-value.
- Do not train on the held-out animal or use whole-recording z-scores.
- Do not call a model perturbation an experiment.
- Do not use Randi held-out-event performance as evidence for a new intervention.
- Do not present a 3-D manifold as a mechanism.

## Recommended sequence

1. P0 replication (decision: is the direction real?).
2. P1 strong behavioural baseline (decision: is it neural?).
3. P2 identity (decision: is it cell-specific?).
4. P3 Randi sensitivity (cheap, independent).
5. P5 only if 1–3 justify a causal hypothesis.

All virtual work remains:

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**
