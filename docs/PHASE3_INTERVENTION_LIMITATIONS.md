# Phase 3 intervention limitations and the evidence required for control

**Short answer:** nothing in Phase 1–3 is an intervention experiment. The
repository contains observational recordings, a model, virtual perturbations and
an external optogenetic response benchmark. None establishes that activating or
silencing a named neuron causes a behavioural transition.

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

## 1. What the current evidence is

| Evidence | What it supports | What it does not support |
|---|---|---|
| Kato 2015 recordings | population dynamics and behaviour decoding | causality, control |
| Phase 2 branching audit | observed outcome diversity is below geometry-preserving nulls | "branch points" are causal |
| Phase 2 virtual perturbation | model response to an edited state | biological effect |
| Randi atlas reanalysis | prediction of an **observed** optogenetic response | response to a new intervention |
| Baskoylu 2026 prospective test | weak/inconclusive observational prediction | a transition-controlling neuron |
| NeuroPAL identities | probabilistic cell labels in a subset of animals | ground-truth identity in every animal |

The closest related work, NEBULA, performs in-silico steering. Its results are
model-dependent and were not reproduced as a biological control experiment.
Citing NEBULA does not turn a model perturbation into causal evidence.

## 2. Why prediction is not control

A feature can predict a transition because it is:

- a consequence of the transition already starting;
- a correlate of arousal, locomotion, satiety or session drift;
- a downstream readout of the same latent state;
- a confound of animal identity or preparation.

Phase 3 tried to reduce the first two with causal features, animal-disjoint
splits, nonlinear behavioural controls and time-shifted nulls. It did **not**
establish that any neuron is necessary or sufficient. A predictive coefficient is
not an intervention target.

The distinction used throughout Phase 3 is:

- **A:** predict a response to an intervention that was observed;
- **B:** predict a response to an intervention not represented in training;
- **C:** predict an intervention effect from spontaneous activity alone.

Randi supports some A-level prediction. It does not establish B or C. The
prospective Baskoylu analysis is observational and does not establish any of
A/B/C.

## 3. Minimum experimental evidence before a control claim

A defensible claim that neuron X controls transition T requires all of the
following, in a preregistered design.

### 3.1 Causal manipulation

- Cell-specific activation and inhibition of X (e.g. ChR2/Arch or a comparable
  bidirectional tool), not a model edit.
- Independent replication of the effector and driver lines.
- Measured opsin expression and illumination volume; report off-target tissue.
- Randomized trial order, interleaved mock/light-only and off-target-neuron
  controls.
- Blinded behavioural scoring or an automated scorer validated against blind
  human labels.

### 3.2 Necessity and sufficiency

- **Sufficiency:** activation of X increases T above the matched control.
- **Necessity:** inhibition or ablation of X reduces T.
- **Specificity:** a neighbouring/off-target neuron and light-only condition do
  not produce the same effect.
- **Dose–response:** effect size varies with illumination power/duration; report
  the full curve, not one chosen point.
- **Temporal window:** the effect depends on stimulation in the claimed window,
  not on any stimulation at any time.

### 3.3 Behaviour and statistics

- Transition defined before the experiment, with inter-rater reliability.
- Effect size and confidence interval, not only a p-value.
- Animal-level replication; stimulation events nested within animals.
- Power analysis for the smallest effect of interest.
- Pre-registered exclusion rules and primary outcome.
- Negative and null results retained; no optional stopping.
- A manipulation check that the neuron was actually modulated.

### 3.4 Circuit and mediation

- Independent measurement of X activity during spontaneous T.
- A mediator downstream of X is measured and manipulated; the effect is reduced
  when the mediator is blocked.
- Anatomical/connectomic plausibility is supporting evidence, not proof.
- The effect is not explained by a global arousal change, movement artifact or
  heat/light artifact.

### 3.5 Cross-animal generalisation

- The same cell identity is verified in each animal (e.g. NeuroPAL or a stable
  genetic marker).
- The effect replicates across animals and, ideally, across labs/preparations.
- The analysis is animal-disjoint; no transductive fitting on the test animal.

### 3.6 Reporting

- Preregistration with the primary outcome, sample size and analysis plan.
- Raw data, stimulation logs, code and exclusion table.
- A clear statement that a model prediction is not an intervention.
- Independent adversarial reanalysis before a press release or a control claim.

## 4. What would be reasonable next experiments

### Short, feasible (6-ECTS scope)

1. **Replication of the prospective result** on a second freely-moving cohort
   with the same target and an animal-disjoint protocol. This is observational,
   not a control experiment.
2. **A stronger behavioural baseline** (nonlinear state, arousal proxies,
   session drift) to see whether the small neural increment survives.
3. **A measurement study** of the 52 high-coverage identified neurons: are their
   pre-transition profiles reproducible across animals?
4. **Randi target/window sensitivity**: vary the post window and baseline to see
   whether reliability is a window artifact. No new animals.
5. **A registered analysis of existing data** with a fixed target and a
   correction for the number of representations tested.

These are realistic for a 6-ECTS project because they reuse existing data and do
not require a new rig.

### Medium (collaboration)

6. **Closed-loop optogenetic perturbation** of one or two candidate neurons with
   random-access illumination and interleaved controls. Requires a rig, animals,
   effectors and behavioural tracking.
7. **Bidirectional manipulation** (activation and inhibition) with dose–response
   and off-target controls.
8. **Cross-lab replication** with independent animals and a preregistered
   analysis.

### Long (thesis programme)

9. **Circuit-level mediation** with simultaneous imaging of the candidate and
   downstream population.
10. **Cell-resolved identity across animals** with a validated registration
    method, not a 52-label intersection.
11. **A causal model** trained on intervention data, tested prospectively on
    held-out interventions and animals. Until then, the dynamics model remains
    observational.

## 5. What must not be said

- "We found the neuron that controls reversal."
- "The model shows that stimulating X will cause a turn."
- "NEBULA-style steering validates our intervention."
- "The Randi benchmark proves the response generalises to new targets."
- "Neural activity is irrelevant" (the Phase 3 primary result is inconclusive,
  not proof of absence).

## 6. Current status

**No neuron-specific control claim is supported.** The correct Phase 3 output is
an observational, animal-disjoint, properly scored negative/inconclusive result
plus a concrete list of experiments that could, in principle, support a causal
claim later. That is the honest boundary of the present data.
