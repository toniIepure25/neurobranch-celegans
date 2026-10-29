# Phase 8 — Adaptive Perturbation Twin (APT)

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

## What the model is

APT is an animal-conditional, time-resolved response operator. For an event
targeting neuron *t* and a recorded responder *r*, it predicts

- a scalar post-stimulus response, and
- a 10 s trajectory sampled at 0.5 s.

The model is deliberately anatomy-first: target and responder representations
come from the Phase 3 anatomy/connectome feature bundle, so an unseen target
still has a representation. A low-rank target×responder interaction, a row MLP,
and an 8-function temporal basis produce the prediction. A StudentT likelihood
(df=5) is used because calcium responses have heavy tails.

## Support-set personalisation

A held-out animal provides a small support set of observed events. APT
summarises those events with a permutation-invariant mean/std profile
(post mean/std/abs-max, pre mean/std, scalar mean, responder count, event count,
interval) and passes the standardised profile through a FiLM modulation of the
row representation.

- `k=0`: the support vector is exactly zero and is masked out. This is the
  prior, not a missing-data accident.
- `k>0`: only support outcomes from that animal are used.
- Support and test targets are disjoint by construction. The model cannot copy
  a test target's response from a same-target support event.

The support profile is fitted only on **training animals**; the held-out
animal's profile is transformed with the frozen scaler.

## Sequence rollout

The H4 exploratory readout is a pseudo-prospective event-order rollout. At each
event the model may use the support set plus the previous three **observed**
test events, then predicts the next trajectory before that event is revealed.
This is not counterfactual imagination and not a closed loop.

## Objectives and evaluation

The scalar and trajectory heads are trained with StudentT negative
log-likelihood. Early stopping uses validation animals only. Evaluation is
paired at the same event/responder cells; uncertainty is an animal-cluster
bootstrap. The preregistered gates are in
`docs/PHASE8_PREREGISTRATION.md`.

## Baselines

- static Phase 6 causal response operator (CRO), trained on the same training
  event roles;
- global training mean;
- per-responder mean trajectory template;
- previous-event persistence in the sequence readout.

## What is novel here

The contribution is the **evaluation design**, not a claim that active
connectomics or optogenetic prediction is new:

1. the experimental unit moves from a target to an animal × support-set
   episode;
2. support-set personalisation is tested under target-disjoint, animal-held-out
   splits with frozen scalers and validation-only calibration;
3. active support selection is compared with random acquisition at the same
   budget;
4. a trajectory endpoint and a sequential rollout are evaluated alongside the
   scalar endpoint;
5. the result includes a leakage/identifiability audit and an explicit
   distinction between model-dependent prediction and causal effect.

## Boundaries

APT is not a validated digital twin. It has no wet-lab closed loop, no causal
graph, no cross-dataset transfer evidence, and no foundation-model claim. The
`eig` acquisition score is a deep-ensemble disagreement proxy, not a formal
posterior expected-information-gain guarantee. The Randi atlas is the only
training cohort; external optogenetic cohorts in this repository do not support
the same-task transfer claim.
