# Questions for Denis Kovacevic

> **Supporting questions, not the canonical scope.** The canonical decision surface is the
> **Scope & decisions** page and [`DENIS_REQUIREMENTS_BRIEF.md`](DENIS_REQUIREMENTS_BRIEF.md).
> Record only Denis's written answers; do not infer them. Public sources are audited in
> [`DENIS_PUBLIC_SOURCE_AUDIT.md`](DENIS_PUBLIC_SOURCE_AUDIT.md).

> These are questions, not reconstructed requirements. The earlier conversation
> was reconstructed provisionally in `docs/ORIGINAL_RESEARCH_OBJECTIVE.md`; do
> not treat that reconstruction as a confirmed brief. Please correct any premise
> that is wrong.

## 1. The central scientific target

1. Is the primary target **trajectory simulation from an observed initial
   state**, or **prediction of a specific behavioural event** (for example
   reversal onset)? This branch treats simulation as primary and decoding as
   supporting evaluation.
2. What would count as "reproducing branching" for you: a conditional outcome
   distribution, a visible split in a low-dimensional projection, a
   parameter-dependent change in a vector field, or something else?
3. Is a **model-dependent virtual perturbation** useful for your purpose even
   when it is not causally validated? If yes, what decision would it inform?
4. Which named neurons are the priority for perturbation, and why? This branch
   used AVBL, AVBR, AVEL, AVER, RIVL, RIVR as a literature-motivated hypothesis
   set, not because you specified them.

## 2. Data and validation

5. Is there an **independent cohort** of Kato-like recordings, or a second
   laboratory/preparation, that can be held out?
6. Is there a **stimulation-response dataset** with matched neuron identities
   (optogenetic, histaminergic, mechanical or thermal)? If so, what is the
   stimulus timing, amplitude calibration and behavioural readout?
7. Are **video-tracked or manually scored behaviour labels** available? The
   current labels are decoded from the same neural population, which limits any
   behavioural claim.
8. Should unnamed traces be retained? They are 60-70% of the segmented traces;
   this branch models only named predictor traces plus the unnamed predictor set
   that survives preprocessing.
9. What is the acceptable definition of a "held-out worm"? Is leave-one-worm-out
   enough, or is a separate cohort required?

## 3. Model and evaluation

10. Should model selection use a **free-running validation objective** rather
    than one-step RÂ²? The current frozen rule selected a model that is worst on
    free-running test error.
11. Is VAR an acceptable descriptive baseline, or does the project require a
    mechanistic/neural-network model? VAR wins the free-running table but was
    not selected.
12. What horizon matters: 5, 10, 20, or the full inter-reversal interval?
13. Should uncertainty be evaluated per worm, per event, or pooled? The current
    design has five worms and four to seven reversal runs per worm.
14. Is a latent representation required to be invertible? BunDLe-Net has no
    published inverse decoder, so this branch applies perturbations in neuron
    space and projects only for display.
15. What would make a perturbation result **specific** in your view: a
    random-neuron control, a matched off-target control, a dose-response, or
    replication in an independent animal?

## 4. Deliverables and constraints

16. Do you need a specific checkpoint format, model card schema, or interface
    (Python API, CLI, notebook, app tab)?
17. Is CPU-only execution a hard constraint? The current runs are CPU PyTorch.
18. Should the simulator be distributed as code plus metadata only, or must
    trained weights be shipped? Binary `.pt` files are currently excluded from
    Git to avoid corruption; the training commands regenerate them.
19. Which claims must never appear in a paper? This branch forbids
    "bifurcation", "causal", "whole-brain", "digital twin", "BCI" and
    "validated perturbation" unless new evidence is supplied.
20. Is there a deadline or a required comparison against a specific published
    model (BunDLe-Net, NC-MCM, or another)?

## 5. What this branch would need to go further

- A confirmed statement of the primary target and the branching definition.
- One independent cohort or one matched stimulation dataset.
- A decision on free-running versus one-step model selection.
- A decision on whether VAR is an acceptable primary descriptive model.
- A named-neuron priority list and a device amplitude calibration, if a
  biological perturbation is intended.

Until then, Gate 6 remains `not_evaluable` and every perturbation output carries
the standing warning.

