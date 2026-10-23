# Related work: NEBULA and the Phase 2 evidence standard

**Status:** literature positioning, not a reproduction.
**Compiled:** 2026-10-23.

> NeuroBranch does **not** implement NEBULA. No NEBULA result was reproduced,
> and no comparison number is claimed. This note records what NEBULA claims,
> why it is the closest current work, and which evaluation gates a NEBULA-style
> model must pass before its in-silico interventions can be called biological.

## 1. What NEBULA is

**NEBULA** = *NEural and Behavioral modeling through Unified LAtent dynamics*
(Nuzzi, Mattia & Pezzulo, bioRxiv 2026,
[doi:10.64898/2026.06.05.730482](https://doi.org/10.64898/2026.06.05.730482),
CC-BY-NC).

From the abstract:

- a generative framework that **jointly** models whole-brain neural activity and
  behaviour;
- applied to brain-wide *C. elegans* recordings;
- identifies a shared low-dimensional dynamical manifold underlying both neural
  activity and behaviour;
- supports long-horizon generation and **targeted in silico interventions**;
- perturbations reveal behaviourally relevant transition points, and “steering”
  interventions manipulate neural and behavioural states without retraining;
- framed as a foundation for virtual experimentation / digital twins.

The last two points are the closest match to NeuroBranch's stated goal. They are
also the points where the epistemic standard must be highest: an in-silico
steering result is a property of the fitted generative model until it is tested
on animals.

## 2. How NeuroBranch differs

| Axis | NEBULA (abstract claims) | NeuroBranch Phase 2 |
|---|---|---|
| Model | Joint generative latent dynamics of neural activity **and** behaviour | Separate dynamics models + separate behaviour decoder |
| Behaviour | Modelled jointly, part of the latent manifold | Decoded post hoc from AVA/SMD/RIB-derived labels; reporter cells excluded |
| Long-horizon generation | Explicit claim | Open-loop rollouts collapse; no long-horizon claim |
| In-silico intervention | Steering / perturbation of learned dynamics | Paired virtual perturbation with matched controls and OOD flags |
| Causal validation | Not claimed in the abstract; in-silico | Explicitly **not causally validated**; external optogenetic benchmark is predictive only |
| Evaluation | Not independently reproduced here | Preserved negative results, corrected branching test, specificity audit |
| Data | Brain-wide *C. elegans* | Kato 2015 (5 worms) + Randi 2023 (110 animals) |

NEBULA is a **stronger model class** than anything in Phase 1/2: joint
brain–behaviour latents could, in principle, resolve the circularity that limits
NeuroBranch's behaviour decoder. That is a hypothesis, not a result reproduced
here.

## 3. Why NEBULA does not rescue the Phase 2 negatives

1. **Joint modelling is not causal identification.** A shared latent can fit
   behaviour and neural activity without either causing the other. NEBULA's
   steering is still an intervention in the model.
2. **The Phase 2 optogenetic benchmark is external and predictive.** A joint
   latent trained on Kato has not been tested on Randi's held-out animals. Until
   it is, its response predictions are unmeasured.
3. **Branching remains a projection property.** A generative model can produce
   beautiful trajectories; the corrected audit shows that observed local entropy
   is often *below* a geometry-preserving null, which is predictability, not
   branching.
4. **The intervention specificity failure is model-class agnostic.** Any model
   that maps a +2 SD perturbation to a large readout change must show that
   arbitrary cells do not do the same. NEBULA-style steering would need the same
   matched-control audit.

## 4. Evaluation gates for a NEBULA comparison

Before claiming that NEBULA (or a NEBULA-like model) improves on NeuroBranch:

1. **Code and weights available at a pinned commit**, with a license permitting
   evaluation.
2. **Strictly held-out animals**, not frames or windows from the same recording.
3. **Causal filtering**; no test frames in the encoder or in a zero-phase filter.
4. **Constant-mean and persistence baselines** on the same split.
5. **Calibration**, not only trajectory aesthetics: log loss, Brier, rollout
   error, and transition accuracy.
6. **External optogenetic test** on Randi held-out animals, with event-level
   paired CIs and the reliability ceiling reported.
7. **Matched-control virtual perturbations**, with a pre-registered candidate
   set and a max-statistic correction.
8. **Ablation of the joint behaviour loss** to test whether the behaviour channel
   actually adds neural predictive value.
9. **Uncertainty** over seeds and animals; no single-run steering demo.

None of these gates is met by the Phase 2 artifacts for NEBULA. The correct
statement is: *NEBULA is the most relevant current framework, but it has not been
independently evaluated here, and its in-silico interventions are not biological
validation.*

## 5. Other anchors

| Work | Role in Phase 2 |
|---|---|
| [Kato et al., *Cell* 2015](https://doi.org/10.1016/j.cell.2015.09.034) | Primary observational dataset; behaviour labels derived from AVA/SMD/RIB |
| [Randi et al., *Nature* 2023](https://www.nature.com/articles/s41586-023-06683-4) | External optogenetic propagation atlas; response benchmark |
| [Kumar et al., BunDLe-Net, bioRxiv 2023](https://doi.org/10.1101/2023.08.08.551978) | Latent representation baseline; official artifact is transductive |
| [Grosse-Wentrup et al., NC-MCM, *PLoS Comput Biol* 2024](https://doi.org/10.1371/journal.pcbi.1012674) | Discrete cognitive-state abstraction; Markov test ported, not retrained |
| [Atanas et al., *Cell* 2023](https://doi.org/10.1016/j.cell.2023.07.035) | Natural external behaviour dataset; not acquired in Phase 2 |
| [Witvliet et al., *Nature* 2021](https://doi.org/10.1038/s41586-021-03778-8) | Connectome reference used for edge-count features |
| [Morrison & Young, *PLoS Comput Biol* 2025](https://doi.org/10.1371/journal.pcbi.1013818) | Connectome-constrained premotor model; mechanistic alternative |
| [Costa et al., *PNAS* 2024](https://doi.org/10.1073/pnas.2318805121) | Behaviour-only Markov model; warns that behavioural Markov structure does not imply a neural latent |

## 6. Bottom line

NEBULA makes the strongest version of the claim NeuroBranch was built to test:
that a low-dimensional joint latent can generate and steer whole-brain behaviour.
Phase 2 does not refute that claim. It supplies the evaluation discipline the
claim needs: held-out animals, causal filtering, constant baselines, external
optogenetic data, matched controls, and explicit non-causal labels.
