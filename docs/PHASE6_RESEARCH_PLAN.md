# Phase 6 research plan and positioning

## 1. What the field already has

- **Randi et al. (2023)**, *Nature* — a whole-brain optogenetic signal-propagation
  atlas in *C. elegans*. The key resource is a target→responder response table,
  not a causal graph.
- **Kato et al. (2015)**, *Cell* — spontaneous whole-brain calcium dynamics with
  behaviour states; useful for dynamics, not for controlled perturbation.
- **Kramer et al. (2026)**, *Nature Neuroscience* — directed turning and
  sensorimotor behaviour; a behavioural cohort, not a stimulation atlas.
- **Baskoylu & Flavell (2026)** — independent behaviour cohort; no controlled
  single-cell perturbation.
- **SBTG (arXiv:2605.02852)** — a causal-discovery competitor on time series.
  It is designed for spontaneous trajectories; it does not solve the
  target-generalisation problem of a stimulation atlas by itself.
- **Homogenized *C. elegans* dataset (arXiv:2411.12091)** — ~900 worms, ~250
  neurons, nine connectomes; promising external cohort but a different task.
- **CAPT (arXiv:2607.23258)** and **iMINDBench (arXiv:2609.18104)** — evidence
  that cross-dataset calcium/neural transfer and protocol-invariant evaluation
  are open problems, not solved defaults.
- **EEG-FM-Audit (arXiv:2605.26910)** — motivates evaluation cards and leakage
  audits for neural foundation models.

## 2. The gap Phase 6 targets

Most atlas analyses answer one of two easier questions:

1. *matrix completion*: predict a measured pair from other measured pairs;
2. *descriptive anatomy*: compare a response to the connectome.

The hard question is **inductive target generalisation**: predict the response
profile of a neuron that has **no trained target embedding**, using only its
anatomy and the stimulation context. That is the question the CRO is built for.

A second gap is **measurement-error honesty**. A split-half profile reliability
near 0.06–0.09 means an absolute correlation threshold can be simultaneously
“small” and close to the ceiling. Phase 6 therefore reports both the raw metric
and an attenuation ceiling.

## 3. What is new in this implementation

1. **Feature-conditioned low-rank operator.** Target/responder factors are partly
   produced by anatomical MLPs; a held-out target uses its connectome row, not a
   lookup embedding.
2. **Leakage-safe grouped splits and audit.** Five target folds, validation
   targets treated as unseen, no post-stimulus `global_delta`, no test scaling.
3. **Reliability-normalised profile metric.** Split-half Spearman–Brown ceiling
   turns a raw 0.094 into “≈64% of what is reliably measurable,” with the
   baseline at ≈11%.
4. **Two dissociated read-outs.** Response-profile prediction can be positive
   while anatomical edge recovery is at chance. Phase 6 reports that dissociation
   instead of claiming a causal atlas.
5. **Model-dependent experiment designer.** Ranks targets by predicted effect,
   interval width and animal coverage; every output is a hypothesis, not an
   intervention.
6. **External-transfer contract.** A validator and schema for a real external
   optogenetic cohort, with an explicit refusal to substitute a different task.

## 4. What would make this revolutionary rather than incremental

The next level is **prospective, pre-registered perturbation selection**:

1. use the CRO + designer to choose 10–20 under-sampled targets;
2. stimulate them in new animals, with predicted-responder and predicted-null
   controls;
3. test whether the observed profile beats the model's own prediction and the
   additive baseline;
4. update the operator only after the prospective test is frozen.

That would convert a retrospective benchmark into an **experimental-design
engine** with a falsifiable out-of-sample claim. It requires wet-lab data that do
not exist in this checkout; Phase 6 delivers the computational half and refuses to
pretend otherwise.

## 5. Non-claims

- No foundation-model claim.
- No causal claim from observational calcium.
- No cross-dataset transfer claim.
- No virtual SAAV intervention without interventional validation.
- No exact reproduction of the paper's 23,433-pair filtered table.

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**
