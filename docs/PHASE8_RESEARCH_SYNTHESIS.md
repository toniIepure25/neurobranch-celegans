# Phase 8 research synthesis — from target acquisition to an adaptive twin

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

## Why Phase 7 was not the end

Phase 7 treated a target as the unit of acquisition and asked which target to
probe next. That is a useful benchmark, but it leaves three scientific gaps:

1. **The animal is not modelled.** A response operator that is good on average
   across animals can still be badly calibrated for the next animal.
2. **A scalar endpoint hides dynamics.** Optogenetic responses have a time
   course; a scalar mean can be right while the trajectory is wrong.
3. **Acquisition is not personalised.** Choosing a target globally is different
   from choosing a *support set* that calibrates a particular animal.

Phase 8 addresses these gaps with an animal × support-set episode design.

## Relevant literature and boundary

- Randi et al. (2023), whole-brain calcium imaging with an optogenetic atlas:
  <https://arxiv.org/abs/2208.04790>. This is the training cohort; the present
  work is a secondary analysis, not a new wet-lab experiment.
- Dvali et al. (2025), neural response prediction / perturbation modelling:
  <https://arxiv.org/abs/2412.14498>.
- Wagenmaker et al. (2024), active exploration and information-directed
  acquisition: <https://arxiv.org/abs/2412.02529>.
- OPhELIA (2026), active connectomics / optogenetic experiment design:
  <https://arxiv.org/abs/2607.12930>. Active connectomics is not new; APT does
  not claim to be the first active-connectomics method.
- CS-GPFA (2024), low-rank latent dynamics:
  <https://arxiv.org/abs/2410.19153>.
- TBFM (2026), meta-learning for time-series foundation models:
  <https://arxiv.org/abs/2608.26649>.
- BraInCoRL (2025), brain-inspired continual representation learning:
  <https://arxiv.org/abs/2505.15813>.
- NeuroLens (2026), neural surrogate modelling:
  <https://arxiv.org/abs/2610.02864>.
- Mus siliconus (2026), digital-twin perspectives:
  <https://arxiv.org/abs/2609.02243>.

The novelty boundary is deliberate:

- **Not** the first active connectomics method (OPhELIA exists).
- **Not** a wet-lab closed loop; no new animals were stimulated.
- **Not** a causal graph or causal effect estimate.
- **Not** cross-dataset transfer; no same-task external optogenetic cohort is
  available in this repository.
- **Not** a foundation-model claim on Randi alone.
- **Not** a validated digital twin.

## What APT contributes

1. A **support-set personalisation** protocol where support and test targets are
   disjoint; the model cannot copy a same-target response.
2. A **time-resolved** endpoint in addition to the scalar endpoint.
3. An **active support-selection** comparison at a fixed budget, with random,
   uncertainty, diversity, ensemble-disagreement (`eig` proxy) and hybrid arms.
4. A **sequence rollout** readout that is prospective in event order but uses
   only previously observed outcomes.
5. A **leakage/identifiability audit** and frozen decision gates, including a
   planned negative-result interpretation.

## Identifiability caveat

Phase 3 pair means show a small animal-effect variance component (η² ≈ 0.00054)
relative to pair identity/interaction. If that estimate holds, a support set can
only calibrate a small animal-specific offset, and H2/H3 may be underpowered.
This was recorded before the full run. A null H2/H3 is therefore an informative
bound on animal-level calibration in this atlas, not evidence that
personalisation is impossible in principle.
