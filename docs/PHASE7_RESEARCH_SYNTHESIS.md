# Phase 7 — Research synthesis: from a static atlas to an adaptive measurement policy

**Access date:** 2026-10-25  
**Scope:** literature positioning for a target-level active response-operator benchmark on the Randi *C. elegans* optogenetic atlas. This document is not a claim that the project invented active connectomics.

## The gap after Phase 6

Phase 6 established a real but bounded result. The CRO reached a held-out-target pair Spearman of **0.0984** (95% CI 0.0817–0.1114), compared with **0.0545** for the strongest deployable additive baseline. The preregistered +0.05 gate was not met (Δ = +0.0439, CI +0.0282…+0.0592). The profile result was more interesting: the CRO recovered about **64%** of the split-half reliability ceiling, whereas the baseline recovered about **11%**. Anatomical edge recovery was at chance (AUROC **0.509**), far below the connectome ridge (**0.762**).

The scientific lesson is not “train a larger network.” It is that the atlas is a **fixed sample of perturbations**, and the next question is which perturbation should be measured next. That changes the object from a static response table to an adaptive measurement policy.

## What the literature already establishes

| Source | Contribution | Consequence for Phase 7 |
|---|---|---|
| [Randi et al. 2023](https://arxiv.org/abs/2208.04790) | Whole-brain optogenetic atlas of *C. elegans* target→responder responses | The endpoint is an induced response, not a causal graph. Target-level generalisation is the right test. |
| [Dvali et al. 2025](https://arxiv.org/abs/2412.14498) | Functional signalling modules diverge from the anatomical connectome | A useful acquisition policy should value targets where anatomy is least predictive. |
| [Wagenmaker et al. 2024](https://arxiv.org/abs/2412.02529) | Active learning of low-rank neural population dynamics with holographic optogenetics; up to 2× data reduction | Low-rank response models can make active target selection sample-efficient; the idea is plausible, not guaranteed. |
| [OPhELIA 2026](https://arxiv.org/abs/2607.12930) | Bayesian active photostimulation selection for connectivity maps; roughly 5% of trials can recover an exhaustive map in zebrafish | “Active connectomics” is already a real research direction. Phase 7 must not claim to be first. |
| [CAPT 2026](https://arxiv.org/abs/2607.23258) | Continuous autoregressive population transformer with cross-dataset/cross-species calcium transfer | A foundation-model direction is credible, but it does not solve the perturbation-response problem. |
| [POCO 2025](https://arxiv.org/abs/2506.14957) | Cross-species neural forecasting, including *C. elegans* | Forecasting spontaneous activity is not the same task as predicting an optogenetic response. |
| [SBTG 2026](https://arxiv.org/abs/2605.02852) | Score-block time graphs for causal discovery from dynamics | A causal-discovery competitor, but a different task and timescale. Phase 7 does not claim causal discovery. |
| [TBFM 2025](https://arxiv.org/abs/2507.15274) | State-dependent stimulation-response modelling and meta-learning | Pre-stimulus population state is a biologically motivated conditioning variable. |
| [Simeon et al. 2024](https://arxiv.org/abs/2411.12091) | Homogenised *C. elegans* dataset, many worms and connectomes | External-only; no same-task optogenetic cohort is available for a defensible cross-dataset claim. |

## The defensible novelty

The project should not claim:

- the first active connectomics method (OPhELIA exists);
- a wet-lab closed loop (none was run);
- a causal graph from observational calcium;
- cross-dataset transfer without a same-task external cohort;
- a foundation model.

The defensible contribution is narrower and testable:

> A target-level, measurement-error-normalised, pseudo-prospective benchmark of an active **response operator** on the Randi atlas, with an acquisition objective that explicitly targets **anatomy-unexplained functional coupling**.

Three choices make this different from generic active learning:

1. **The unit is a target, not a row or a photostimulation pattern.** The policy chooses which neuron to stimulate next; all responders for that target are revealed together. This matches the experimental cost structure.
2. **The endpoint is a held-out-target response profile, not an edge label.** The model must generalise to a target with no trained embedding, using anatomy and context.
3. **The acquisition score includes a discrepancy term.** The policy is rewarded for selecting targets whose responses are poorly explained by a simple anatomical prior, not merely for model uncertainty.

## Why not a foundation model now

CAPT and POCO make cross-dataset representation learning plausible. But a foundation-model claim requires either (a) a same-task external perturbation cohort, or (b) a genuinely prospective wet-lab test. Neither exists in this repository. Training a large model on the same 110 animals and calling it a foundation model would be a rhetorical upgrade, not a scientific one. Phase 7 therefore invests in the measurement policy and the evaluation contract; a cross-dataset model is a later phase, conditional on external data.

## What would count as success

The primary question is whether an active policy beats random acquisition at a fixed budget on **outer-test targets that were never available to the policy**. A positive result is useful; a negative result is also publishable because it bounds the value of active probing on this atlas. The benchmark must report calibration, profile reliability, edge association and the discrepancy readout even when they fail.

## Non-claims

**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.** The policy selects hypotheses for a future experiment. It does not establish that a neuron causes a response, and it does not replace a prospective wet-lab test.
