# Context-conditioned generative dynamics — research protocol

**Branch:** `research/context-conditioned-generative-dynamics`
**Protocol version:** 1.0.0 (pre-registered before the final context-model evaluation)
**Parent state:** `4da6c24236d2c6399f5f1b859a21ae479e3521cc`
**Evidence tier:** exploratory. This is not a systematic review, a confirmatory trial, or a validated whole-animal simulator.

## 1. Question

Can a compositional, stimulus-aware, low-dimensional dynamical model produce a more faithful *generative simulation* of *C. elegans* population activity than a stable linear latent model or a VAR, while keeping all four forecasting tasks distinct?

The question is deliberately narrower than “build a digital worm”. It asks whether context conditioning improves held-out, causal, free-running prediction on the available recordings. A negative answer is a valid and useful result.

## 2. Four tasks that are never pooled

| Task | What may be used | What it is called |
|---|---|---|
| **A. Retrospective reconstruction** | observed data at every frame, including the scored frame | reconstruction, not forecast |
| **B. Causal latent filtering** | observations up to and including the current frame | filtered state / nowcast |
| **C. Autonomous rollout** | only the initial history and model dynamics | free-running forecast |
| **D. Schedule-conditioned rollout** | initial history plus a **known measured schedule** for the scored window | known-schedule forecast |

A reconstruction metric is never presented as evidence for C or D. Task D is not a forecast of a future stimulus: the measured oxygen series for the rollout window is supplied. No future *neural observation* is supplied to any model.

## 3. Data and provenance

- Kato et al. (2015) `WT_NoStim.mat`: five recordings, approximately 1080 s each. Used for Stages B, C, E, F, G and the context support audit.
- Kato et al. (2015) `WT_Stim.mat`: seven recordings, approximately 720 s each, with a measured 4%/21% oxygen schedule. Used only for Stage D. The file is external to Git; SHA256 `96ffc39afbc90b66ff1b4c98de199f8424391450d4a8050cbe2b65b3a2fdaa72`.
- Phase 6 frozen reports are read-only. Corrected analyses carry separate provenance in `reports/simulator/context_dynamics/`.
- No raw dataset or large training artifact is committed to Git.

**Independence warning.** The five NoStim rows are recording sessions, not five independently identified animals. Session-disjoint evaluation is not animal-independent evaluation. The seven Stim sessions share one fixed schedule; treating transitions as independent replicates would be pseudoreplication.

## 4. Splits

- NoStim and Stim use a strict chronological **70% train / 15% validation / 15% test** split with a 12-frame purge. The test block is the final chronological tail.
- The NoStim test block is new relative to Phase 6's blocked split, but it uses the same recordings. It is **not untouched in the Phase 6 sense** and is not animal-independent.
- The Stim default split has four training transitions, four validation transitions, and three held-out transitions per session (21 held-out transition frames across seven sessions). This is low power.
- Oxygen transitions are assigned to a split only when the transition frame itself lies inside that split. Train and test never share a transition.
- Calibration for Stage E uses a prefix of the held-out session's **training** block. The test block is never used for PCA, alignment, support calibration, or model selection.

## 5. Models and analysis plan

### Stage A — stability and support audit
1. Exact companion-matrix spectral cap with verification of the realized radius after scaling.
2. q90/q95/q99 support audit with raw distances and thresholds. Phase 6's official q99 choice is preserved; q95 and q90 are reported as sensitivity analyses.
3. Raw-byte and canonical-text SHA256 records for new artifacts; no silent normalization substitution.

### Stage B — latent LDS and stable VAR
Train-only PCA, latent dimensions 5/10/15, ridge and cap grid, causal Kalman-style filtering, deterministic and process-covariance rollouts. Comparators: persistence, frozen Phase 5/6 VAR, corrected stable VAR, frozen latent models. Selection uses validation H20 median skill only; the test block is untouched until selection.

### Stage C — compositional operators
Independent implementation of the published dLDS *form* (`x_{t+1} = (Σ_m c_{m,t} F_m) x_t`) with stable operators and a causal coefficient policy that sees only the current latent state and known input. No oracle future coefficients. The implementation is not a reproduction of the authors' code and does not identify synapses.

### Stage D — oxygen input
Models: (A) no input, (B) measured O2, (C) compositional O2-conditioned, (D) lagged O2 and time-since-transition controls, (E) simple stimulus-response baseline. The measured schedule is supplied for the scored window. No-input and input models are compared on the same starts. Low power and schedule non-independence are reported.

### Stage E — shared dynamics
Common **alphabetic named** panel only; numeric trace IDs are worm-specific and are never pooled as identities. Shared transition fitted on four sessions; held-out session observation map fitted on its calibration prefix. Zero-shot (`R=I`) and calibration-adapted (`R` fitted on calibration transitions) are compared with a local model fitted on exactly the same calibration data. Session-disjoint, not animal-disjoint.

### Stage F — branching
Futures are generated by sampling the fitted latent process covariance. No arbitrary neural noise and no biological bifurcation claim. The observed local kNN outcome distribution, persistence, Markov, and marginal models are comparators.

### Stage G — bounded perturbations
Only AVBL, AVBR, AVEL, AVER, RIVL, RIVR are offered, and only when verified present. Amplitudes are capped at the smaller of one-step SD and half the central 90% range. Controls are same-amplitude non-target neurons. These are mathematical model interventions, not optogenetic doses. Gate 7 is not evaluable without matched experimental interventions; Gate 8 is `NOT_EVALUABLE` without a compatible intervention dataset.

### Stage H — dashboard
A new “CONTEXT-AWARE NEURAL SIMULATOR” view preserves all existing tabs. It labels measured oxygen, hypothetical oxygen, modelled input, mathematical perturbation, and actual stimulation data separately. Missing reports show an explicit message rather than an empty or fabricated result.

## 6. Gates (version 1.0.0)

| Gate | Question | Decision rule |
|---|---|---|
| G1 data integrity | Are inputs and outputs hashed and traceable? | PASS only with raw/canonical manifests and dataset hashes |
| G2 stability | Is the realized companion radius at or below the cap? | FAIL on any unverified or above-cap model |
| G3 causality/leakage | Are PCA, support, calibration, and coefficient policies train-only and causal? | PASS only with automated no-future tests |
| G4 support | Does the support audit preserve q99 and report q90/q95 deviations? | CONDITIONAL if lower quantiles materially change coverage |
| G5 autonomous rollout | Does the latent model beat persistence at H20/H100? | PASS only if median skill > 0 across the five sessions |
| G6 compositionality | Does the operator model add held-out value over the latent model? | FAIL if not better and not distinguishable from collapse |
| G7 perturbation | Is there matched experimental validation? | NOT_EVALUABLE without such data |
| G8 intervention compatibility | Is there a compatible targeted-intervention dataset? | NOT_EVALUABLE |

Thresholds are fixed before the final test evaluation. They are not moved after seeing test results. Frozen Phase 6 gates retain their original provenance.

## 7. Reporting rules

- Report raw distances, thresholds, number of starts, number of independent sessions, and uncertainty.
- Report negative results, collapsed operators, OOD fractions, and failed gates.
- Never call this rapid or exploratory work “systematic”.
- Never convert a model perturbation into a biological dose, a synaptic weight, or a circuit mechanism.
- Never present a known-schedule forecast as an autonomous forecast.
