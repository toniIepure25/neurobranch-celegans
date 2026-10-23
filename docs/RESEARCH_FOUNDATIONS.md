# Research Foundations: Formal Problem, Assumptions, and a Falsifiable Programme

**Status:** conceptual foundation for NeuroBranch.
**Compiled:** 2026-10-23.
**Companion:** [`docs/LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md) — sources, BunDLe-Net/NC-MCM analysis, evidence ledger.

This document fixes the object of study, the notation, the difference between observational and interventional quantities, the assumptions any causal claim would need, why a trajectory branch is not a bifurcation, how results will be evaluated, and seven research questions with explicit decision rules. No repository document previously defined RQ1–RQ7; the set below is adopted here so that each question is answerable with a pre-specified test.

---

## 1. Formal problem statement

**System.** Worm *i* has neural state `x_t ∈ R^{N_i}` (identified plus unidentified traces) and label `b_t ∈ B`, `|B| = 8`, with macro map `m: B → {FWD, REV, TURN, NOSTATE}`. Recordings `i = 1…5`, frames `t = 1…T_i`, `T_i ∈ [3021, 3311]`, `Δt ≈ 0.34 s`.

**Data.** `D = {(x_{1:T_i}^{(i)}, b_{1:T_i}^{(i)})}_{i=1}^{5}` from an unknown stochastic process. `N_i ∈ [109, 135]`, of which only 31–47 traces carry cell identities; the 8 label-defining neurons (`AVAL, AVAR, SMDDR, SMDDL, SMDVR, SMDVL, RIBR, RIBL`) are excluded from model input.

**Objects to estimate.**

1. **Predictive model.** `P_θ(x_{t+1} | x_{t-h+1:t})` and `P_θ(b_{t+Δ} | x_{t-h+1:t})` for horizons `Δ ∈ {1, 3, 5, 10}` (≈0.3–3.4 s).
2. **Latent abstraction.** An encoder `z_t = τ_φ(x_{t-h+1:t}) ∈ R^d`, `d ∈ [2, 16]`, with dynamics `z_{t+1} = f_θ(z_t) + ε_t`, `ε_t ~ N(0, Σ_θ(z_t))`; the deterministic case is `Σ ≡ 0`.
3. **Branching functional.** A scalar field measuring how predictive mass splits across futures (§5), with its local maxima as "branch points".
4. **Perturbation map.** `P_θ(b_{t+Δ} | do(x_t = x′))` for candidate interventions (clamp a neuron, silence a population, shift a latent coordinate).

**Estimand vs. estimate.** The target is the *data-generating* conditional distribution of behaviour given neural state, and its response to intervention. Every fitted model is an estimate conditional on a model class, a split, and five worms — the gap is the subject of §4 and §6.

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| `x_t ∈ R^N` | observed neural state at frame `t` (ΔF/F, bleaching-corrected) |
| `b_t ∈ B` | annotated behavioural state (8 classes); `m(b_t)` its macro state |
| `h` | history length in frames (BunDLe-Net uses 15) |
| `Δ` | prediction horizon in frames |
| `z_t ∈ R^d` | latent state; `d` latent dimension |
| `τ_φ, f_θ, g_ψ` | encoder, latent dynamics, behaviour decoder |
| `P, P_do` | observational and interventional distributions |
| `π: R^N → C` | abstraction map to cognitive/discrete states `C` |
| `T_Δ(x, ·)` | Δ-step predictive kernel from state `x` |
| `D_Δ(x)` | trajectory-divergence functional (§5) |
| `β(x)` | branch-point score; `B_Δ = {x : β(x) ≥ κ}` |
| `μ` | mechanistic parameter (e.g. synaptic gain) for bifurcation analysis |

---

## 3. Observational versus interventional distributions

The observational quantity is `P(b_{t+Δ} | x_t = x)`: what behaviour followed states like `x`. The interventional quantity `P(b_{t+Δ} | do(x_t = x′))` is what behaviour would follow if the state were *set* to `x′`, all other mechanisms intact.

They coincide only under strong conditions. With a latent confounder `u_t`,

```
P(b_{t+Δ} | do(x_t = x′)) = ∫ P(b_{t+Δ} | x′, u) dP(u),
```

which is not in general equal to `P(b_{t+Δ} | x′)`. The observational conditional averages over the `u` that happen to accompany `x′`; the intervention does not. That is the whole reason a decoder is not a controller.

- **Model-dependent counterfactual.** `P_θ(b | do(x = x′))` computed inside a fitted SCM is well defined *given* the model, and is the only interventional quantity available from `WT_NoStim`. Its justification rests on A1, A6 and A7.
- **Transportability.** An effect identified in worm *i* need not hold in worm *j*; cross-worm invariance requires worm-held-out tests.

NC-MCM's framing is the correct template: with observational data only, one may claim *observational* consistency of behaviour and, at most, *dynamic* causal consistency for the abstract state sequence. Perturbation vectors from state differences are predictions, not effects.

---

## 4. Assumptions required for causal claims

Each assumption is stated with its failure signature and a test that could undermine it. None is established for the Kato dataset; several are doubtful.

| ID | Assumption | Failure signature | Available test |
|---|---|---|---|
| A1 | **Causal sufficiency**: no unobserved common cause of `x_t` and `b_{t+Δ}` | Latent-variable model beats Markovian model out of sample; Markov order > 1 persists | Deep Markov / neural SDE vs. deterministic latent |
| A2 | **Markov order `h` sufficiency** | Likelihood ratio favours `h+1`; autocorrelated residuals | Nested-model likelihood ratio, purged folds |
| A3 | **Faithfulness** | Conclusions flip under small conditioning changes | Sensitivity over conditioning sets (not directly testable) |
| A4 | **Stationarity** within 1 080 s | Parameter drift across split halves; change points | Split-half refit, rolling-window prediction |
| A5 | **No instantaneous causation** at `Δt ≈ 0.34 s` | Same-frame edges vanish when coarsened | Lag-resolved independence; downsample-and-refit |
| A6 | **Positivity / overlap** for perturbed states | Counterfactuals extrapolate off the data manifold | Density ratio; Mahalanobis distance |
| A7 | **Correct functional form** for counterfactual extrapolation | Predictions depend on model class | Cross-class consistency of effects |
| A8 | **Measurement model**: ΔF/F is a noisy, slow read-out | Inferred dynamics track indicator kinetics, not neurons | Compare deconvolved traces; simulate indicator filtering |

A1 is the assumption most likely to fail and the one most often glossed: with 23–38% of traces identified and 109–135 of 302 neurons observed, unobserved neurons are the default. A5 is doubtful on physical grounds — calcium indicators integrate over roughly 0.5–1 s, so "instantaneous" coupling at 2.9 Hz is partly shared indicator kinetics and aliasing. A7 cannot be verified from observational data at all, which is why cross-model robustness (§6, E7) is the only practical substitute.

**Causal language is licensed only up to the rung the assumptions support:**

| Rung | Claim | Minimum requirement |
|---|---|---|
| L0 | Descriptive: transition counts, dwell times, occupancy | None |
| L1 | Predictive: beats baselines on held-out worms | Blocked + worm-held-out splits |
| L2 | Model-based counterfactual | A1, A6, A7 explicitly stated |
| L3 | Quasi-causal: effect invariant across worms, model classes, and an independent dataset (Atanas 2023) | L2 + invariance tests |
| L4 | Experimentally validated causal effect | L3 + agreement with a real perturbation (Leifer-style optogenetics, HisCl silencing) |

Nothing in this literature reaches L4 for whole-brain *C. elegans*. NeuroBranch should report its rung explicitly next to every perturbation number.

---

## 5. Why trajectory branching is not automatically a bifurcation

### 5.1 Definitions

**Bifurcation.** For a parameterised family `ẋ = f(x; μ)`, a bifurcation at `μ*` is a qualitative change in the phase portrait: the number or stability of invariant sets changes. Locally this is detected by an eigenvalue of the Jacobian at an equilibrium crossing the imaginary axis (saddle-node, Hopf, pitchfork) or a Floquet multiplier crossing the unit circle for a periodic orbit.

**Branch point.** A state `x` from which `T_Δ(x, ·)` is markedly multi-modal or high-variance relative to neighbours — a local maximum of

```
D_Δ(x) = E[ ||x_{t+Δ}^{(a)} − x_{t+Δ}^{(b)}||² | x_t = x ] − 2σ_obs²,
```

the expected pairwise divergence of two independent model rollouts from `x`, corrected for observation noise. `B_Δ = {x : D_Δ(x) ≥ κ}` with `κ` calibrated on a null (behaviour-only Markov or persistence model).

### 5.2 The category error

A bifurcation is a property of a *family* of systems indexed by a parameter. Along a single orbit with fixed `μ` the future is determined by the state: trajectories cannot choose. A split in an ensemble from similar states is therefore *not* by itself evidence of a bifurcation. It is evidence of one of:

1. **Intrinsic stochasticity** — `Σ(z) > 0`; divergence persists and vanishes as noise → 0.
2. **Unobserved state** — the ensemble is not conditioned on the full state; divergence disappears with a richer history or fitted latent.
3. **Deterministic expansion** — sensitive dependence without qualitative change; the Lyapunov spectrum, not a picture of forks, is the evidence.
4. **Parameter heterogeneity** — different worms, or a slowly drifting internal parameter.
5. **Observation artefacts** — indicator noise, smoothing, and 3-D projection.
6. **A genuine dynamic bifurcation** — a slow parameter `μ(t)` crosses `μ*`; the only case that earns the word, and it needs parameter dependence shown.

### 5.3 Diagnostic battery

- **B1 Horizon sensitivity.** `D_Δ` for `Δ ∈ {1, 3, 5, 10, 20}`; a bifurcation-like transition has a characteristic timescale, noise gives a smooth increase.
- **B2 Noise titration.** Refit at `σ ∈ {0, 0.25σ̂, 0.5σ̂, σ̂, 2σ̂}`; divergence surviving `σ → 0` is deterministic.
- **B3 Conditioning audit.** Add history or a fitted latent; if `D_Δ` collapses, the branch was a hidden-state artefact.
- **B4 Jacobian spectrum.** Eigenvalues of `Df` at the nearest fitted equilibrium/limit cycle; a near-zero real eigenvalue or a complex pair near the imaginary axis is bifurcation-consistent.
- **B5 Parameter sweep.** Sweep an interpretable `μ` (e.g. Morrison–Young synaptic gain), detect qualitative change in invariant sets, and test co-location with data-derived branch points.
- **B6 Cross-worm recurrence.** Bifurcations are system properties; idiosyncratic forks suggest heterogeneity or noise.
- **B7 Subsampling correction.** Wilting–Priesemann-style correction before any branching-ratio claim.

Only B4 + B5 together license "bifurcation"; B1–B3 and B6–B7 are required for any branch-point claim.

---

## 6. Evaluation principles

- **E1 Blocked temporal splits with a purge** of at least `max(h, Δ)` frames on both sides, so no window straddles train and test.
- **E2 Worm-held-out evaluation** (leave-one-worm-out) alongside within-worm temporal splits; with five worms this is the claim that matters and the one with least power.
- **E3 Baselines that can win**: persistence, VAR(1), a behaviour-only Markov model, the marginal distribution, and a time-shuffled-label control.
- **E4 Skill, not raw error**: `R²` and `1 − MSE_model/MSE_baseline` against persistence and VAR(1); negative `R²` is a result.
- **E5 Proper probabilistic scores**: log-loss, Brier, calibration, plus balanced accuracy and macro-F1 under imbalance.
- **E6 Distributional metrics**: transition TV/KL, dwell times, occupancy, rollout divergence time; one-step MSE cannot separate a model that reproduces statistics from one that smooths.
- **E7 Uncertainty over the right units**: bootstrap over windows *and* worms, multiple seeds, and the number of independent units (≤5) stated. Intervals over frames alone are too narrow.
- **E8 Negative controls**: phase-randomised traces, worm-permuted latents, label-shuffled training.
- **E9 Pre-specification**: fix `k`, `Δ`, `d`, seeds and clustering before seeing test metrics; never select the value that maximises a reported p-value (the NC-MCM restart issue).
- **E10 Rung labelling**: every table states its rung (L0–L4); virtual perturbations carry `EPISTEMIC_WARNING` in outputs and figures.

---

## 7. Falsifiable research programme (RQ1–RQ7)

Each question gives a hypothesis, test, metric and refutation condition; all use E1–E3 by default.

**RQ1 — Is population dynamics predictable beyond linear and persistence baselines?**
H: a nonlinear model (MLP/GRU) improves one-step `R²` by ≥0.10 over VAR(1) on held-out worms, and the improvement persists at `Δ = 5`.
Refuted if: no model beats VAR(1) outside the bootstrap CI, or the gain disappears under worm holdout.

**RQ2 — Does a low-dimensional latent preserve both dynamics and behaviour?**
H: some `d ∈ [2, 16]` achieves dynamic prediction error above the autoregressive baseline *and* balanced behaviour accuracy above the behaviour-Markov baseline.
Test: sweep `d`, compare with PCA at matched dimension, report the Pareto front.
Refuted if: no `d` satisfies both, or the latent adds nothing over PCA.

**RQ3 — Does neural state carry information about future behaviour beyond past behaviour?**
H: `P(b_{t+Δ} | x_t, b_{≤t})` improves held-out log-loss over `P(b_{t+Δ} | b_{≤t})` by a pre-specified margin (e.g. 0.02 nats) at `Δ = 5, 10`.
Test: conditional mutual information / log-loss decomposition with blocked CV; behaviour history is the strong null.
Refuted if: the improvement is within CI at every horizon. This is the programme's most important test: if neural state adds nothing beyond behaviour history, the neural model is decorative.

**RQ4 — Are branch points localised and recurrent across worms?**
H: states with high `D_Δ` cluster in latent space and recur across worms more than a phase-randomised null.
Test: Ripley's K or nearest-neighbour distance plus cross-worm recurrence rate; B1–B3.
Refuted if: high-`D_Δ` states are uniformly distributed, or recurrence matches the null.

**RQ5 — Are branch points bifurcations?**
H: at least one recurrent branch point coincides with an eigenvalue crossing under a parameter sweep of a mechanistic model (B4 + B5), and its divergence survives noise reduction (B2).
Test: Jacobian spectrum along fitted trajectories; sweep `μ` in the Morrison–Young model; compare locations.
Refuted if: branch points vanish under noise titration, or no sweep produces a qualitative change at the corresponding state. The correct report is then "noise-driven decision points", and the central metaphor must be retired.

**RQ6 — Are virtual perturbation effects robust and consistent with real perturbations?**
H: predicted effects (e.g. AVA/RIB/RIM manipulation on reversal probability) agree in sign across ≥3 model classes and multiple seeds, and with published optogenetic/HisCl results.
Test: rank correlation of effect sizes across model classes; sign agreement; support check (A6).
Refuted if: effects flip sign across model classes or lie outside observational support — a publishable negative result, not a failure.

**RQ7 — Can intrinsic noise be separated from an unobserved latent state?**
H: a neural SDE and a deterministic model with one extra latent dimension are distinguishable by held-out log-likelihood at this sample size, and noise titration separates their signatures.
Test: matched-parameter model comparison; B2; subsampling correction (B7).
Refuted if: they are indistinguishable — then the honest output is an explicit non-identifiability statement, a stronger contribution than another decoder.

---

## 8. What would refute the programme

1. **Predictive.** If no model beats VAR(1) under worm holdout (RQ1), the dynamical premise fails and NeuroBranch should restrict itself to descriptive transition analysis.
2. **Informational.** If neural state adds no information beyond behaviour history (RQ3), "neural branching causes behavioural branching" is unsupported regardless of model quality.
3. **Mechanistic.** If branch points dissolve under noise titration and conditioning (RQ5), they are decision points in a noisy predictive distribution, not bifurcations, and the vocabulary must change.

The programme succeeds if it delivers a calibrated branch functional, an explicit branch-versus-bifurcation verdict, perturbation effects with a rung label and cross-class robustness, and a documented non-identifiability result. All four are achievable with existing data and code; none requires claiming a biological control that was never performed.
