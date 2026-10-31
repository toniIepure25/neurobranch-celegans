# Phase 6 simulator results

**Branch:** `research/phase6-dynamical-validation`
**Start:** `8d945f9ad755aabb4f0a30e93c2f24a8928a0614`
**Evidence tier:** exploratory replication. Test block 2 was inspected in Phase 5; no untouched Kato block remains.

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

## 1. What was built

- Corrected free-running evaluation contract with real history windows, no teacher forcing, validation-only selection, and explicit test-reuse disclosure.
- Regularized VAR(1–3), stochastic VAR (Gaussian, i.i.d. and block bootstrap), compact switching VAR(K=2/3), all with frozen persistence and Phase 5 baselines.
- Support-constrained named-neuron perturbations with observed one-step amplitude units.
- Machine-readable validation/test/branching/perturbation/gate reports, figures, model cards and a simulator manifest.
- Reworked Core Dynamics Simulator tab that preserves the existing app tabs and loads Phase 6 checkpoints by family.

## 2. Validation result

No learned family beat persistence on validation by median skill (every configuration had a negative median; a few individual worms were positive).

| Family | Selected | Validation median skill | Positive worms |
|---|---|---:|---:|
| VAR | p2, ridge 10, cap 0.995 | −0.2209 | 0/5 |
| Stochastic VAR | Gaussian | −0.7820 | 0/5 |
| Switching VAR | K=3 | −3.3084 | 0/5 |

The nonlinear trigger was not met (`0/5` positive), so B4 was not fitted.

## 3. Exploratory test result

| Family | H20 skill | H100 skill | H20 energy | H20 autocorr | H20 covariance | H20 readout support |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 0.000 | 0.000 | **0.203** | 0.289 | **0.395** | 0.968 |
| Phase 5 MLP | −1.356 | −17.999 | 1.957 | 0.276 | 1.655 | 0.718 |
| Phase 5 VAR | 0.101 | 0.254 | 2.078 | 0.275 | 0.635 | 0.998 |
| New VAR | 0.093 | **0.255** | 2.618 | 0.248 | 0.686 | 0.998 |
| Stochastic VAR | −0.410 | −0.173 | 1.277 | **0.169** | 0.656 | 0.998 |
| Switching VAR | −1.748 | −0.721 | 3.393 | 0.311 | 1.057 | 0.991 |

Key facts:

- The new VAR beats persistence in 4/5 worms at H20 (worm-cluster bootstrap CI 0.003–0.187, descriptive) but **does not beat the frozen Phase 5 VAR** (3/5; mean +0.093 vs +0.101).
- The Phase 5 MLP, previously selected on one-step validation \(R^2\), diverges catastrophically in free running.
- Stochastic VAR improves autocorrelation but is worse than persistence on energy and covariance.
- The constant-state baseline is the best distributional model in the comparison.
- All H=200 rollouts were finite; finite is not accurate.

## 4. Gates

| Gate | Status | Evidence |
|---|---|---|
| G1 functionality | **pass** | 5/5 worms finite at H20/H100 |
| G2 autonomous skill | **fail** | 4/5 vs persistence but only 3/5 vs frozen Phase 5 VAR |
| G3 distributional fidelity | **fail** | 0/5 worms better on energy, autocorrelation and covariance simultaneously |
| G4 branching | **fail** | TV 0.513 vs Markov 0.264 and marginal 0.245 |
| G5 support | **pass** | 100% of perturbed states in support across 36/36 conditions |
| G6 robust sensitivity | **not_evaluable** | only one model family; protocol requires ≥2/3 |
| G7 biological validation | **not_evaluable** | no matched intervention data |

## 5. The 13 scientific questions

1. **Can the simulator run autonomously from real histories?** Yes. All selected VAR/stochastic/switching checkpoints produce finite H=100 and H=200 rollouts from real 3-frame histories (G1 pass).
2. **Does it outperform persistence on validation?** No. Every validation configuration had negative median skill (G2 fail).
3. **Does the validation-selected VAR improve on the frozen Phase 5 VAR?** No. Test H20 0.093 vs 0.101; it wins in only 3/5 worms.
4. **Did validation selection prevent the Phase 5 MLP failure?** Yes in the sense that the MLP was not reselected, but the MLP checkpoint still diverges catastrophically, showing that one-step \(R^2\) is an invalid surrogate for free-running skill.
5. **Are multi-horizon dynamical statistics reproduced?** Only partially. VAR/stochastic improve autocorrelation, but energy distance and covariance are worse than persistence.
6. **Is the joint distribution reproduced?** No. Persistence has the lowest energy distance at H20 and H100.
7. **Is stochastic branching realistic?** No. Stochastic VAR TV is 0.513 vs 0.264 for a first-order Markov chain and 0.245 for the marginal (G4 fail).
8. **Does the switching model identify regimes?** It fits and samples, but K=2/3 both fail validation skill; the diagonal restriction removes cross-neuron switching, so no regime mechanism claim is supported.
9. **Are long rollouts stable?** Numerically finite, but several VAR trajectories collapse toward small norms and stochastic trajectories drift; stability is not accuracy.
10. **Are virtual perturbations inside calibrated support?** Yes for the tested q=1 edits (100% in support), but the support envelope is permissive and this is not biological validation (G5 pass). At 90/95/99% calibration, 92.9/94.0/97.4% of observed test states are in support, while selected rollouts are almost entirely in support; the diagnostic is weak against shrunken trajectories.
11. **Are named-neuron effects robust and specific?** Not established. AVEL shows an exploratory positive excess in 4/5 worms, but only one family and a weak random-neuron control; G6 is not_evaluable.
12. **Can any causal or stimulation claim be made?** No. Kato has no stimulation; G7 is not_evaluable.
13. **What is the decisive next experiment?** An independent cohort with a pre-registered VAR/persistence comparison, followed by a matched intervention dataset before any perturbation claim (see `docs/PHASE6_NEXT_EXPERIMENTS.md`).

## 6. Negative outcomes worth keeping

- More flexible is not better: the residual MLP is the worst free-running model.
- Added stochasticity is not automatically realistic: it lowers trajectory MSE skill and worsens energy distance.
- A switching model can be stable and still fail; diagonalization traded mechanism for stability.
- Support calibration can pass while target specificity remains unidentifiable.
- Selecting on validation can still select a model that loses to the previous baseline on test.

## 7. Artifacts and reproduction

Reports are collision-safe under `reports/simulator/phase6/`; figures under `figures/simulator/phase6/`; model cards under `reports/simulator/phase6/model_cards/`; provenance in `reports/simulator/phase6/simulator_manifest.json`.

```powershell
python scripts/phase6_audit.py
python scripts/phase6_select.py --families var stochastic switching
python scripts/phase6_evaluate.py
python scripts/phase6_branching.py --rollouts 80
python scripts/phase6_perturbation.py
python scripts/phase6_support_sensitivity.py
python scripts/phase6_gates.py
python scripts/phase6_cards.py
python scripts/phase6_figures.py
python -m pytest tests/simulator -q
python -m ruff check src scripts app tests
```

The Randi Phase 6 CRO under `reports/phase6/` and `docs/PHASE6_RESULTS.md` is untouched.
