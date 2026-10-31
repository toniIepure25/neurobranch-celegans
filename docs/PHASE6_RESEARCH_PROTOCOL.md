# Phase 6 simulator protocol: autonomous dynamics, stochastic futures and bounded perturbations

**Status:** frozen before Phase 6 model fitting and before the Phase 6 test evaluation.
**Date:** 2026-10-31.
**Branch:** `research/phase6-dynamical-validation`.
**Starting HEAD:** `8d945f9ad755aabb4f0a30e93c2f24a8928a0614`.
**Collision note:** the repository's earlier Randi Causal Response Operator also used the Phase 6 label. It is preserved unchanged. Simulator artifacts live under `reports/simulator/phase6/` and `figures/simulator/phase6/`; two colliding document names use the `SIMULATOR_PHASE6_` prefix.

This protocol fixes the estimand, splits, selection rule, metrics, seeds, gates and claim labels. Any change is recorded in `docs/PHASE6_SIMULATOR_DEVIATIONS.md` with whether it happened before or after test inspection.

---

## 1. Question

> Can a fitted generative model produce autonomous *C. elegans* neural trajectories from real observed histories, reproduce multi-horizon dynamical and distributional statistics, and support bounded named-neuron perturbation analysis without extreme extrapolation?

The primary object is neural-state simulation. Behaviour decoding is a supporting readout, not the target.

The modelled quantity is standardised calcium-imaging \(\Delta F/F\), not membrane potential, spiking, muscle activity, or an experimentally calibrated stimulation amplitude.

---

## 2. Data and splits

- Dataset: Kato et al. (2015), `WT_NoStim.mat`, five wild-type, unstimulated, immobilised worms.
- Predictors: 103–127 bleaching-corrected traces; eight behaviour-label neurons excluded.
- Frame interval: 0.326–0.357 s; no resampling.
- Split: blocked interpolation from `configs/experiments/blocked_suite.yaml`.
  - training blocks: 0, 1, 3;
  - validation block: 4;
  - test block: 2;
  - purge: 12 frames.
- Every model receives a real history window wholly inside the evaluation split.
- Scalers, PCA, probes, VAR coefficients, residual covariances and support sets are fitted on training frames only.
- Support thresholds are calibrated on validation frames.
- Test frames never enter fitting, early stopping, hyperparameter selection, support calibration or figure selection.

### Test reuse disclosure

Phase 5 already evaluated block 2. No untouched Kato block remains. Therefore Phase 6 test results on block 2 are **exploratory replication**, not a confirmatory untouched test. Every gate records `evidence_tier: exploratory`.

A confirmatory claim would require an independent cohort or a genuinely unused preregistered block. Neither exists in this checkout.

---

## 3. Audit baseline and confirmed defects

The Phase 5 saved checkpoints were re-evaluated before new model fitting.

| Kind | Stored Phase 5 MSE | Corrected H=20 MSE | Difference |
|---|---:|---:|---:|
| `var` | 0.49913 | 0.49913 | 0.00000 |
| `mlp_residual` | 1.38802 | 1.38802 | 0.00000 |
| `gru` | 0.64075 | 0.66198 | +0.02123 |

The h1 models reproduce their stored MSE exactly. GRU changes because Phase 5 discarded its real two-frame history and padded a single state.

Confirmed defects and their impact:

1. **History construction:** `test_starts()` hardcoded history=1, and rollout accepted only `x0`. Models with history>1 repeated `x0` instead of using the observed past. Corrected by `initial_history` and history-aware start lists.
2. **Time indexing:** evaluation now uses `x[t-history+1:t+1]` and scores `t+1 ... t+H`; no future frame passes into the model.
3. **Autocorrelation:** the old formula correlated concatenated simulated and real blocks. It was not a per-neuron lag correlation. Mean corrected errors are 0.457 (VAR), 0.533 (selected MLP) and 0.539 (GRU), versus 0.266, 0.225 and 0.331 under the old formula. The old metric understated temporal mismatch.
4. **Gate 2:** the old Gate 2 used a one-step R² difference. It now uses free-running skill against a constant-state baseline.
5. **Support:** the old threshold was the 99th percentile of leave-one-out training distance. It now defaults to the 99th percentile of held-out validation distances, with 90%/95%/99% sensitivity reported. The old threshold flagged 79% of selected-model rollout states as OOD; the calibrated diagnostic flags 22.6%. Sensitivity on the selected rollouts is 100% in support at all three quantiles for VAR/stochastic VAR, 91.7/93.7/99.1% for switching; observed test states are 92.9/94.0/97.4% in support. The metric is permissive for shrunken rollouts, so in-support is not accuracy.
6. **Perturbations:** previous ±1–2 SD inputs accumulated after each prediction. They are re-scaled to observed one-step variation and gated on support.
7. **Initial behaviour:** ensembles now record the observed initial macrostate instead of inferring it from the model probe.

The corrected free-running skill at H=20 (mean over 3 seeds and 5 worms) is:

| Kind | One-step ΔR² vs persistence | Free-running skill vs constant state |
|---|---:|---:|
| `var` | +0.0075 | **+0.1370** |
| `gru` | −0.0933 | −0.1403 |
| `mlp_residual` | +0.0079 | −1.6273 |

The Phase 5 selected residual MLP is dramatically worse than a constant state over a 20-step autonomous rollout. VAR is the only Phase 5 family with positive mean free-running skill.

---

## 4. Candidate models

1. Persistence (constant initial state).
2. Frozen Phase 5 selected `mlp_residual`, seed 17.
3. Frozen Phase 5 `var` checkpoint.
4. B1 regularized VAR: lag `1–3`, ridge `1e-3 … 10`, optional spectral cap.
5. B2 stochastic VAR: Gaussian innovations with Ledoit–Wolf covariance; iid residual bootstrap; moving-block residual bootstrap.
6. B3 compact switching VAR: K=2 primary, K=3 sensitivity; diagonal Gaussian innovations; sticky transitions; five restarts.
7. B4 compact nonlinear: only if B1–B3 have positive validation skill in at least three worms.

No Transformer and no large model.

---

## 5. Validation-only selection

Primary validation metric at H=20:

\[
S_{m,w}=1-\frac{\mathrm{MSE}_{m,w,20}}{\mathrm{MSE}_{\mathrm{persistence},w,20}}
\]

- Pool eligible validation starts per worm.
- Select one global hyperparameter configuration per family by median worm skill.
- Hard requirements: finite fraction ≥0.95, no NaN/Inf, a valid history-aware rollout.
- Tie-break: lower validation energy distance, then simpler model.
- No weighted composite optimized on test.
- Freeze the selected configuration as `selection/frozen_config.json` with its SHA256 before test evaluation.

B3: K=2 is primary. K=3 is retained only if it improves median validation skill by ≥2% and every state has ≥5% occupancy.

B4 trigger: at least one B1–B3 configuration has positive validation skill in ≥3/5 worms. If not triggered, B4 is reported as not pursued.

Seeds: 17, 23, 31 for stochastic fitting and probes; deterministic VAR uses 17 for the shared probe and a training-transition bootstrap for coefficient uncertainty.

---

## 6. Evaluation contract

Every model uses the same frozen start manifest per worm:

- H=20: up to 50 eligible test starts;
- H=100: 20 evenly spaced eligible test starts;
- H=200: stability subset.

For each start, the model receives only the observed history window. No teacher forcing is used.

Primary measurements:

- one-step prediction (descriptive only);
- autonomous MSE, NMSE skill, R² at H=1, 5, 10, 20, 100;
- per-worm and per-horizon rows;
- finite fraction and numerical stability.

Secondary measurements:

- per-neuron autocorrelation at lags 1, 2, 5, 10;
- cross-neuron covariance error;
- Welch log-power spectrum error;
- PCA state-occupancy Jensen–Shannon divergence;
- ensemble energy distance and variogram-style score;
- ensemble spread–error ratio and rank histogram;
- calibrated support/OOD fractions;
- long-run norm growth, spectral radius and fixed-point proximity;
- behavioural transition TV/JS and categorical log score.

Realized-trajectory accuracy and predictive-distribution fidelity are reported separately. Stochastic models are not judged solely by sample MSE.

---

## 7. Branching

- Use held-out non-overlapping REV states with a complete H=20 future.
- Empirical conditional distribution: kNN over held-out REV states, excluding neighbours within `H+2` frames.
- Simulated distribution: 100–200 native stochastic rollouts per state.
- Baselines: persistence, outcome marginal, first-order Markov chain fitted on training transitions, and the empirical kNN reference.
- Metrics: TV, JS, categorical log score, entropy difference, occupancy, paired block-bootstrap intervals.

Success is agreement with observed conditional transition statistics, not entropy excess over a shuffled null. A visually diverging trajectory is not called a bifurcation.

Behaviour labels are inferred from AVA/SMD/RIB calcium activity. If the probe is mostly outside training support, behavioural branching is provisional and the primary claim is restricted to neural-state statistics.

---

## 8. Support and perturbations

### Support

- kNN support fitted on training states.
- Threshold calibrated at the 95% quantile of validation distances; 90% and 99% reported as sensitivity.
- Add PCA Mahalanobis and per-neuron z-extreme diagnostics.
- Report the old training-only 99% threshold for continuity.

### Bounded perturbations

- Targets: `AVBL`, `AVBR`, `AVEL`, `AVER`, `RIVL`, `RIVR`.
- Verify target presence in every worm.
- Amplitudes based on the observed one-step change of the target neuron:
  `δ = q × sd(Δx_j)`, `q ∈ {0.5, 1, 2}`, capped to the central 90% marginal range.
- Protocols: initial-state perturbation; transient dynamical input of 1, 3 or 5 frames.
- Matched baseline and perturbed rollouts share the initial history and random seed.
- Report support fraction, divergence, latent displacement, decay, timing and seed consistency.
- Primary intervention comparison requires ≥90% perturbed states inside calibrated support in ≥4/5 worms.
- If that requirement fails, stop ranking neurons and report the failure.

Controls are matched on innovation SD and autocorrelation, not arbitrary random neurons. No causal specificity is inferred from simulation.

---

## 9. Gates

| Gate | Criterion |
|---|---|
| G1 | 5/5 worms produce finite autonomous H=20 and H=100 trajectories from real histories |
| G2 | selected model beats persistence and frozen Phase 5 selected model; ≥4/5 worms positive, ≥3/5 block-bootstrap lower CI > 0; exploratory |
| G3 | lower energy distance, autocorrelation error and covariance error than persistence in ≥4/5 worms; exploratory |
| G4 | lower mean TV than Markov and marginal baselines in ≥4/5 worms; exploratory |
| G5 | at least one protocol/amplitude has ≥90% in-support states in ≥4/5 worms |
| G6 | one named target/direction has consistent sign in ≥4/5 worms and ≥2/3 model families, conditional on G5 |
| G7 | matched experimental intervention evidence; otherwise `not_evaluable` |

Allowed statuses are `pass`, `fail`, `not_evaluable`. No failed gate is relabelled.

---

## 10. Claim boundaries

- No whole-brain, digital-twin, BCI or causal-intervention claim.
- No optogenetic interpretation of a standardised calcium-trace edit.
- No formal bifurcation without parameter-dependent vector-field evidence.
- No cross-dataset transfer from Kato to Randi without a validated alignment.
- Every intervention output keeps the warning:

> MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.

---

## 11. Reproduction

All Phase 6 scripts write to `reports/simulator/phase6/` and are described in `docs/REPRODUCING.md`. The manifest records the Git revision, data hash, config hash, protocol hash, environment, seeds, commands and report hashes.
