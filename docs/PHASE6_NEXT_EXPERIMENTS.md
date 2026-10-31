# Phase 6 next experiments

## 1. Independent cohort before another model

The current test block is reused. The highest-value next step is not a larger model; it is a genuinely held-out recording set (new worms, same preparation) evaluated under the frozen contract in `docs/PHASE6_RESEARCH_PROTOCOL.md`.

Success criterion: pre-register the selected VAR and persistence baseline, then report H=20/100 skill and distributional metrics on animals never used for selection.

## 2. Fix the validation failure, not the test score

The validation block produced no learned model better than a constant state. Before any nonlinear model:

- diagnose whether block 4 is dominated by low-variance or nonstationary segments;
- fit a variance-normalized or segment-level baseline;
- report per-segment skill, not only a pooled median;
- test whether the validation failure is a split artifact or a real generalization failure.

A nonlinear model is justified only after a simple model beats persistence on validation in at least three worms.

## 3. Noise model for the right reason

Stochastic VAR improved autocorrelation but worsened energy and covariance. The next stochastic model should be judged on calibrated ensemble coverage and spread–error ratio, not entropy.

Predeclare:

- a stationary-preserving noise model (state-dependent or latent);
- ensemble calibration (PIT/rank histogram);
- a comparison against a Gaussian process or block bootstrap with the same mean;
- a fixed number of rollouts per state.

## 4. Switching without diagonal collapse

The diagonal restriction was necessary for stability but removes cross-neuron switching. A defensible next version is a low-rank switching model:

\[
A_k = U_kV_k^\top,\quad \text{rank } r \ll N,
\]

fitted in a PCA latent space with a validated decoder. Test rank 2, 5, 10 on validation; require stable long rollouts and >5% occupancy per state.

## 5. Branching with a stronger empirical reference

The marginal and Markov baselines beat both simulators. Before another branching claim:

- use a continuous neural outcome (e.g., PC trajectory) in addition to the four macro labels;
- report calibration of the conditional distribution;
- increase held-out REV states without temporal overlap;
- predefine whether the target is first-exit, occupancy, or full path distribution.

## 6. Perturbation only with support and controls

If a perturbation study is repeated:

- calibrate support on an independent validation cohort;
- match target and control on pre-perturbation state and amplitude;
- report the full dose–response, not the best cell;
- use a held-out animal for the effect estimate;
- do not describe any result as stimulation until a matched empirical response model exists.

## 7. Engineering follow-ups

- Add checkpoint schema versioning and a migration test for every model kind.
- Add a CI smoke test that loads one checkpoint per family and performs a finite H=20 rollout.
- Keep the Phase 6 reports separate from the Randi CRO `reports/phase6/` tree.
- Add a dashboard test that renders the simulator tab with a toy dataset.
