# Simulator supervisor brief

**Branch:** `research/core-dynamics-simulator`  
**Question:** can a fitted *C. elegans* population model simulate held-out neural
trajectories, reproduce trajectory branching, and represent a named-neuron
perturbation well enough to study altered futures?

## What was built

- A free-running simulator interface over the frozen Phase 2 checkpoints:
  `x(t+1)=F_theta(x(t), history, u(t))+epsilon(t)`, with explicit initial state,
  seed, noise, support diagnostics and a typed perturbation protocol.
- Rollout ensembles, one-step/free-running metrics, constant-state and
  one-step-persistence baselines, block-bootstrap uncertainty.
- Train-only PCA and BunDLe encoders; observed-vs-simulated latent comparisons
  and seed-stability checks.
- Reversal-outcome branching comparison against persistence and an
  outcome-marginal null, plus block-permutation nulls.
- Six-neuron virtual perturbation sweep with random-neuron controls, dose,
  timing, initial-state probes and three model seeds.
- Static figures, a Streamlit tab, a schema-validated gate report and a
  reproducible manifest.

## What was found

1. **G1 pass.** Free-running rollouts are finite, start at `x(0)`, do not read
   future observations, and reproduce under a fixed seed.
2. **G2 fail, 0/5 worms.** The selected model (`mlp_residual`, best validation
   one-step R² 0.713) has the **worst** free-running test error (R² −0.510,
   MSE +0.840 versus constant baseline, OOD 0.79). VAR is the best descriptive
   free-running model (R² 0.533, ΔMSE −0.049, OOD 0.008), but it was not selected
   and is not a causal simulator.
3. **G3 fail, 0/5.** Simulated reversal outcomes are farther from observed
   conditional distributions than the marginal null. Observed block-permutation
   nulls show an entropy **deficit** (p=0.01 in three worms), not an excess.
4. **G4 pass (interface only).** A named predictor neuron can be selected,
   perturbed under `initial_state` or `dynamical_input`, and evaluated with
   matched controls. The primary targets are the pre-registered AVBL, AVBR, AVEL,
   AVER, RIVL, RIVR.
5. **G5 fail.** The top effect (AVEL inhibition, mean |ΔTV| 0.146) is below the
   random-control 95th percentile (0.500), agrees in sign in only 2/5 worms, and
   comes with OOD 0.99 and a mean shift of 7.7 training SD. No candidate is
   specific or robust.
6. **G6 not_evaluable.** Kato is observational; no stimulation-response model is
   registered; category C is refused.
7. **Latent:** PCA 3-D explains 39–53% of variance; the selected model's path is
   too short (ratio 0.19) and too straight (curvature difference −1.18 rad).
   BunDLe is unstable across seeds (CKA 0.10/0.28), so it is encoder-only.

## What this means

The software objective is met: the project can simulate futures, project them,
compare branching distributions, and run controlled virtual perturbations. The
scientific objective is **not** met: there is no demonstrated dynamical or
branching advantage, and no validated neuron-specific effect. The result is
useful as a negative result and as a measurement of what the current data and
model class cannot support.

## Decisions requested

1. **Accept the negative result** as the branch's scientific conclusion, or
   authorize a new model class selected on a prespecified *free-running*
   validation objective (not test).
2. **Provide an independent cohort or stimulation dataset** if a causal claim is
   required. Without one, Gate 6 stays `not_evaluable`.
3. **Approve the frozen gate definitions** in
   `docs/SIMULATOR_RESEARCH_PROTOCOL.md`. Changing them after seeing the test
   results would invalidate the comparison.
4. **Decide whether VAR should be promoted** to the primary descriptive
   simulator. It wins the free-running table but was not selected by the frozen
   one-step rule; promoting it now would be a protocol deviation.
5. **Clarify the Denis Kovačević brief.** The reconstructed context in
   `docs/ORIGINAL_RESEARCH_OBJECTIVE.md` is provisional; do not treat it as his
   confirmed requirements.

## Honest one-line summary

We built the simulator and the perturbation machinery; on this dataset they do
not beat persistence or marginal nulls, and no named-neuron effect is specific.
