# Supervisor brief

**Decision:** do not promote the context-conditioned latent or compositional model as a validated generative simulator. Promote the technical corrections, the negative results, and the reproducibility artifacts.

## What was done

- Created the `research/context-conditioned-generative-dynamics` branch from `4da6c24` without rewriting history.
- Corrected the VAR stability procedure to enforce and verify the actual companion-matrix radius.
- Added explicit raw-byte vs canonical-text hashing.
- Audited support at q90/q95/q99 without changing Phase 6's official q99 choice.
- Fit train-only latent LDS models at dimensions 5/10/15, causal filters, deterministic and process-covariance rollouts.
- Implemented an independent compositional-operator model with causal coefficients; did not copy the unlicensed official code.
- Audited the Kato WT_Stim oxygen schedule and evaluated no-input, measured-O2, compositional-O2, lag/time controls, and a stimulus-only baseline.
- Evaluated session-disjoint shared latent dynamics on a named-neuron panel.
- Evaluated branching from fitted process covariance and bounded named-neuron sensitivities with controls.
- Added a “CONTEXT-AWARE NEURAL SIMULATOR” dashboard view without removing existing tabs.

## Main results

1. **Autonomous rollout fails.** Selected latent LDS median H20 skill is −0.911; stable VAR median is −0.770. Only one of ten latent session-horizon cells is positive.
2. **Compositional operators do not help.** Two operators were selected on validation, but test H20 mean skill is −1.771 and all five sessions are worse than the latent model. Coefficient policy RMSE is ~0.38 and coefficient entropy is far below its maximum: collapse, not a discovered mechanism.
3. **Measured oxygen does not improve H20 prediction.** B−A paired MSE difference is +0.008 (worse), B is better in only 1/7 sessions, and the composition model is worse in 7/7. A non-specific time-since-transition control is numerically similar.
4. **Branching does not match observed local outcomes.** The stochastic latent ensemble has mean TV 0.580 versus persistence 0.548, marginal 0.317, and Markov 0.360.
5. **Shared latent dynamics fail.** Zero-shot shared H20 mean skill is −2.822; calibration adaptation is worse (−5.174). Session-disjoint, not animal-independent.
6. **Support audit reveals a real limitation.** The q99 maximum-distance audit is 0.86 for switching VAR, and context latent session 1 has only 9.3% of starts with maximum distance inside q99.
7. **Perturbations are model sensitivities only.** AVBL dynamical input has the largest mean excess over a matched control (0.075), but every target and control moves the linear model. No causal or dose claim is possible.

## Gate status

G1 PASS (manifests), G2 PASS (stability), G3 PASS (causality/leakage), G4 CONDITIONAL (support), G5 FAIL (autonomous rollout), G6 FAIL (compositionality), G7 NOT_EVALUABLE, G8 NOT_EVALUABLE.

## What is scientifically defensible

- The technical correction to VAR stability and hashing.
- The negative predictive evidence on these recordings.
- The support audit and the explicit power limitations.
- The model-level sensitivity tables, with the warning that they are not biological interventions.

## What is not defensible

- “The model simulates the worm.”
- “Oxygen drives the population dynamics” from this design.
- “The compositional operators are synaptic/circuit mechanisms.”
- “The named perturbation is an optogenetic dose.”
- “The shared model generalizes across animals.”
- “This was a systematic review.”

## Recommendation

Stop the current model family here. The next meaningful step is not another architecture search; it is a stronger experimental design (independent animals, randomized oxygen schedules, matched interventions) or a qualitatively different model class evaluated on truly held-out animals.
