# Limitations

## Data and independence

1. The five WT_NoStim recordings are sessions, not five independently identified animals. Session-disjoint is not animal-disjoint.
2. The seven WT_Stim recordings use one repeated oxygen schedule. The 21 held-out transition frames are not 21 independent biological events.
3. The NoStim chronological test tail is a new split relative to Phase 6, but the recordings were used in earlier phases. It is not untouched in the Phase 6 sense.
4. There is no common neuron panel across worms/sessions beyond the subset of explicitly named neurons. Numeric trace IDs are worm-specific and were not pooled.
5. Kato 2015 recordings are population calcium imaging with behavioural state labels; they are not simultaneous whole-brain electrophysiology.

## Model and inference

6. The latent LDS is linear-Gaussian with a fixed PCA observation map. It cannot represent nonlinear or non-stationary population dynamics.
7. The stable VAR is a linear model; exact stability does not imply predictive validity.
8. The compositional operator model is an independent implementation of a published mathematical form, not a reproduction of the authors' software. Its coefficients and operators are not identifiable and are not synaptic weights.
9. The coefficient policy collapsed toward low-entropy coefficients and did not improve held-out prediction. This is an estimator failure, not evidence about biological compositionality.
10. Shared latent dynamics pool non-identifiable PCA bases. The shared coordinate is a model coordinate, not a biological common coordinate.
11. Support distance is nearest-neighbour distance in standardized neural space. It is not a likelihood, confidence interval, or mechanistic boundary.
12. A high per-step support fraction can coexist with poor forecast skill and with a low fraction of starts whose maximum distance is inside the threshold (session 1).

## Oxygen

13. Oxygen is an environmental sensory input, not targeted neuronal stimulation.
14. The measured oxygen schedule is supplied for the scored window. These are known-schedule forecasts, not forecasts of a future stimulus and not autonomous rollouts.
15. Seven sessions and three held-out transitions per session give low power. The time-since-transition control is numerically similar to measured O2, so a specific oxygen effect is not established.
16. The oxygen-conditioned model does not establish a causal pathway from oxygen to any neuron.

## Perturbations and branching

17. Named-neuron perturbations are mathematical edits to a model. They are not optogenetic doses, experimental interventions, or causal circuit evidence.
18. Branching draws from a fitted process covariance. It is model uncertainty, not measured biological variability, and it does not establish a bifurcation.
19. Gate 7 is NOT_EVALUABLE without matched experimental interventions; Gate 8 is NOT_EVALUABLE without a compatible intervention dataset.

## Evidence status

20. This is an exploratory research branch, not a systematic review or a confirmatory study.
21. Frozen Phase 6 results remain frozen. New support and stability analyses have separate provenance.
22. The external dLDS Python repository had no LICENSE file; no code was copied. The independent implementation is not a reproduction of the published numerical results.
