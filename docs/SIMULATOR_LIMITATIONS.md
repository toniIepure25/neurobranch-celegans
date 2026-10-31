# Simulator limitations

> Read this before quoting any simulator number. The simulator is a
> phenomenological model fitted to observational calcium imaging. It is not a
> digital twin, not a whole-brain model, and not a causal intervention platform.

## 1. Data limitations

- **Five worms, one laboratory, one preparation.** The unit of generalisation is
  the worm, and n=5 is too small for a population claim. Per-worm heterogeneity
  is large.
- **Partial coverage.** 103–127 predictor traces per worm, 31–47 named, out of
  302 neurons. "Whole-brain" is false for this pipeline.
- **Immobilised animals.** Proprioception, muscle activity and free locomotion
  are altered or absent.
- **No interventions.** Kato 2015 is observational. No optogenetic, thermal,
  mechanical or chemogenetic stimulation was delivered.
- **No controlled stimulus.** `WT_NoStim.mat` is spontaneous activity; input
  identification is limited.
- **Calcium, not spikes.** ΔF/F is slow, nonlinear and noisy; no deconvolution.
  The 2.9 Hz sampling aliases fast dynamics.
- **Behaviour labels are inferred.** The four macro states are decoded from the
  same neural population; eight behaviour-reporting neurons are excluded, but
  circularity is reduced, not eliminated. There is no independent video or
  manual scoring in this checkout.

## 2. Model limitations

- **Phenomenological map.** No membrane potential, synaptic conductance, gap
  junction, muscle or body mechanics. A state edit is not a biological
  intervention.
- **One-step selection is not dynamical validation.** The selected model is
  chosen on validation one-step R². It has the worst free-running test error of
  the seven families and is outside support for ~79% of test rollout states.
- **Free-running error compounds.** Even the best descriptive model (VAR) only
  reduces MSE relative to a constant-state baseline by about 0.049 and does not
  establish a mechanistic advantage.
- **Noise is not learned branching.** A nonzero `noise_std` creates ensemble
  diversity, but that diversity is injected, not inferred from the data.
- **Support model is descriptive.** The 99th-percentile training-distance
  envelope is not a calibrated OOD probability. High OOD means extrapolation,
  not necessarily failure; low OOD does not prove validity.
- **Model selection is per worm.** There is no pooled model and no independent
  cohort; cross-worm generalisation is not tested.
- **Only one architecture is selected.** Alternative models are reported
  descriptively but not promoted post hoc.

## 3. Branching limitations

- **Conditional outcome distributions are not a vector field.** No parameter is
  varied, so no bifurcation is identified.
- **The observed null shows a deficit.** For three worms, local entropy is
  significantly *below* a geometry-unrelated block null. That is not evidence
  for branching.
- **Few independent events.** Four to seven reversal runs per worm; overlapping
  candidate frames are handled with block bootstrap, but effective sample size
  is small.
- **Outcome definition is coarse.** First macro state different from `REV`
  within 20 frames collapses distinct motor programs and treats `NOSTATE` as a
  category.
- **Marginal null is strong.** Reversal outcomes are imbalanced; a model that
  predicts the majority outcome can look stable without conditional skill.
- **Simulator OOD is near 1.0** for reversal rollouts, so the comparison is
  dominated by extrapolation.

## 4. Perturbation limitations

- **Category C is absent.** No empirical stimulation-response model exists;
  `empirical_response` is refused. Gate 6 is `not_evaluable`.
- **Amplitudes are model units.** One SD is not a light power, drug dose or
  current. There is no device calibration.
- **State edits, not stimulation.** `dynamical_input` edits the model state after
  a prediction. It does not model opsin kinetics, calcium buffering, network
  recruitment or behaviour feedback.
- **Specificity fails.** No candidate exceeds the random-neuron 95th percentile
  of `|ΔTV|`; the best sign agreement is 3/5 worms.
- **Model-seed instability.** The top effect changes magnitude and sign across
  seeds 17/23/31.
- **Dose is non-monotonic** in the expected direction for most candidates.
- **OOD ~0.99.** Perturbed trajectories leave the training support almost
  immediately; a large apparent effect can be an extrapolation artefact.
- **Initial-state probes are not interventions.** They test mathematical
  sensitivity; they do not represent a reversible biological manipulation.

## 5. Latent limitations

- **PCA is linear and lossy.** Three components explain 39–53% of training
  variance. A 3-D plot is not the population state.
- **BunDLe has no published inverse.** A named-neuron change is not one latent
  coordinate.
- **BunDLe is unstable across seeds.** CKA 0.102/0.284 and Procrustes residuals
  1.02/2.03 on held-out frames; it is used for visualisation only.
- **Official BunDLe artifacts are in-sample.** They were fitted on all frames
  and are not held-out validation.

## 6. Evaluation limitations

- **Small test blocks.** Twenty starts per worm for the main rollout metrics and
  eight reversal states for perturbations; uncertainty is block-bootstrap, but
  the design is underpowered for small effects.
- **No external cohort.** No independent lab, preparation or species.
- **No calibrated interval for the simulator.** Gate 2 fails; no interval is
  presented as a calibrated dynamical bound.
- **No causal claim.** A model-dependent effect is not an experimental effect,
  even when its TV distance is large.
- **No BCI or NeuroAI claim.** This is not a brain-computer interface, not a
  foundation model, and not a general NeuroAI benchmark.

## 7. Interpretation rules

1. Do not call a projected split a bifurcation.
2. Do not call a noise-induced ensemble "learned branching".
3. Do not call a state edit "stimulation" or "optogenetics".
4. Do not call a model-dependent ranking a biological target.
5. Do not call a one-step R² a dynamical validation.
6. Do not call this a whole-brain or digital-twin model.
7. Do not quote a perturbed effect without the OOD fraction and the control
   comparison.
8. Do not promote VAR to a causal simulator because it wins a descriptive table.
