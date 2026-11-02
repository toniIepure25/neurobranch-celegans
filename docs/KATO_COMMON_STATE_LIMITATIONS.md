# Kato common-state controls: limitations

## Evidence status

- The five Kato `WT_NoStim` recordings are repeated sessions, not documented
  independent animals.
- The test block has been reused across earlier project phases. This audit is
  exploratory and is not fresh independent confirmation.
- All test landmarks overlap. Intervals are repeated-session/block intervals, not
  animal-level confidence intervals.
- Only Kato `WT_NoStim` is used. No WT_Stim, external dataset, or intervention
  data are loaded.

## Model limitations

- The common factors are PCA components fitted on training frames. They are
  statistical summaries of shared activity, not measured common inputs,
  identified circuits, or biological command signals.
- B5 is a diagonal-plus-rank-`k` approximation. It cannot represent arbitrary
  residual cross-neuron interactions, and its factor VAR can miss nonlinear or
  time-varying common structure.
- B2 controls only a linear local time trend. It does not model arbitrary global
  drift, behavioural state, or slowly changing preparation variables.
- B3 is a lag-one linear VAR with a 500-frame window and ridge `1.0`. A positive
  B3-over-B5 result does not prove that a direct cross-neuron interaction exists;
  it means this estimator retains a measurable H20 advantage beyond the ten-factor
  control. An unmeasured common input, nonlinearity, state variable, or estimator
  bias remains compatible with the result.
- B6 is a coefficient ablation of B3, not a refitted restricted model. It is
  diagnostic only.
- The historical B1 and B4r use a much weaker effective penalty than B3. They are
  reproduction checks, not fair controls. B1m/B4m/B5 are the matched controls.

## Evaluation limitations

- The primary metric is multivariate energy on a 10-D PCA representation. It is
  one proper scoring rule, not a complete description of forecast quality.
- Energy, MSE, covariance and autocorrelation can disagree. The audit reports
  them separately rather than selecting the most favourable metric.
- The synthetic scenarios are deliberately simple linear systems. Passing the
  synthetic check does not validate the Kato analysis; it only shows that the
  implementation can separate the intended mechanisms in a known system.
- The synthetic primary control uses three factors because the synthetic panel has
  20 neurons. The Kato primary control uses ten factors. This difference is
  documented and is not treated as equivalent.

## Identifiability and causality

Observational prediction cannot identify direct coupling, synapses, anatomical
connectivity, or a causal driver. A positive residual predictive advantage would
still be compatible with an unmeasured common input, a nonlinear interaction, a
state variable, or a model approximation. G7 is therefore fixed to `FAIL` for
structural claims, and G8 remains `NOT_EVALUABLE`.

No new perturbation screen was run in this phase. Any reference to a virtual
perturbation from the prior audit remains model-dependent:

**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

Biological intervention validity: `NOT_EVALUABLE`.

## Reporting limitations

The earlier "103 predictors" statement was correct only for session 0. The native
dimensions are 103, 127, 123, 118 and 121. New documents correct this; the frozen
historical reports are preserved unchanged and should be read with this erratum.
