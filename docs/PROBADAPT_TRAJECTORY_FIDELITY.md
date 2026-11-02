# Trajectory and branching fidelity

## Scope

This stage asks a descriptive question: if the stochastic model is started from
one observed history, do its simulated paths look like plausible continuations?
It is not a test of biological branching and cannot establish a generative
mechanism.

The example is session 0, landmark 2800, horizon 20, with 100 ensemble members
and the first 20 paths stored. The observed continuation is shown only after the
simulation; it was never used to fit or calibrate that rollout.

## Example

- Deterministic conditional-mean H20 population MSE: `0.471`.
- Ensemble-mean H20 population MSE: `0.463`.
- Population-mean ensemble standard deviation at H20: `0.542`.
- Absolute population-mean error at H20: `0.115`.
- Spread/error at H20: `4.73`.

The ensemble mean is close to the deterministic forecast and to the observed
continuation on this single history. The ensemble is nevertheless far too broad:
the simulated population mean varies about five times more than its own error.

The local-mean example is even more overdispersed at the population level
(`spread/error 5.04`) and its deterministic H20 MSE is worse (`0.795`). The
population mean averages over neurons, so these numbers are not a substitute for
the full-space calibration results.

## What the figure shows

`figures/prob_example_rollouts.png` shows:

- 20 individual simulated population-mean paths;
- a 90% pointwise predictive band;
- the ensemble mean;
- the deterministic conditional mean;
- the real continuation.

The paths fan out smoothly. This fan-out is a property of the sampled
innovations and the near-unit-root operator. It is **not** evidence of
biological alternative futures. A wide fan is exactly what an overdispersed
model produces.

## Full-space fidelity

The H20 distributional results are the relevant evidence:

| quantity | P6 adaptive block | target |
|---|---:|---:|
| total implied variance | 12.12 | observed 0.861 |
| total std ratio | 3.40 | 1.0 |
| 95% coverage | 0.999 | 0.95 |
| spread/error | 14.85 | near 1 |
| OOD fraction at H20 | 0.380 | low |
| mean energy H20 | 4.605 | lower than baselines |

The model's trajectory diversity is not faithful. It is too diverse by a large
margin, and many simulated states leave the training support.

## Long horizon

Mean energy grows from `2.007` at H5 to `4.605` at H20, `7.387` at H50 and
`8.838` at H100. The growth is steeper than persistence or local mean. The
variance decomposition explains why: innovations are propagated through an
operator with spectral radius close to the cap.

The model remains numerically finite. Finite trajectories are not evidence of
fidelity.

## Branching language

No branching test was run. The following claims are not supported:

- that observed or simulated divergence corresponds to a biological
  bifurcation;
- that the model identifies alternative behavioral futures;
- that an ensemble spread is a measurement of biological variability;
- that a visually compelling fan is a validated generative model.

The earlier local-outcome diversity audit already found that observed diversity
was below geometry-preserving nulls, so "branching" is not an established
phenomenon in this dataset. That historical result is not re-used as evidence
for the stochastic model.

## Perturbation

The perturbation stage was **not launched**. Gates D–F failed, so propagating a
virtual intervention through this model would produce an indefensible
response. The exact status is in `prob_perturbation_status.json`.

If a future model passes calibration and variance fidelity, a perturbation study
must use:

- common random numbers for baseline and intervention;
- a small prespecified neuron set;
- predictive-distribution changes rather than a single trajectory;
- OOD fraction and model-disagreement checks;
- descriptive language only unless independent interventional data exist.

Every such result must carry:

**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

## Artifacts

- `prob_example_trajectories.csv`
- `prob_example_bands.csv`
- `prob_example_diagnostics.csv`
- `prob_example_metadata.json`
- `figures/prob_example_rollouts.png`
- `prob_perturbation_status.json`

## Conclusion

The conditional mean can follow one observed history reasonably well. The
predictive distribution around that mean is not credible. Trajectory realism is
therefore not established, and no branching or perturbation claim is made.
