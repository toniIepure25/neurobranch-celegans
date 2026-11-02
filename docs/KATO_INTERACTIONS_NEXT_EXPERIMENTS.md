# Next experiments

The current audit should stop at the observational boundary. The next step is
not another architecture or hyperparameter search on the reused split.

## 1. Independent-animal replication

Acquire or identify recordings with documented animal IDs, not repeated sessions
from the same preparation. Pre-register the split, the H20 primary endpoint, the
baseline family, and the cross-neuron contrast before looking at the new scores.
The minimum credible unit is the animal, not the landmark.

**Falsification:** the B3-minus-B1 energy interval includes zero at the animal
level, or the sign reverses in a majority of animals.

## 2. Matched neuron-specific perturbation

Required design elements:

- identified target neuron and a verified expression/illumination map;
- sham/control illumination and a vehicle/control group;
- at least two doses or amplitudes;
- pre-specified time window relative to the measured neural state;
- behavioural and neural outcomes;
- held-out animals and held-out targets;
- a pre-registered direction and effect size.

Without these, a virtual perturbation cannot be promoted to a causal hypothesis.
A future analysis should predict the **difference between stimulated and sham**
before seeing the outcome, not fit the outcome after the fact.

## 3. Measure or manipulate the common input

The synthetic audit shows that a shared latent drive can create spurious
off-diagonal coefficients and reverse impulse responses. The next Kato-style
analysis should include an independent proxy for the common drive (for example,
a measured stimulus, behavioural state, or external context variable) and test
whether the cross-neuron coefficients survive conditioning on it.

**Falsification:** conditioning on the common input removes the cross-neuron
energy advantage.

## 4. Pre-registered operator-stability test

Define a minimum cross-segment coefficient correlation and a minimum
cross-session mode agreement before fitting. A positive predictive result should
not be interpreted as an interaction if the operator fails that stability test.
The current audit already treats the `0.052` off-diagonal correlation as a
failure; a future study should make that a gate, not a post hoc observation.

## 5. State model with an independent behavioural readout

Replace Kato macro labels as the only readout with an independent behavioural
measurement and a pre-specified state resolution. Test transition statistics
against a training-only Markov null and against a hidden semi-Markov model. Do
not select the number of states by test likelihood without a held-out animal.

## 6. Perturbation-support stress test

Before any experimental claim, test whether the proposed amplitude and target
combination lies inside the training support under a stronger density model than
the 1st-99th percentile envelope. Report the fraction of out-of-support entries
and the sensitivity of the predicted sign to the support threshold.

## 7. Decision rule

Proceed to an intervention study only if:

1. the cross-neuron predictive advantage replicates at the animal level;
2. the operator stability gate passes on held-out animals;
3. the perturbation direction is stable across permitted fits;
4. the target/amplitude is inside a defensible support region;
5. a pre-registered sham-controlled outcome is available.

If any condition fails, preserve the negative result and do not describe the
model as a causal simulator.

**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**
