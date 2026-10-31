# Phase 6 intervention and perturbation limitations

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

## What was done

The Phase 6 perturbation module edits a named neuron in a fitted simulator:

- initial-state additive edit, or
- transient additive dynamical input for 1 or 5 frames.

Amplitudes are \(q\) times the target's observed one-step change SD, capped to half the central 90% marginal range. Both signs are tested. Baseline and perturbed ensembles share the initial history and random seeds.

## What this is not

- It is not optogenetic, chemogenetic, electrical or sensory stimulation.
- It is not calibrated to any stimulation amplitude, opsin, promoter, or behaviour.
- It does not identify a causal neuron.
- A non-zero TV between outcome distributions is a model prediction, not an experimental effect.
- A support violation is not evidence that the underlying biology is impossible; it means the model is extrapolating.

## Required interpretation

1. Report the model family, checkpoint, amplitude rule and support fraction for every perturbation.
2. If fewer than 90% of perturbed states lie inside calibrated support, do not rank targets.
3. Do not compare effect sizes across neurons unless the amplitude is matched in observed-variation units.
4. Do not call a perturbation "specific" because the target effect exceeds a random-neuron control in one seed.
5. Keep Gate 6 and Gate 7 `not_evaluable` without matched empirical interventions.

## Why Kato cannot validate causality

Kato et al. (2015) contains unstimulated, immobilised wild-type worms. There is no stimulation channel, no optogenetic actuator, and no matched intervention cohort. No amount of simulation can turn these data into a stimulation-response experiment.

A causal claim requires a dataset with:

- an identified perturbation;
- a matched control;
- a known dose/timing;
- held-out animals;
- an independent replication cohort.

None is present in this checkout.
