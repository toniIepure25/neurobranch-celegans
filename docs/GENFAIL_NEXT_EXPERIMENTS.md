# Next experiments — smallest decisive tests

Ordered by information gained per unit cost. None of these is another architecture search.

## 1. Randomized oxygen timing with a sham schedule (highest value)

**Question:** Is the oxygen null biological, or an artifact of a fixed schedule confounded with elapsed time?

**Design:**

- independent animals;
- randomized transition times, not a fixed 30 s schedule;
- a sham schedule with the same number of valve/computer events but no oxygen change;
- many transitions per animal (not 12);
- pre-registered causal baseline window and response kernel;
- animal-disjoint test split;
- primary comparison: event kernel vs time-only vs sham, with no-response as the floor.

**Decision:** if randomized timing still yields no reproducible response, close the oxygen route for this preparation. If it does, estimate the response kernel and only then consider an oxygen-aware model.

## 2. Animal-disjoint re-split of existing data (cheapest)

**Question:** Does the forecast failure survive when test frames are not the previously inspected tails?

**Design:**

- leave-one-animal-out train/test where possible;
- the same persistence/AR/population-VAR baselines;
- report per-animal skill and uncertainty, not a pooled mean.

**Limit:** it cannot create new animals, and with five animals the uncertainty will remain wide. It is a robustness check, not a confirmatory test.

## 3. Direct state measurement to validate the observation model

**Question:** Is the calcium observation the limiting factor, or is the latent dynamics itself non-stationary?

**Design:**

- simultaneous voltage or a direct synaptic/state proxy in a subset of neurons;
- fit the observation map on training data;
- compare latent transitions inferred from calcium alone versus calcium plus direct state;
- test whether the half-split subspace instability persists with the direct measurement.

This is the cleanest way to separate H2 from H4/H5, but it requires new recording capability.

## 4. Time-varying/regime model with a stability-preserving rollout

**Question:** Does explicitly allowing regime change improve autonomous forecast over a time-invariant model?

**Design:**

- one transparent regime-switching or time-varying latent model, not a zoo;
- train-only regime inference; no future-dependent smoothing;
- compare one-step and autonomous multistep separately;
- include a time-only control and persistence;
- require improvement on animal-disjoint validation before looking at test.

**Stop rule:** if it does not beat persistence on animal-disjoint validation, do not scale it.

## 5. Perturbation-consistency pre-registration

**Question:** Can any virtual perturbation claim survive cross-model disagreement?

**Design:**

- pre-register the target set, amplitude and horizon;
- require agreement across at least two independently parameterized model classes before reporting a target ranking;
- report the full cross-model cosine matrix, not a single model’s ranking;
- keep the exact warning: `MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.`

No biological intervention claim should be made without a matched experimental perturbation.

## What not to do next

- Do not add a larger architecture zoo before the animal-disjoint and randomized-stimulus tests.
- Do not select a model on the already-inspected test tails.
- Do not report one-step improvement as autonomous simulator success.
- Do not interpret latent axes anatomically.
- Do not treat repeated transitions in one session as independent samples.
