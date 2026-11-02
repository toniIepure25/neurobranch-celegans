# Adaptive dynamics: next experiments

## 1. The decisive experiment

**Randomized oxygen timing with a sham schedule in independent animals.**

Why: the present design cannot separate an oxygen response from elapsed time,
and repeated sessions cannot provide animal-level replication. Randomizing
transition times within session, plus a yoked sham schedule, makes the time
confound testable. Pre-register the prediction landmarks, the H20 paired
endpoint, and the variance/covariance endpoint before data collection.

Minimum design:

- independent animals, not repeated sessions from the same preparation;
- at least two oxygen transitions per animal at randomized times;
- a sham schedule with the same number and duration of valve events;
- held-out animals used for no model selection;
- primary endpoint: paired H20 free-running MSE versus frozen VAR;
- co-primary endpoint: predicted/observed variance ratio and covariance
  relative error, so a shrink-to-mean solution cannot pass;
- secondary: per-neuron median skill, not mean skill dominated by near-zero
  denominators.

## 2. Animal-disjoint re-split of existing data

If new animals are not immediately available, the cheapest useful check is a
leave-one-session-out re-split. Fit and select on four sessions and evaluate on
the fifth, rotating. This does not create independent animals, but it tests
whether the primary adaptive advantage survives when the evaluation session is
not the one that contributed most to the aggregate.

Report the session-2 failure explicitly. If the method fails leave-one-session-
out, the current positive aggregate is not robust.

## 3. Variance-preserving adaptive model

The present adaptive VAR wins MSE by shrinking. A small, justified next model
is a locally fitted VAR with an explicit innovation/noise term whose covariance
is estimated from recent residuals, not a larger deterministic architecture.
Evaluate whether it closes the standard-deviation ratio gap without losing the
paired MSE advantage. Reject it if it only adds noise without improving
distributional fidelity.

## 4. Direct state measurement

A voltage or spike recording, even in a small subset of neurons, would separate
the latent drive from the calcium observation. The corrected calcium model is
stable but underfits; direct state measurements would show whether the problem
is the observation model or the latent dynamics. This is a higher-cost
experiment than the randomized oxygen design.

## 5. Matched perturbation validation

Only after a model passes both dynamical and distributional criteria: choose a
small predefined set of neurons, apply a bounded experimental perturbation, and
compare measured responses with model impulse predictions. Until then, retain
the model-dependent warning and do not rank neurons.

## Do not do next

- Do not add another latent architecture or operator library.
- Do not optimize the window or half-life on the test set.
- Do not report a single aggregate MSE without variance/covariance fidelity.
- Do not interpret model impulses as biological interventions.
- Do not call the current result a validated digital twin.
