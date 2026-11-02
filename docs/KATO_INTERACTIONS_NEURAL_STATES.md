# Neural states and transitions

## Construction

Six KMeans states were fitted on the training frames of the fixed 10-D PCA only
(seed 23, ten initializations). Test labels were assigned to those fixed
centroids. No test frame influenced the partition. The state model is a
descriptive partition of neural geometry; it is not a behavioural label and does
not use Kato macro labels as ground truth.

For each of the 175 landmarks, the observed 100-frame continuation and each
model rollout were mapped to the same training centroids. The audit compares
occupancy, row-normalized transition matrices (weighted by observed transition
counts), dwell times and recurrence.

Two nulls are included:

- a Markov chain fitted on the training transition matrix only;
- an i.i.d. marginal chain fitted on training occupancy.

## Result

Averaged over landmarks:

| Model | Transition TV to observed | Occupancy TV | Dwell L1 | Recurrence |
| --- | ---: | ---: | ---: | ---: |
| B0 persistence | 0.0503 | 0.3847 | 35.98 | 0.9436 |
| B1 independent AR | 0.1413 | 0.2548 | 17.22 | 0.9407 |
| B3 population | 0.0533 | 0.2758 | 35.17 | 0.9382 |
| B4 diagonalized | 0.2966 | 0.3552 | 40.06 | 0.8820 |
| Markov null | 0.0305 | 0.3483 | 28.49 | 0.9784 |
| Marginal null | 0.7969 | 0.7046 | 50.05 | 0.1826 |

B3 is much better than the marginal null and than the diagonalized operator, but
it does **not** beat the training-only Markov null on transition TV
(`0.0533` versus `0.0305`). Gate G4 is `FAIL` under the prespecified rule.

B3's occupancy TV (`0.276`) is better than persistence (`0.385`) and Markov
(`0.348`), but its dwell-time error is large. The model reproduces the *marginal
presence* of states more readily than the *switching law*.

## Why this is a negative result

The Markov null uses only the training transition matrix, no neural dynamics.
If a 103-neuron generative model cannot beat that baseline on held-out transition
statistics, it is not justified to describe its rollouts as a validated
state-transition simulator. The correct statement is:

> B3 reproduces some occupancy structure, but its transition statistics are
> worse than a training-only Markov chain and its state dwell times are not
> reliable.

## Boundary

- The six states are an arbitrary, fixed descriptive resolution.
- Kato's macro labels are decoded from a subset of the same neurons, so they are
  not independent behavioural validation.
- A transition matrix is not a wiring diagram.
- The five sessions are repeated recordings; no animal-level transition
  variability is estimable.
- No dorsal/ventral behavioural claim is made.
