# Branching simulation

> **Terminology.** A visible split in a projected trajectory is not a
> bifurcation. This project estimates **conditional distributions of the next
> behavioural state reached after leaving the current state**. A bifurcation
> would require a parameter-dependent change in the vector field, which is not
> identified here.

## 1. Question

Can the simulator reproduce the observed diversity of futures from held-out
reversal states? The evaluation compares:

- **observed**: kNN outcome distribution among held-out reversal states;
- **simulated**: outcome distribution of free-running rollouts from the same
  initial state;
- **persistence**: rollouts of the constant-state null;
- **marginal**: the outcome distribution over all held-out reversal states,
  ignoring the initial condition.

A simulator that merely reproduces the marginal distribution is not reproducing
conditional branching.

## 2. Outcome definition

For an initial frame `t` with macro label `REV`, the outcome is the **first macro
state different from `REV`** within 20 frames. If no different state occurs, the
outcome is `REV`. The four macro states are `FWD`, `REV`, `TURN`, `NOSTATE`.

Observed local distributions use `k=40` behaviourally matched held-out reversal
states and exclude temporal neighbours within ±12 frames. Candidate frames
overlap in time, so a moving-block bootstrap is used for uncertainty; the
permutation null also permutes contiguous blocks of outcomes rather than
individual frames.

Evaluated states are non-overlapping held-out reversal states sampled every 20
frames through each reversal run. Onsets alone are often degenerate (the worm is
committed to turning at the first reversal frame), so the evaluation includes
later states where futures actually diverge.

## 3. Gate 3

Frozen criterion: in at least 3/5 worms,

```text
mean TV(simulated, observed) < mean TV(persistence, observed) - 0.02
and
mean TV(simulated, observed) < mean TV(marginal, observed) - 0.02
and
mean simulated entropy inside the observed 95% event range.
```

Result: **fail, 0/5 worms.** No worm satisfies both TV comparisons.

| Worm | States | Observed entropy | Simulated entropy | TV sim | TV persistence | TV marginal | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 21 | 0.482 | 0.107 | 0.713 | 0.555 | 0.328 | no |
| 1 | 15 | 0.584 | 0.000 | 0.657 | 0.660 | 0.285 | no |
| 2 | 23 | 0.680 | 0.084 | 0.551 | 0.443 | 0.170 | no |
| 3 | 13 | 0.837 | 0.011 | 0.541 | 0.546 | 0.194 | no |
| 4 | 21 | 0.609 | 0.004 | 0.662 | 0.586 | 0.251 | no |

The marginal null is closer to the observed conditional distributions than the
simulator in every worm. The simulator collapses most outcomes to a single state
(entropy near zero) while observed reversals retain 0.48–0.84 bits of local
outcome diversity.

## 4. Observed null: a deficit, not an excess

The block-permutation null tests whether observed local entropy exceeds what is
expected when outcomes are unrelated to the local geometry. On all held-out
reversal frames:

| Worm | Candidate frames | Observed mean entropy | Null mean | Two-sided p | Excess p |
|---|---:|---:|---:|---:|---:|
| 0 | 328 | 0.577 | 0.884 | 0.010 | 1.000 |
| 1 | 248 | 0.625 | 0.903 | 0.010 | 1.000 |
| 2 | 366 | 0.747 | 0.792 | 0.209 | 0.900 |
| 3 | 225 | 0.888 | 0.823 | 0.119 | 0.060 |
| 4 | 409 | 0.606 | 0.867 | 0.010 | 1.000 |

For worms 0, 1 and 4 the observed entropy is **significantly lower** than the
geometry-unrelated null (a deficit). A deficit means nearby states have *more*
predictable outcomes than chance, not that the system branches more than chance.
It is not evidence for branching. No worm shows a significant excess.

This reproduces the existing audit's conclusion: the data show an entropy
deficit relative to a block-permutation null, not an entropy excess. The
simulator does not repair that.

## 5. Interpretation

- The simulator is **not** a branching generator in any validated sense. Its
  ensembles can be non-identical because noise is injected, but an ensemble with
  nonzero noise is not learned branching.
- The selected `mlp_residual` model is outside its training support for almost
  all reversal rollouts (OOD about 1.0), so its near-deterministic outcomes are
  extrapolation.
- The marginal null is a strong baseline because reversal outcomes are
  imbalanced. A model that always predicts the majority outcome can look
  superficially stable while carrying no conditional information.
- No parameter was varied to ask whether the vector field changes qualitatively.
  Calling this a bifurcation analysis would be wrong.

## 6. Reproduction

```powershell
python scripts/simulator_branching.py --horizon 20 --rollouts 100 --permutations 200
python scripts/simulator_figures.py
```

Outputs: `reports/simulator/branching_events.csv`,
`branching_summary.json`, `figures/simulator/branching.png`.

## 7. Next experiments that could change the conclusion

1. Fit a model with an explicit discrete outcome head and evaluate calibrated
   conditional probabilities, not only argmax labels.
2. Add a validation-selected branching objective. Selecting on the test
   reversal distribution would leak the evaluation.
3. Increase the number of independent reversal events. Five worms with 4–7
   events each cannot resolve small conditional effects.
4. Test on an independent cohort or a preparation with independent behavioural
   labels. The current behaviour labels are inferred from the same neural
   population.
