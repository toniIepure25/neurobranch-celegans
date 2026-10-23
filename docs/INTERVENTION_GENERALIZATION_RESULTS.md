# Intervention generalization results (Phase 2)

**Status:** executed.
**Artifacts:** `reports/phase2a_scientific_audit/intervention/`,
`reports/phase2b_optogenetic_atlas/`
**Code:** `src/neurobranch/interventions/audit.py`,
`src/neurobranch/evaluation/randi_benchmark.py`
**Scripts:** `scripts/phase2a_intervention_audit.py`,
`scripts/phase2a_optogenetic_benchmark.py`

> Every result in Part A is a **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT
> CAUSALLY VALIDATED.** Part B uses real optogenetic data but tests prediction,
> not causation, and the model was not fitted to those trials.

## Part A — Is the Phase 1 RIVL effect specific?

### A.1 Design

- 13 literature-motivated candidate cells × amplitudes `−2` and `+2`
  standardized units = 26 cells.
- 6 persisted models: MLP seeds 17/23/31, residual MLP seed 17, VAR seed 17,
  GRU seed 17.
- 19 matched non-candidate controls per model: same amplitude, same initial
  states, same simulation seeds, same horizon (`12` frames), same target
  behaviour (`TURN`), and current state `REV`.
- 30 initial frames × 3 simulation seeds per cell.
- Primary readout: change in `P(TURN)` relative to the paired no-intervention
  rollout; cluster bootstrap over initial frames.

Matching on the observed effect would be circular. Controls are matched on the
**design**, not on the outcome.

### A.2 RIVL across architectures and seeds

| Model | RIVL +2 ΔP(TURN) [95% CI] | Controls ≥ RIVL | Empirical p | Control max | OOD fraction |
|---|---:|---:|---:|---:|---:|
| MLP seed 17 | 0.333 [0.167, 0.533] | 4/19 | 0.25 | 0.667 | 0.106 |
| MLP seed 23 | 0.733 [0.567, 0.867] | 2/19 | 0.15 | 0.767 | 0.485 |
| MLP seed 31 | 0.267 [0.133, 0.433] | 6/19 | 0.35 | 0.567 | 0.359 |
| Residual MLP 17 | 0.000 [0.000, 0.000] | 19/19 | 1.00 | 0.000 | 0.994 |
| VAR 17 | 0.544 [0.378, 0.695] | 2/19 | 0.15 | 0.544 | 0.939 |
| GRU 17 | 0.367 [0.200, 0.567] | 1/19 | 0.10 | 0.422 | 0.245 |

No model reaches `p < 0.05`. The effect is model- and seed-dependent: it ranges
from zero (residual MLP) to 0.733 (MLP seed 23). In the original MLP seed 17, the
strongest control is `OLQVL` at **0.667**, twice RIVL's effect, with a lower OOD
fraction.

Selection-adjusted max-statistic tests (best candidate vs maximum control):

| Model | Best candidate effect | Control max | p(max control ≥ best) |
|---|---:|---:|---:|
| MLP 17 | 0.333 | 0.667 | 0.25 |
| MLP 23 | 0.733 | 0.767 | 0.15 |
| MLP 31 | 0.422 | 0.567 | 0.20 |
| Residual MLP 17 | 0.000 | 0.000 | 1.00 |
| VAR 17 | 0.544 | 0.544 | 0.15 |
| GRU 17 | 0.367 | 0.422 | 0.10 |

The candidate set was chosen from prior literature, not from this worm's
effects. Even so, the **winner** is not distinguishable from the best arbitrary
control.

### A.3 Candidate ranking stability

At amplitude +2, across the five non-degenerate models:

- pairwise Spearman of the 13-candidate ranking ranges from **−0.049 to 0.786**;
- top-5 overlap ranges from **1 to 4** of 5;
- RIVL is in the top 5 for 5/6 models, but the residual MLP has all-zero effects
  and its ranking is arbitrary.

RIVL is therefore a *recurrent* top candidate, not a *specific* one.

### A.4 Amplitude and sign

| Cell | −3 | −2 | −1 | +0.5 | +1 | +2 | +3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RIVL | −0.100 | −0.089 | −0.044 | 0.000 | 0.011 | 0.333 | 0.667 |
| RIVR | −0.100 | −0.100 | −0.089 | 0.000 | 0.033 | 0.233 | 0.400 |
| OLQVL | −0.100 | −0.100 | −0.100 | 0.056 | 0.289 | 0.667 | 0.767 |
| OLQVR | −0.100 | −0.100 | −0.089 | 0.022 | 0.133 | 0.633 | 0.767 |
| RIFR | −0.100 | −0.067 | −0.067 | 0.000 | 0.000 | 0.078 | 0.244 |
| ASKR | 0.078 | 0.067 | 0.000 | −0.033 | −0.033 | −0.067 | −0.100 |

The response is strongly asymmetric and threshold-like. Large positive
amplitudes move nearly any cell's readout; negative amplitudes mostly produce a
−0.1 floor. `OLQVL` and `OLQVR` exceed RIVL at +2. This is the signature of an
off-support stress test, not a dose–response curve.

### A.5 Part A conclusion

The Phase 1 RIVL result **does not survive** a specificity audit. It is not
distinguishable from arbitrary non-candidate cells, is not stable across seeds
or architectures, and is dominated by amplitude/OOD behaviour.

## Part B — Real optogenetic response prediction (Randi 2023)

### B.1 Data and target

- 113 source datasets; 110 with a valid event; 3,371 valid events;
  237,639 downstream rows; 259 stimulated cells; 305 recorded cells;
  40,341 target–neuron pairs; 110 animals.
- Response = signed peak ΔGCaMP in frames +1…+12 relative to a −8…−1 median
  baseline; stimulated neuron's own trace excluded.
- Ambiguous duplicate labels excluded. See
  [`INTERVENTION_DATASET_AUDIT.md`](INTERVENTION_DATASET_AUDIT.md) and
  [`NEURON_IDENTITY_MAPPING.md`](NEURON_IDENTITY_MAPPING.md).

### B.2 Repeatability ceiling

Repeated stimulation of the same target in the same animal:

| Quantity | Value |
|---|---:|
| Repeat groups | 580 |
| Event pairs | 925 |
| Median pairwise response-profile r | **0.090** |
| Mean pairwise r | 0.085 |
| Fraction positive | 63.2% |
| IQR | [−0.126, 0.297] |

Even before prediction, single-trial response profiles are only weakly
repeatable. This is a hard ceiling on any pre-stimulus predictor and a major
reason the benchmark is negative.

### B.3 Held-out benchmark

Three group-disjoint protocols:

1. **heldout_events**: 20% of events held out; same animals/targets.
2. **heldout_targets**: 5-fold by stimulated cell; new targets.
3. **heldout_animals**: 5-fold by dataset; new animals.

Models: zero, training mean, neuron mean (shrunk), target mean (shrunk),
additive target+neuron, one-hot ridge (target + recorded cell + pre-activity),
and connectome ridge (Witvliet chemical/electrical edge counts + pre-activity).
All scalers/means are fitted on training rows only.

Primary target `response`; aggregate mean across folds (event protocol is one
split). Full table: `benchmark_aggregate_response.csv`.

| Protocol | Global mean MAE | Best non-constant | MAE | Δ vs global | Pearson |
|---|---:|---|---:|---:|---:|
| heldout_events | 6.1113 | connectome ridge | 6.1112 | **−0.000086** | 0.148 |
| heldout_targets | 5.7251 | connectome ridge | 5.7467 | +0.0216 | 0.094 |
| heldout_animals | 5.7249 | neuron mean | 5.7919 | +0.0670 | 0.001 |

The event-level connectome "win" is numerically zero. Paired cluster bootstrap
over 675 held-out events:

| Target | Δ MAE (connectome − global) | 95% CI | Bootstrap fraction better |
|---|---:|---:|---:|
| response | −0.000086 | [−0.0177, +0.0165] | 0.502 |
| response_residual | +0.000036 | [−0.0179, +0.0169] | 0.495 |

The confidence interval is symmetric around zero and the bootstrap is a coin
flip. Under new targets and new animals, every learned model is worse than the
constant mean. Residualizing the event-wide response does not change this.

`sign_accuracy` is ~0.538 for the constant mean (the majority sign), and the
learned models do not materially exceed it. The `zero` model's 0.0005 is an
artifact of `sign(0) = 0`; it is not a meaningful direction predictor.

### B.4 Part B conclusion

On this audit's response target, **pre-stimulus state, target identity, recorded
cell identity, and connectome edges do not predict held-out optogenetic
responses better than a constant**. The result holds for new events, new
targets, and new animals. This is a negative result about the benchmark and the
available signal; it is not evidence that the biological propagation atlas is
uninformative, because the response detector and repeatability ceiling are
limiting.

## Part C — Answers to RQ4 and RQ5

**RQ4 (predict real optogenetic responses):** No positive transfer. The best
model is a tie with the constant mean on held-out events and worse on held-out
targets/animals. Repeated-response reliability is only r ≈ 0.09.

**RQ5 (conditions for causal intervention generalization):** Not established.
The internal candidate is nonspecific; the external benchmark is predictive, not
interventional; no perturbation was performed on a living animal by this
project. The evidence gate correctly refuses a causal label.
