# Probabilistic models and model cards

This document answers Stage D and Stage E: which small innovation models were compared, and what exactly is probabilistic about them? The configuration is frozen in `configurations.json`; cards are in `model_cards.json`.

## 1. Model space

The experiment did not run an architecture search. Five innovation choices were prespecified, and only those justified by the residual audit were fitted.

| ID | Innovation | Status |
|---|---|---|
| D1 | diagonal Gaussian from lead-1 prequential residuals | **primary** (`P2_independent_ar_lead1`) |
| D2 | state-conditioned diagonal Gaussian | fitted as `P3_state_ar` |
| D3 | low-rank plus diagonal | **not built**; held-out likelihood selected diagonal in 4/5 sessions |
| D4 | validation-only global variance scaling | diagnostic only |
| D5 | historical block bootstrap | frozen negative comparator |

The low-rank decision is in `lowrank_decision.json`. The D4 factors are in `d4_validation.csv` and `d4_summary.json`.

## 2. D1: independent adaptive AR with lead-1 diagonal innovations

**Mean.** For each neuron `j`:

\[
\hat{x}_{j,t+1} = a_j x_{j,t} + b_j,
\]

with `a_j` fitted by ridge regression on a 200-frame causal window, clipped to `|a_j| <= 0.995`. No cross-neuron terms.

**Innovation.** A diagonal Gaussian with variance estimated from causal lead-1 block-prequential residuals:

- fit at each block boundary using only data before the block;
- predict each frame in the block at lead 1;
- pool the resulting one-step residuals;
- regularize the diagonal scale, with a floor;
- use the most recent 1000 residuals.

This is a **marginal per-neuron** model. It does not model cross-neuron covariance. Its joint samples are independent across neurons conditional on the mean.

**Why it is the primary.** It is the simplest model that (a) uses a causal one-step innovation, (b) adapts online, (c) avoids the invalid block-50 pool, and (d) can be evaluated with the same proper scores as the historical model. It was fixed before the final evaluation.

**Card.** Window 200; ridge 1.0; cap 0.995; residual block 10; lead 1; diagonal Gaussian; history 1000; 100 members.

## 3. D2: state-conditioned diagonal

`P3_state_ar` keeps the D1 mean and replaces the constant diagonal scale with a per-neuron ridge model of log variance on four causal features:

- absolute level;
- absolute one-step change;
- recent standard deviation (10 frames);
- recent absolute mean (10 frames).

The fit uses training data only, with a smearing correction and a variance clip. The intent was to test whether heteroscedasticity, if present, could be captured without a population covariance.

**Result.** It did not improve CRPS (0.419 vs 0.418) or coverage (0.463/0.739/0.884 vs 0.471/0.746/0.890). The state-variance regression had near-zero or negative held-out R². The model is retained as a negative result, not promoted.

## 4. D3: low-rank plus diagonal was not justified

The residual audit permitted D3 only if low-rank structure was stable. A held-out Gaussian likelihood over ranks `{0,1,3,5,10}` per session selected:

- rank 0 in sessions 0, 2, 3, 4;
- rank 3 in session 1;
- mean rank 0.6;
- diagonal wins 4/5.

The retained eigenvalue fraction at rank 3 was only about 0.17 in session 1. There was no stable population factor. Building a low-rank model would have been a model-zoo addition without evidence. It was not built, and the final comparison contains no D3 row.

This decision is a result: **the residual covariance did not support a low-rank innovation model**.

## 5. D4: validation-fitted global scaling is diagnostic only

D4 fits one global variance factor on the validation split and applies it to the diagonal Gaussian. The fitted factors were:

| Session | Factor |
|---|---:|
| 0 | 2.974 |
| 1 | 1.843 |
| 2 | 4.000 |
| 3 | 3.794 |
| 4 | 2.181 |
| mean | 2.959 |

The factors are large and unstable. On test, D4 gave:

| Metric | D4 |
|---|---:|
| CRPS H20 | 0.4465 |
| Coverage 50 / 80 / 95 | 0.691 / 0.915 / 0.974 |
| Width 95 | 4.064 |
| Energy H20 | 2.961 |
| Trajectory energy H20 | 2.358 |

D4 improves the 95% coverage relative to D1 but **worsens CRPS** (0.446 vs 0.418) and over-covers at 50% and 80%. It hides the defect by inflating every interval. It is reported as a diagnostic, not as a calibrated model.

The validation factor is not transportable across sessions: a factor fitted on one session's validation frames would be wrong for another. This is why the protocol forbids selecting D4 as the primary.

## 6. D5: historical bootstrap, frozen negative

`P6_adaptive_block` is the historical model, rerun with its original definition:

- rolling population VAR, window 500;
- adaptive residual pool from a block-50 prequential series, all leads 1-50;
- moving-block bootstrap, block 10, history 1000.

It reproduces the frozen energy 4.6046 as 4.6088, coverage 0.904/0.992/0.999 as 0.910/0.992/0.999, and width 9.967 as 10.041. The maximum relative difference is 0.74%, within the Gate A tolerance.

It is the negative comparator that motivates the whole experiment.

## 7. Diagnostic correction: population lead-1

`P6_adaptive_lead1` keeps the rolling population VAR mean but uses only lead-1 residuals for the innovation. It is a diagnostic test of the pooling hypothesis, not a replacement primary.

| Metric | P6 block | P6 lead 1 | D1 primary |
|---|---:|---:|---:|
| Energy H20 | 4.609 | **2.700** | 3.146 |
| CRPS H20 | 0.649 | **0.413** | 0.418 |
| Coverage 50 | 0.910 | 0.588 | 0.471 |
| Coverage 80 | 0.992 | 0.849 | 0.746 |
| Coverage 95 | 0.999 | **0.948** | 0.890 |
| Width 95 | 10.041 | 3.090 | 2.385 |
| Covariance relative error | 0.862 | 0.492 | 0.729 |

Correcting the pool cuts energy by 41% and brings 95% coverage close to nominal. It still over-covers at 50% and does not reproduce the joint covariance. The correction is real but partial.

## 8. What each model claims

| Model | Deterministic mean | Marginal predictive distribution | Joint trajectory distribution |
|---|---|---|---|
| P0 persistence | yes | approximate, 95% close | no |
| P1 local mean | yes | approximate | no |
| D1 primary | yes | **not calibrated** | no |
| D2 state AR | yes | **not calibrated** | no |
| D3 low-rank | not built | not built | not built |
| D4 scaled | yes | **over-inflated, not calibrated** | no |
| D5 historical | yes | **severely overdispersed** | no |
| P6 lead 1 | yes | 95% close, 50% over | no |

No row in this table is a validated generative simulator.

## 9. Reproducibility

```powershell
python scripts/uncal_evaluation.py
python scripts/uncal_d4.py
python scripts/uncal_lowrank.py
python scripts/uncal_synthetic.py
python scripts/uncal_summarize.py
```

`uncal_lowrank.py` writes the rank decision; `uncal_d4.py` writes the validation factors and test scores; `uncal_synthetic.py` writes the identifiability controls.

## Related documents

- [Simple baselines](UNCAL_SIMPLE_BASELINES.md)
- [Calibration](UNCAL_CALIBRATION.md)
- [Trajectory fidelity](UNCAL_TRAJECTORY_FIDELITY.md)
- [Results](UNCAL_RESULTS.md)
