# Phase 6 dynamical model selection

**Selection split:** validation block 4 only. **Test block:** not read by `scripts/phase6_select.py`.
**Evidence tier:** exploratory replication because block 2 was previously inspected.

## Predeclared grid

- B1 regularized VAR: lags 1, 2, 3; ridge \(10^{-3}, 10^{-2}, 10^{-1}, 1, 10\); spectral cap 0.995.
- B2 stochastic VAR: Gaussian innovations with Ledoit–Wolf covariance; i.i.d. residual bootstrap; moving-block residual bootstrap (block 5).
- B3 switching VAR: K=2 primary, K=3 sensitivity; five restarts; diagonal Gaussian emissions.
- Baseline: persistence (constant initial state).
- B4 nonlinear: only if B1–B3 reached positive validation skill in at least three worms.

## Result

| Family | Selected configuration | Validation median skill | Positive worms |
|---|---|---:|---:|
| VAR | `var_p2_r10_c0.995` | −0.2209 | 0/5 |
| Stochastic VAR | `stochastic_gaussian` | −0.7820 | 0/5 |
| Switching VAR | `switching_K3` | −3.3084 | 0/5 |

Every configuration had a negative median validation skill. No learned family beat a constant initial state by the predeclared median criterion; individual positive worms occurred but did not change selection. The selected VAR is the least-bad configuration by median skill, not evidence of dynamical skill.

Figure: `figures/simulator/phase6/validation_var_heatmap.png`.

The B4 trigger was **not met** (`positive_worms=0`), so no compact nonlinear model was fitted. Fitting one after seeing the negative validation result would change the selection rule post hoc.

## Selection instability

On the exploratory test block, the validation-selected `var_p2_r10` scored H=20 mean skill +0.0930, while the frozen Phase 5 VAR scored +0.1008 and the p1 variant would have scored higher in the earlier audit. This is a concrete example of selection noise with five worms: the validation and test orderings disagree.

Do not interpret the test ordering as a model improvement. The honest statement is:

> Validation provides no support for a learned autonomous model beyond persistence, and the validation-selected VAR does not outperform the frozen Phase 5 VAR on the reused test block.

## Files

- `reports/simulator/phase6/var_validation_grid.csv`
- `reports/simulator/phase6/var_selection.json`
- `reports/simulator/phase6/stochastic_validation_grid.csv`
- `reports/simulator/phase6/stochastic_selection.json`
- `reports/simulator/phase6/switching_validation_grid.csv`
- `reports/simulator/phase6/switching_selection.json`
- `reports/simulator/phase6/selection_frozen.json`
