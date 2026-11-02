# Neural uncertainty and calibration: research protocol

**Status:** frozen before the final evaluation reported in `UNCAL_RESULTS.md`.
**Scope:** a focused diagnostic and corrective experiment, not an architecture search.
**Data:** Kato *C. elegans* WT_NoStim, five repeated sessions, existing train-standardized calcium activity.
**Primary question:** why were the earlier probabilistic adaptive simulators severely overdispersed, and can a smaller, properly calibrated probabilistic model generate credible future neural-activity distributions?

The earlier reports and artifacts are frozen under `reports/simulator/probabilistic_adaptive/`. Nothing in this experiment amends, resets, rebases or force-pushes that history. No raw recording or model weight is committed.

## 1. What was known before this experiment

The historical stochastic model was a 10-dimensional population VAR with an adaptive/rolling mean and a moving-block bootstrap of residuals. It had a lower deterministic H20 MSE than persistence but a much worse proper score:

| Quantity | Historical value |
|---|---:|
| H20 MSE, frozen mean | 1.597159 |
| H20 MSE, rolling population VAR | 0.940602 |
| H20 energy score, primary stochastic model | 4.6046 |
| Coverage 50 / 80 / 95 | 0.904 / 0.992 / 0.999 |
| Total standard-deviation ratio | 3.398 |
| Residual lag-1 absolute ACF | 0.584 |
| PCA effective rank | 27.4 |

A model can have a low mean error and still be a bad probability model. The historical result was therefore not evidence for a calibrated generative simulator.

## 2. Hypotheses

- **H1 (innovation pooling):** the residual pool mixed one-step innovations with multi-step forecast errors, inflating the innovation covariance.
- **H2 (serial dependence):** block bootstrapping preserved residual serial dependence inside each innovation, so uncertainty accumulated more than a white-noise model predicts.
- **H3 (operator amplification):** the fitted population operator was near unit root and non-normal, so even correctly scaled innovations were amplified transiently over the forecast horizon.
- **H4 (misspecification):** residual variance depended on state, time or neuron, so a constant diagonal innovation was wrong.
- **H5 (implementation):** the ensemble code or covariance propagation contained an error.
- **H6 (irreducible joint uncertainty):** even a calibrated marginal model would fail to reproduce the joint trajectory covariance.

H1-H3 are not mutually exclusive. The analysis was designed to attribute variance rather than to pick one story.

## 3. Data and splits

Only the existing processed arrays were used. The split boundaries already stored in the data object were respected:

- train: frames `0:2195`
- validation: `2207:2677`
- test: `2689:3137`
- gaps are retained; no frame from validation or test enters a fit.
- test landmarks: `landmark_sequence(test_start, history=2, horizon=100)[::10]`, 35 per session, 175 total.
- horizons: H1, H5, H10, H20, H50, H100. **H20 is primary.**
- H20 test windows overlap by 90 frames; H50 and H100 overlap more. Landmark bootstrap intervals are therefore blocked and are not animal-level confidence intervals.

Five repeated sessions are treated as repeated measurements, not five independent animals. Per-session results are reported separately.

## 4. Fixed representation

Forecasts are evaluated in the training-fitted 10-component PCA space used by the historical experiment, so scores are comparable. The PCA is fitted on training frames only. Neuron-level marginal scores use the same invertible projection back to neuron space. No test data is used to choose the projection.

## 5. Models compared

**Frozen comparators**

- `P0_persistence`: last state plus a moving-block bootstrap of historical first differences.
- `P1_local_mean`: 100-frame local mean plus a moving-block bootstrap of local residuals.
- `P6_adaptive_block`: historical rolling population VAR, block-50 adaptive residual pool, leads 1-50.
- Historical frozen AR comparator, where available in the frozen artifacts.

**Primary prespecified model**

`P2_independent_ar_lead1`:

- independent per-neuron AR(1), window 200, ridge 1.0, spectral cap 0.995;
- diagonal Gaussian innovations;
- scale and mean estimated from causal **lead-1 block-prequential residuals** (block 10, history 1000);
- online causal updates; no future observation enters a prediction;
- 100 ensemble members; seed `7000 + worm_id*10000 + landmark`.

**Secondary models**

- `P2_fixed_train_scale`: same mean, innovation scale frozen from training residuals only.
- `P3_state_ar`: same mean, diagonal variance conditioned on four causal state features.
- `P6_adaptive_lead1`: diagnostic correction of the historical population model using lead-1 residuals.
- `D4_scaled_diag`: validation-only global variance factor; diagnostic, never eligible as a mechanistic repair.
- `D5_historical_bootstrap`: the historical bootstrap retained as a frozen negative comparator.

The primary model was fixed before the final evaluation. `P6_adaptive_lead1` was added as a diagnostic because H1 predicted that it should improve. It is not substituted for the primary after seeing results.

## 6. Evaluation

Primary score: **mean per-neuron CRPS at H20**, lower is better.

Secondary proper and diagnostic scores:

- multivariate energy score, variogram score;
- PIT histogram and interval coverage at 50/80/95%;
- interval width, standardized-error variance, spread-error ratio;
- variance ratio, covariance relative error, lag-1 autocorrelation error;
- OOD fraction, finite fraction, update time;
- joint H20 trajectory energy in the flattened 10-D PCA path;
- per-session and per-neuron breakdowns;
- paired block-bootstrap intervals over landmarks.

The marginal predictive distribution and the joint trajectory distribution are separate targets. Passing marginal coverage does not establish a generative trajectory model.

## 7. Pre-registered gates

| Gate | Pass condition |
|---|---|
| A. Reproduction | rerun of the frozen historical comparator reproduces energy and coverage within 1% relative |
| B. Variance diagnosis | at least one measured mechanism accounts for a substantial fraction of the inflation, with analytical/Monte Carlo agreement |
| C. Innovation validity | primary residual ACF and state predictability are small; if not, the innovation is labelled invalid |
| D. Proper score | H20 CRPS beats P0 and P1; paired 95% lower bound > 0 for both |
| E. Marginal calibration | 50/80/95 coverage within 0.07/0.07/0.05; PIT error < 0.05; width ratio 0.3-2.0; standardized-error variance 0.7-1.4; no horizon deviation > 0.12 |
| F. Joint trajectory | trajectory energy beats P0 with 95% lower bound > 0; total std ratio 0.75-1.35; lag-1 error < 0.25; covariance relative error < 0.50; OOD < 0.20; finite |
| G. Generative credibility | D, E and F all pass |
| H. Biological intervention | `NOT_EVALUABLE`; no intervention was performed |

The stop labels were fixed:

- `CALIBRATED MARGINAL FORECASTING WITHOUT VALIDATED GENERATIVE TRAJECTORIES` only if D and E pass and F fails;
- `PROBABILISTIC CALIBRATION NOT ESTABLISHED` otherwise;
- `VALIDATED_GENERATIVE_SIMULATOR` only if D, E and F all pass.

No result below changes these gates after the fact.

## 8. What would count as a negative outcome

A negative outcome is informative and is retained:

- the primary model can improve CRPS and still fail marginal calibration;
- a corrected population model can fix 95% coverage and still fail 50% coverage and joint covariance;
- a validation-fitted variance factor can make coverage look better while making CRPS worse;
- the historical low MSE can coexist with severe overdispersion.

## 9. Reproducibility contract

All scripts write to `reports/simulator/uncertainty_calibration/`. The report manifest records SHA-256 for every artifact. Commands are listed in `metadata.json` and in `UNCAL_RESULTS.md`. The analysis uses single-threaded BLAS (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`) so that ensemble results are stable.

No claim in this report is a systematic review, a new external validation, or a causal intervention result.

## Related documents

- [Variance propagation](UNCAL_VARIANCE_PROPAGATION.md)
- [Innovation validity](UNCAL_INNOVATION_VALIDITY.md)
- [Simple baselines](UNCAL_SIMPLE_BASELINES.md)
- [Probabilistic models](UNCAL_PROBABILISTIC_MODELS.md)
- [Calibration](UNCAL_CALIBRATION.md)
- [Trajectory fidelity](UNCAL_TRAJECTORY_FIDELITY.md)
- [Results](UNCAL_RESULTS.md)
- [Limitations](UNCAL_LIMITATIONS.md)
- [Supervisor brief](UNCAL_SUPERVISOR_BRIEF.md)
- [Next experiments](UNCAL_NEXT_EXPERIMENTS.md)
