# JOINTDYN research protocol

**Status:** executed, frozen before final scoring summary  
**Branch:** `research/joint-dynamics-calibration`  
**Experiment type:** exploratory repeated-session reanalysis, not a systematic review and not an external validation  
**Data:** Kato et al. (2015) `WT_NoStim`, five repeated recording sessions

## 1. Question

Does a corrected population-level stochastic dynamics model reproduce the **joint conditional distribution** of future neural trajectories better than strong independent-neuron baselines?

Three claims are kept separate:

- **CLAIM A - marginal neural forecasting:** per-neuron predictive distributions are better.
- **CLAIM B - joint neural trajectory forecasting:** the multivariate and trajectory distributions are better.
- **CLAIM C - faithful generative model:** marginal calibration, joint covariance, temporal structure and robustness jointly pass.

Passing A does not imply B. Passing B does not imply C. None implies biological causal intervention validity.

## 2. Why this follow-up exists

The previous uncertainty-calibration study found that pooling multi-step forecast errors as one-step innovations inflated the stochastic population model. Correcting the pool reduced H20 energy from 4.609 to 2.700 and moved 95% full-neuron coverage from 0.999 to 0.947. The corrected model still over-covered at 50% (0.588) and had joint covariance error 0.604 in the present evaluation.

The corrected population model was examined after earlier test results had already been observed. Its apparent advantage is therefore **not independently confirmed** by this experiment.

## 3. Data and splits

| Item | Value |
|---|---|
| Dataset | `data/raw/kato2015/WT_NoStim.mat` |
| Sessions | 5 repeated recordings, worm IDs 0-4 |
| Frames | 3137, 3134, 3059, 3311, 3021 |
| Predictors | 103 (one session 102) |
| Sampling | approximately 2.8-3.1 frames/s |
| Train | frames `0:2195` (per-session fraction) |
| Validation | frames `2207:2677` |
| Test | frames `2689:3137` |
| Gaps | 12-frame purge retained |
| Test landmarks | 35 per session, 175 total |
| Landmark stride | 10 frames; overlapping windows |
| Evaluation projection | fixed 10-component PCA fitted on training frames only |

The five sessions are repeated recordings, not five independent animals. The 175 landmarks are not independent observations. All test comparisons are exploratory because the Kato test split has been analysed repeatedly.

## 4. Frozen model set

All models use the same landmarks, the same initial observed state and the same fixed PCA projection. Innovation pools contain only `lead == 1` prequential residuals unless the historical invalid comparator is explicitly identified.

| ID | Mean | Innovation |
|---|---|---|
| J0 persistence | persistence | moving-block bootstrap of persistence residuals, block 20 |
| J1 independent diag | per-neuron AR(1), window 200, ridge 1.0, cap 0.995 | diagonal Gaussian, lead-1 residuals |
| J2 population diag | rolling population VAR(1), window 500, ridge 1.0, cap 0.995 | diagonal Gaussian, lead-1 residuals |
| J3 population full | same population mean | Ledoit-Wolf full covariance, lead-1 residuals |
| J4 independent full | same independent mean | Ledoit-Wolf full covariance, lead-1 residuals |
| J5 population lead-1 | same population mean | moving-block bootstrap, lead-1 pool only, block 10 |
| P6 historical block | population VAR | block bootstrap of multi-step residuals, block 10; **invalid diagnostic only** |

J1/J4 and J2/J3 form a 2 x 2 factorial: independent vs population mean crossed with diagonal vs full innovation covariance. This isolates conditional-mean information from covariance information.

Ensembles use 100 members. Seed: `7000 + worm_id * 10000 + landmark`.

## 5. Stages

- **A. Reproduction:** rerun persistence, corrected lead-1 population and the historical overdispersed population model; compare with the frozen previous report.
- **B. Population information:** factorial mean/covariance decomposition and paired contrasts.
- **C. Innovation structure:** lag 1-20 residual autocorrelation, temporal cross-covariance, conditional residual mean, effective rank, split-half covariance stability, factor AR structure.
- **D. Calibration:** D1 historical corrected lead-1, D2 regularized full covariance, D3 temporal innovation only if Stage C justified it, D4 one Student-t scale calibration fitted only on validation.
- **E. Probabilistic evaluation:** H1/H5/H10/H20/H50/H100; primary H20 state energy, H20 trajectory energy and 50/80/95 coverage.
- **F. Robustness:** session-level contrasts, bootstrap block lengths 3/5/10, top-20 training-variance neurons, all neurons, exclusion of the top-5 training-variance neurons.
- **G. Generative trajectories:** observed continuation, conditional means, stochastic continuations, covariance, autocorrelation, occupancy and recurrence.
- **H. Parameter uncertainty:** synthetic audit and a five-draw causal moving-block bootstrap of transition parameters.

## 6. Primary metrics

- H20 multivariate energy score in the fixed 10-D PCA.
- H20 joint trajectory energy over the flattened 20-frame path.
- Full-neuron CRPS for marginal forecasting.
- Central coverage at nominal 50%, 80% and 95%.
- Pooled neural covariance relative error and within-path covariance error.
- Lag-1 autocorrelation profile error.
- Variogram score.
- Ensemble spread, mean RMSE, OOD fraction, long-horizon finiteness.

Lower energy and CRPS are better. A model is not called calibrated because one nominal level happens to be close.

## 7. Frozen gates

The machine-readable thresholds are in `reports/simulator/joint_dynamics/gate_thresholds.json`.

| Gate | Criterion |
|---|---|
| G1 | reproduction relative difference <= 2% |
| G2 | J5 beats J1 on H20 energy by at least 0.05, paired 95% CI lower bound > 0 |
| G3 | population median absolute ACF1 <= 0.15 and ACF5 <= 0.10; no multi-step errors in the pool |
| G4 | full-neuron 50/80/95 coverage within 0.05 at H20 and maximum horizon deviation <= 0.10 |
| G5 | trajectory energy no worse than persistence; neural covariance error <= 0.5; autocorrelation error <= 0.35 |
| G6 | the main direction holds in at least 4/5 sessions and declared sensitivity analyses |
| G7 | G2, G4, G5 and G6 all pass |
| G8 | biological intervention: `NOT_EVALUABLE` |

## 8. Stop rule

This is not another architecture search. If the corrected population model has no reliable advantage, the negative result is retained. If a marginal score improves while calibration and trajectory fidelity fail, that is reported explicitly. If parameter uncertainty worsens dispersion, no further stochastic components are added. If G7 fails, autonomous model expansion stops.

## 9. Perturbation safety

No neuron ranking or stimulation-target optimization was launched. No biological causal claim is permitted.

**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

Biological intervention validation: `NOT_EVALUABLE`.

## 10. Deviations

- D3 was not fitted: Stage C found median one-step population ACF1 = 0.0217 and ACF5 = 0.0078, below the prespecified justification threshold.
- D4 Student-t sampling was normalized to unit covariance after a unit test caught an unnormalized scale; D4 was rerun and remains a negative result.
- No Kato stimulation file is present in the repository, so a stimulation per-session refit was not attempted.
