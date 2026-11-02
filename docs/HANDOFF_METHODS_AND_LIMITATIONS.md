# Methods, provenance and limitations

**Study type:** exploratory repeated-session reanalysis. This is not a systematic
review, a prospective study, an external validation, or an intervention study.

## 1. Data actually used

| Dataset | Role | What is available in this checkout |
| --- | --- | --- |
| Kato et al. (2015), `WT_NoStim` | Primary neural dynamics data | `data/raw/kato2015/WT_NoStim.mat`, five sessions, 15,662 frames |
| Randi et al. (2023) | Intervention-response benchmark | Parsed event tables under `reports/phase3/`, raw archive not committed |
| Baskoylu & Flavell (2026) | Independent behavioural/neural cohort | Raw data not committed; derived reports under `reports/phase3/` and `reports/phase4/` |
| Kato et al. (2015), `WT_Stim` | Historical external autonomous-dynamics pilot | **Not present in this checkout**; historical pilot artifacts under `reports/external_validation/` |
| Kramer et al. (2026) | Turn-direction reconstruction | Raw data not committed; derived reports under `reports/phase5/` |

The primary Kato file is five **recording sessions**, not five documented
independent animals. The OSF release does not provide animal IDs. Every
animal-level statement must say "session" unless independence is separately
established.

### 1.1 Provenance discrepancy: Kato Stim

`docs/DATA_PROVENANCE.md` states that no `WT_Stim.mat` was downloaded or
analysed. That sentence is true for the original Phase 1/2 data inventory, but it
is superseded by later external-validation work:

- `reports/external_validation/dataset_inventory.json` records seven Kato Stim
  sessions and their SHA256 values.
- `reports/external_validation/kato_stim_pilot_summary.json` records a completed
  per-animal pilot, run 2026-10-31, with seven sessions fitted and 0/7 positive.
- `docs/SIMULATOR_CONTEXT_RESULTS.md` and `docs/GENFAIL_OXYGEN_RESPONSE.md` also
  analyse Kato Stim oxygen schedules.

The raw file is absent from the checkout at handoff, so the historical pilot
cannot be rerun without re-downloading it from OSF. The handoff reports the pilot
as exploratory and does not present it as a current-workspace reproduction. Do
not "fix" the historical `DATA_PROVENANCE.md` sentence by deleting the later
evidence; record the scope and date of each statement.

### 1.2 Why raw data are not committed

The Kato OSF project has no declared licence in its metadata. Raw `.mat` files,
Randi exports and Kramer files are excluded from Git. Reproduce from the public
sources and verify the hashes in `docs/DATA_PROVENANCE.md` and the relevant
manifests.

## 2. Primary evaluation design

### 2.1 Sessions, frames and splits

- Five Kato `WT_NoStim` sessions.
- 15,662 frames total, ~2.8–3.1 fps.
- Predictors: 103 per session, one session with 102; behaviour-label neurons
  excluded.
- Blocked split used in the historical dynamics work: `0:2195`, `2207:2677`,
  `2689:3137`, with a 12-frame purge.
- Joint-dynamics test landmarks: 175 overlapping landmarks, 35 per session,
  stride 10, H1/H5/H10/H20/H50/H100; primary H20.
- Landmarks overlap. They are not independent samples, and the 175 count is not
  175 animals or 175 independent trajectories.

### 2.2 Fixed representation

A 10-D PCA was fitted on **training frames only** and then frozen for all models
and all test landmarks. It explains 54.7%–71.0% of variance depending on session.
The projection is an evaluation coordinate system, not a biological latent space.
It has no validated inverse decoder.

The choice of 10-D was fixed before the joint-dynamics evaluation, but it is not
a claim that the true manifold is 10-D. The all-neuron and top-20 sensitivity
analyses are reported precisely because the representation matters.

### 2.3 Ensembles and seeds

- 100 ensemble members per model and landmark.
- Seed: `7000 + worm_id*10000 + landmark`.
- Single-thread BLAS for determinism:
  `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `NUMEXPR_NUM_THREADS=1`.
- All reported ensembles are finite at H100; finiteness is not calibration.

## 3. Models

All models share the same landmarks, the same initial observed state and the same
fixed PCA projection. Innovation pools contain only `lead == 1` prequential
residuals unless the historical invalid comparator is explicitly named.

| ID | Conditional mean | Innovation |
| --- | --- | --- |
| J0 persistence | hold initial state | moving-block bootstrap of persistence residuals, block 20 |
| J1 independent AR | per-neuron AR(1), window 200, ridge 1.0, spectral cap 0.995 | diagonal Gaussian, lead-1 residuals |
| J2 population VAR | rolling population VAR(1), window 500, ridge 1.0, cap 0.995 | diagonal Gaussian, lead-1 residuals |
| J3 population full | same population mean | Ledoit-Wolf full covariance, lead-1 residuals |
| J4 independent full | same independent mean | Ledoit-Wolf full covariance, lead-1 residuals |
| J5 population lead-1 | same population mean | moving-block bootstrap, lead-1 pool only, block 10 |
| P6 historical block | population VAR | block bootstrap of multi-step residuals, block 10; **invalid diagnostic only** |

J1/J4 and J2/J3 form a 2 × 2 factorial: independent versus population mean
crossed with diagonal versus full innovation covariance.

### 3.1 What "corrected" means

The historical stochastic model pooled residuals from leads 1–50 and resampled
them as if they were one-step innovations. That injected horizon-inflated,
serially dependent errors at every rollout step. The corrected pool uses only
one-step causal residuals. This is a **construction correction**, not a new
biological model.

J2 and J5 have effectively identical scores (energy 2.700; trajectory 2.058 versus
2.060; coverage 0.586/0.846/0.944 versus 0.588/0.849/0.947). The mean advantage
does not depend on the block-bootstrap versus parametric-diagonal choice. J5 is
used for the headline paired intervals because it is the corrected comparator
frozen in the joint-dynamics protocol; J2 is an internal replication of the mean
effect.

## 4. Metrics

### 4.1 Energy score

For an ensemble forecast distribution \(P\) and observed vector \(y\), the energy
score is

\[
E(P,y) = \mathbb{E}\|X-y\| - \tfrac{1}{2}\mathbb{E}\|X-X'\|,
\]

with \(X,X'\) independent draws from \(P\). Lower is better. The H20 state energy
is computed in the 10-D PCA; all-neuron energy is a sensitivity analysis.

### 4.2 Trajectory energy

The trajectory score treats the 20-step path as one object rather than averaging
per-time energy. It penalizes a forecast that has good marginal states but the
wrong temporal ordering or path shape. Lower is better.

### 4.3 CRPS and coverage

CRPS is computed per neuron and averaged. Full-neuron CRPS is a marginal score; it
does not test joint dependence. Coverage is the fraction of observed values inside
the central 50/80/95% ensemble interval. Nominal coverage is not a quality score
by itself; sharpness and covariance matter.

### 4.4 Covariance and autocorrelation error

Neural covariance error is a relative error between simulated and observed H20
neural covariance. Autocorrelation error compares simulated and observed
temporal correlation profiles. The frozen joint gate requires covariance error
≤ 0.50 and autocorrelation error ≤ 0.35. J5 passes autocorrelation (0.164) and
fails covariance (0.604).

### 4.5 Variogram score

The variogram score is a proper multivariate score based on pairwise absolute
differences. It is reported as a secondary metric and is not used to rescue the
failed calibration gate.

## 5. Inference

- Paired contrasts use a moving-block bootstrap over the 175 test landmarks.
  Blocks of 5 are the default; blocks 3, 5 and 10 are reported as sensitivity.
- "Improvement" is reference minus model, so positive favours the first model.
- Intervals are not animal-level intervals. The resampling unit is a landmark,
  and landmarks overlap.
- The 5 sessions contribute 35 landmarks each; session-level intervals are
  descriptive and are not independent animal estimates.
- No multiplicity correction is applied to the exploratory secondary metrics.
  They are reported as a sensitivity family, not as confirmatory tests.
- The test split has been reused across phases. The joint-dynamics gate was
  frozen before this run, but the model choice was informed by earlier results.
  It is therefore not a confirmatory test.

### 5.1 What would make the result confirmatory

A genuinely untouched cohort with documented independence, a representation
frozen before seeing that cohort, a prespecified H20 joint-energy endpoint, and
animal-level resampling. None of these exists in the current data.

## 6. Reconciliations that matter

### 6.1 Joint dynamics versus earlier "no model beats persistence"

`docs/GENFAIL_RESULTS.md` reports no model with positive mean full-neuron MSE
skill at any horizon for its frozen model set; `docs/ADAPTIVE_DYNAMICS_RESULTS.md`
reports that the rolling 500-frame VAR improves H20 MSE over the frozen VAR and
modestly over persistence. These are not the same model or estimand:

- GenFail uses stable/frozen models and full-neuron MSE across its own starts.
- Adaptive dynamics uses a rolling refit and a different evaluation window.
- Joint dynamics uses a probabilistic ensemble, a fixed 10-D PCA, energy and
  trajectory scores, and 175 landmarks.

The joint result does **not** overturn the earlier full-neuron point-forecast
failure. It adds a narrower joint-distribution result in one representation.

### 6.2 Joint dynamics versus persistence

J5 beats persistence on H20 energy (+0.250 [0.020, 0.427]) and trajectory
(+0.226 [0.089, 0.337]), but the block-10 energy interval crosses zero
([−0.032, 0.442]) and only sessions 0, 3 and 4 are positive against persistence.
The population result is more robust against independent AR than against
persistence.

### 6.3 J5 versus J2

The two population-diagonal constructions are statistically and numerically
almost indistinguishable. Do not present J5 as uniquely superior. Present the
mean effect as replicated across two innovation constructions.

### 6.4 Innovation autocorrelation

The protocol gate uses the median **absolute** ACF: ACF1 0.0217, ACF5 0.00886.
The signed medians are ACF1 0.0217 and ACF5 0.00783. Both support the same
decision: a temporal innovation model is not justified. The difference is a
summary convention, not a contradiction.

## 7. Limitations

1. **No independent animals.** Five Kato sessions, repeated recordings, animal
   independence undocumented.
2. **Reused test split.** All test-split claims are exploratory.
3. **Overlapping landmarks.** Effective sample size is smaller than 175.
4. **One fixed representation.** The 10-D PCA is an evaluation choice; all-neuron
   and top-20 results are sensitivity analyses, not independent replications.
5. **Post-hoc model choice.** The corrected model was examined after earlier test
   results had been observed.
6. **Calibration fails.** 50% coverage is too high, covariance is wrong, and
   Student-t/parameter-uncertainty corrections do not fix it.
7. **No biological intervention.** Virtual edits are model-state edits.
8. **Behaviour labels are not independent.** They are decoded from a subset of the
   same neurons.
9. **Calcium observation model.** \(\Delta F/F\) is a slow, nonlinear indicator,
   not membrane potential or spike rate.
10. **External pilot is weak.** Seven sessions, undocumented independence, raw
    file absent, no oxygen-response test.
11. **No target ranking.** Measurement reliability does not support ranking
    neurons or targets.
12. **No animal-budget guarantee.** Phase 12/15 design calculations are
    conditional on models that failed calibration.

## 8. Standing labels

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

Biological validation: `NOT_EVALUABLE`.

Final probabilistic label: `PROBABILISTIC CALIBRATION NOT ESTABLISHED`.
