# Simulator research protocol (frozen before fitting)

**Status:** frozen 2026-10-31, before simulator model fitting or test
evaluation.
**Branch:** `research/core-dynamics-simulator`
**Base commit:** `e18139b15048b31bc384cab1cb4591b09b80ab34`
**Canonical copy:** `research/simulator_protocol.md`

This document fixes the estimands, splits, model-selection rule, gates,
metrics, seeds and decision labels. Changes after this point are recorded in
section 10 with a date, reason, and whether they were made before or after
inspecting test results.

---

## 1. Unit of analysis and data

- Dataset: Kato et al. (2015) whole-brain calcium imaging, wild-type,
  no-stimulation recordings.
- Animals: five. All summaries are per worm, then described across worms. Five
  animals are not a population sample.
- Predictors: bleaching-corrected \(\Delta F/F\) traces that pass the existing
  preprocessing and identity filters.
- Excluded: `AVAL`, `AVAR`, `SMDDR`, `SMDDL`, `SMDVR`, `SMDVL`, `RIBR`, `RIBL`,
  because they define the behavioural labels.
- Standardisation: training-split mean and standard deviation only.
- Frame interval: 0.326–0.357 s; no resampling.
- State dimension: 103–127, worm-dependent.
- Behaviour: four macro labels (`FWD`, `REV`, `TURN`, `NOSTATE`) inferred from
  AVA/SMD/RIB activity.

The simulator models standardised calcium dynamics, not membrane potential,
spiking, muscles, or the whole 302-neuron nervous system.

---

## 2. Splits

### Primary: blocked interpolation

`configs/experiments/blocked_suite.yaml`:

- contiguous blocks;
- test block 2;
- validation block 4;
- one-frame history for persistence/VAR; model-specific history elsewhere;
- purge frames respected.

An initial state is eligible only if every real frame in the evaluation horizon
lies inside the same held-out block. This prevents a rollout from being scored
against a frame that was available during training.

### Stress: strict chronological

`configs/experiments/forward_stress.yaml`, reported separately. It is not used
to select the model and cannot rescue a failed blocked result.

### Leakage rules

- Every scaler, PCA basis, BunDLe encoder, behaviour probe and kNN support set
  is fitted on training frames only.
- No test frame enters early stopping, model selection, calibration,
  perturbation design or figure selection.
- Model-class selection uses validation block 4 only.
- Test block 2 is evaluated once per frozen model class and seed.

---

## 3. Simulator estimand

For worm \(i\), initial history \(h_t\), and optional input \(u\):

\[
\hat{x}_{t+1:t+H} \sim P_\theta(\cdot \mid h_t, u),
\]

generated recursively:

\[
\hat{x}_{t+k+1} = F_\theta(\hat{x}_{t+k-h+1:t+k}, u_{t+k}) + \varepsilon_{t+k}.
\]

At no point may \(F_\theta\) receive \(x_{t+k}\), \(k \ge 1\). Teacher-forced
one-step prediction is reported separately and is never called a rollout.

Observation model: identity on standardised \(\Delta F/F\) plus optional
Gaussian or model-native noise. No deconvolution or spike inference.

Behaviour readout: existing multinomial probe \(g_\psi(\hat{x}_t)\), trained on
training frames only, with label-defining neurons excluded. Behaviour results
are conditional on the Kato label definition.

---

## 4. Fixed model set and selection rule

No hyperparameter search on test. The candidate set is fixed:

1. persistence;
2. ridge VAR(1);
3. MLP one-step;
4. MLP residual;
5. GRU;
6. PCA-latent MLP;
7. PCA-latent kNN.

The existing fitted configurations and seeds are used. If a checkpoint is
missing, the model is fitted with the frozen configuration and three seeds
(`17`, `23`, `31`).

Selection:

- Rank candidates on validation block 4 by mean one-step \(R^2\), then
  five-step MSE (lower is better), then model simplicity.
- Persistence is in the selection set. If persistence wins validation, no
  learned model is promoted and Gate 2 fails.
- Freeze the selected class before touching test block 2.
- Report every candidate, not only the winner.

A multi-step-trained MLP is an exploratory challenger and cannot replace the
frozen selection. No Transformer is introduced.

---

## 5. Gate definitions

### G1 — functioning simulator

Pass only if all hold:

- all five worms produce a 20-step rollout from a held-out state;
- the rollout API has no access to future observations;
- at least 95% of mandatory rollouts are finite and within the documented
  numerical range;
- the first state equals the supplied initial state exactly;
- the same seed and inputs reproduce the same output bit-for-bit within a
  session;
- stochastic ensembles differ when the seed changes;
- the trajectory and manifest are saved.

Failure blocks any simulator claim.

### G2 — dynamical validity

For each worm, paired over eligible test initial states, compare the
validation-selected learned model with persistence.

Pass only if:

- \(\Delta R^2_1 = R^2_{\text{model}} - R^2_{\text{persistence}} \ge 0.02\);
- the 95% moving-block bootstrap interval for \(\Delta R^2_1\) excludes zero;
- \(\mathrm{MSE}_5 \le \mathrm{MSE}_{5,\text{persistence}}\);
- both conditions hold in at least 4/5 worms;
- distributional error is not worse than persistence by more than 10% on the
  pre-specified composite (per-neuron mean/SD, autocorrelation, cross-neuron
  correlation).

Failure label: `functional_simulator_no_dynamical_advantage`.

### G3 — branching validity

Evaluation states: held-out frames whose macro state is `REV`, preferring
sustained reversal (`revsus`) where available, with a 20-frame future inside
the held-out block.

For each state:

- observed local distribution: kNN outcomes among behaviourally matched
  held-out states, excluding temporal neighbours;
- simulated distribution: at least 100 stochastic rollouts;
- outcome: first macro state different from `REV` within 20 frames.

Pass only if:

- mean total variation to the observed local distribution is at least 0.02
  lower than both persistence and the marginal-outcome null;
- the paired bootstrap interval excludes zero in at least 3/5 worms;
- simulated outcome entropy lies inside the observed 95% interval;
- no null run produces a larger apparent branch effect than the model.

Failure label: `branching_not_supported`. The word “bifurcation” is prohibited
without a parameter-dependent vector-field analysis.

### G4 — named-neuron perturbation interface

Pre-registered targets: `AVBL`, `AVBR`, `AVEL`, `AVER`, `RIVL`, `RIVR`.

Pass only if:

- each target maps to a predictor index in all five worms;
- initial-state, dynamical-input and empirical-response protocols are distinct
  in code and documentation;
- at least 100 paired baseline/perturbed rollouts are reproducible;
- OOD fraction, model seed and support distance are reported;
- the empirical-response path refuses to run without matched evidence;
- every output carries the standing warning.

G4 is a capability gate. Passing it does not establish a biological effect.

### G5 — robust sensitivity

Pass only if one pre-registered candidate has:

- the same sign of \(\Delta P(\mathrm{TURN})\) across at least 4/5 worms;
- the same sign across at least 2/3 model seeds;
- the same sign across at least 2/3 perturbation amplitudes or timings;
- an absolute effect above the 95th percentile of matched control neurons in
  at least 4/5 worms;
- a max-statistic correction over the six candidates that still excludes zero.

Failure label: `model_dependent_only`.

### G6 — biological validation

Pass only with matched held-out actual interventions in the same preparation:

- same neuron identities;
- same stimulation modality and dose range;
- sham and random-target controls;
- at least three independent animals;
- repeated trials;
- an independent behavioural endpoint.

Kato contains no such intervention. Randi is a different preparation and
protocol. Gate 6 status is `not_evaluable`, not “failed” and not “passed.”

---

## 6. Metrics

All metrics are computed per worm, then summarised. Report mean, median,
range, number of eligible states and number of stochastic rollouts.

### Prediction

- one-step MSE, \(R^2\), Pearson correlation;
- free-running MSE/correlation at horizons 1, 3, 5, 10, 20;
- persistence-relative deltas and win counts.

### Distribution

- per-neuron mean and SD error;
- pooled activity quantile error;
- energy distance or RBF-MMD between simulated and real state samples;
- autocorrelation error at lags 1, 2, 5, 10;
- cross-neuron correlation Frobenius error;
- PCA eigenvalue-spectrum error.

### State dynamics

- macro-state occupancy total variation;
- transition-matrix total variation and Jensen–Shannon divergence;
- outcome entropy;
- recurrence rate within a fixed radius.

### Stability and support

- finite fraction;
- variance ratio simulated/real;
- saturation fraction at \(\pm 5\) standardised units;
- OOD fraction using training kNN distance;
- divergence flags.

### Latent

- real vs simulated MMD/energy distance in PCA and BunDLe spaces;
- path length and curvature;
- \(k\)-NN neighbour overlap;
- CKA/Procrustes encoder stability across seeds;
- OOD latent fraction.

### Perturbation

- paired \(\Delta\) trajectory norm;
- paired \(\Delta\) latent trajectory;
- \(\Delta P(\mathrm{TURN})\);
- transition-distribution TV;
- OOD fraction;
- matched-control percentile and rank;
- amplitude/timing/model-seed sensitivity.

---

## 7. Uncertainty

- Seeds: model seeds `17`, `23`, `31`; rollout seeds derived deterministically
  from `(worm, model seed, initial state, condition, rollout index)`.
- Bootstrap: moving-block bootstrap over initial frames, 2,000 replicates,
  block length 10; worm-level paired deltas.
- For candidate neurons: max-statistic over the six pre-registered targets and
  matched controls.
- Report intervals, not only point estimates. Do not treat five worm-level
  estimates as a confidence interval for the species.

---

## 8. Model cards and manifests

Each fitted model card records:

- model kind, hyperparameters, history length, noise model;
- worm id, train/validation/test bounds, purge;
- train seed and package versions;
- data file SHA256 and preprocessing hash;
- checkpoint SHA256 and path;
- git revision and dirty state;
- training command and runtime;
- validation and test metrics.

`reports/simulator/manifest.json` records every artifact hash, the raw-data
hash, protocol hash, git revision, software versions, and the exact commands.

---

## 9. Decision labels

- `simulator_released`: G1 and G4 pass, manifests and tests pass; scientific
  gate outcomes still reported separately.
- `functional_simulator_only`: G1/G4 pass, not both G2 and G3 pass.
- `no_dynamical_advantage`: G2 fails.
- `branching_not_supported`: G3 fails.
- `model_dependent_only`: G5 fails.
- `not_evaluable`: required evidence is absent (always G6 for Kato).

No combination of labels permits “causal,” “validated stimulation,”
“whole-brain,” or “bifurcation.”

---

## 10. Deviations

| Date | Deviation | Reason | Test results seen? |
|---|---|---|---|
| 2026-10-31 | Branch based on `e18139b`, not the reported `cd5b6f4` | `cd5b6f4` is 59 commits behind and lacks Phase 15; branching from it would discard later audit work | No |
| 2026-10-31 | Existing Phase 2 checkpoints may be reused if metadata and split match | Avoid retraining equivalent models; cards record provenance | No |

Any further deviation is appended before the affected analysis is run.
