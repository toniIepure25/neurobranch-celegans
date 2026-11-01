# GENFAIL research protocol

**Branch:** `research/generative-failure-diagnostics`  
**Base commit:** `b48a4847efde8d6c0e63640afc6ff472f3057759`  
**Status:** frozen before any new test evaluation. Deviations are logged in `GENFAIL_REPRODUCTION_AUDIT.md`.

## 1. Question

Why do the existing NeuroBranch generative simulators fail to beat simple persistence, and what kind of information or model structure is missing?

This is a diagnostic study, not an architecture search. It distinguishes:

- **H1 limited forecastability** — the recorded calcium signal has little autonomous multistep information;
- **H2 observation-model mismatch** — fluorescence is a slow, filtered observation of a faster latent process;
- **H3 missing state variables** — the recorded population is incomplete;
- **H4 inadequate identification** — the estimator cannot recover structure that is present;
- **H5 evaluation/data mismatch** — preprocessing, split, timing, or metric choices create apparent failure;
- **H6 structural non-identifiability** — observationally similar models imply incompatible perturbations.

## 2. Evidence status and splits

- Kato 2015 WT_NoStim: 5 sessions. The temporal test tails have already been inspected in earlier phases. Every new NoStim analysis is **exploratory**, not a newly untouched confirmation.
- Kato 2015 WT_Stim: 7 sessions, repeated oxygen schedule. Test transitions are a chronological tail, not independent animals or trials.
- Sessions are the experimental unit. Repeated time-series windows are dependent; confidence intervals use a moving-block bootstrap or session-level resampling where appropriate.
- No animal-disjoint claim is possible from these files alone.
- No test data may influence feature fitting, hyperparameter selection, threshold choice, or which result is presented.

## 3. Tasks kept separate

- **A** retrospective reconstruction;
- **B** causal filtering;
- **C** autonomous rollout;
- **D** known-schedule-conditioned rollout.

No A result is reported as C or D. Oxygen models are D: they receive the measured schedule for the scored window. They are not forecasts of a future stimulus and not autonomous rollouts.

## 4. Primary experiments and decision criteria

### P1 — Forecastability map

Use identical held-out frames and horizons for all baselines:

1. persistence;
2. train-only mean;
3. local causal linear trend;
4. per-neuron ridge autoregression;
5. population VAR;
6. frozen-family reference models refitted on the same training split (stable VAR and latent LDS).

Metrics: MSE, skill vs persistence, per-neuron and per-session curves, error growth, autocorrelation, spectral power, noise sensitivity, and period stability. Horizons are reported in frames and seconds.

**F1 PASS** if the map identifies, with uncertainty, which horizons/components are predictable and whether population models add value beyond independent per-neuron models. A negative result still passes F1 if it is measured correctly.

### P2 — Observation model

1. Synthetic systems with known latent dynamics: stable linear, stochastic, oscillatory, partially observed, and calcium-filtered with first-order and slower kernels.
2. Test recovery of one-step dynamics, autonomous dynamics, latent transition distributions, and known mathematical perturbations.
3. One transparent latent second-order / calcium-memory state-space model on real data, fitted only on training frames, with a causal filter.
4. Compare on identical test starts against persistence, stable VAR, and the AR(1) latent LDS.

**F2 PASS** if the observation-aware model is evaluated under valid conditions, regardless of whether it improves. It is **FAIL** if synthetic recovery is not established before the real-data claim.

### P3 — Dynamical identifiability

Diagnostics: effective dimensionality, latent observability, transition eigenvalue spectra, cross-seed/block subspace stability, operator usage, operator redundancy, coefficient collapse, short- vs long-horizon reconstruction, true autonomous forecast, simulated vs observed recurrence, and synthetic ground-truth recovery.

Account for rotation, sign, and permutation ambiguity. Do not infer connectivity from a statistical operator.

**F3 PASS** if the study quantifies stability or ambiguity. It does not require the model to be identifiable.

### P4 — Oxygen response

Confirm actual oxygen levels, transition times, direction, session identity, and sampling interval. Quantify time-schedule confounding. Use causal pre-event baselines. Compare no-response, time-only, event-kernel, time+event, and persistence on held-out transition events and leave-one-session-out tests. Report amplitude, timing, adaptation, within/across-session reproducibility, and sensory-neuron availability.

**F4 PASS** if the response and the time confound are assessed with valid uncertainty. A null response is a valid PASS; an unadjusted event effect is **FAIL**.

### P5 — Perturbation identifiability

Use a small fixed set of states and named neurons. Compare stable VAR, latent LDS, compositional model, and the observation-aware model under the same observation-space perturbation. Report cross-model response disagreement, seed stability, and empirical support. Include a synthetic known-truth recovery check.

**F5 PASS** if cross-model disagreement is characterized. **F6 biological intervention validation remains NOT_EVALUABLE** without matched experimental data.

## 5. Metrics and uncertainty

- Primary forecast metric: skill = `1 − MSE_model / MSE_persistence`, computed on identical starts.
- Secondary: R², correlation, energy distance, autocorrelation error, covariance error, support fraction.
- Uncertainty: moving-block bootstrap over held-out start indices (block length 20 frames) and session-level summaries. Intervals are descriptive, not population CIs over animals.
- Statistical tests are exploratory and reported with effect sizes; no multiplicity-adjusted discovery claim.
- Any sampling reduction is recorded in the output metadata.

## 6. Synthetic controls

Synthetic systems are generated from known stable dynamics. Latent recovery is compared after Procrustes alignment up to rotation/sign/permutation. True perturbation responses are computed from the generating model, not from a fitted model.

Synthetic success does **not** validate biological intervention.

## 7. Stop rules

- If persistence is not beaten in a properly evaluated setting, do not add model complexity.
- If the event design cannot separate oxygen from elapsed time, report that and stop.
- If operators are non-identifiable, do not assign biological meaning.
- If cross-model perturbation predictions disagree without a validated observation model, report NOT_EVALUABLE rather than selecting a target.

## 8. Artifacts

- Reports: `reports/simulator/failure_diagnostics/`
- Docs: `docs/GENFAIL_*.md`
- Machine-readable gates: `reports/simulator/failure_diagnostics/gates.json`
- No raw datasets or large checkpoints in Git.

## 9. Deviations logged after execution

- The synthetic observation suite contains stable and oscillatory conditions, with and without calcium filtering. A separate partially observed condition was **not** added; this is a narrower test than the original P2 wording.
- Cross-seed stability is implemented as independent 80% start subsamples, not different random initialisations, because the fitted linear models have no stochastic initialisation.
- The oxygen schedule jitter is sub-second; the design remains a repeated fixed schedule and is not randomized.
