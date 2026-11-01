# Final report — context-conditioned generative dynamics

Branch: `research/context-conditioned-generative-dynamics`
Base commit: `4da6c24236d2c6399f5f1b859a21ae479e3521cc`
Commit hashes for the work are reported in the final response and in `git log` (a file cannot contain the hash of the commit that creates it).

## 1. What was the question, and which task was measured?

Whether compositional, stimulus-aware, low-dimensional dynamics can produce a more faithful generative simulation of *C. elegans* population activity. Four tasks were kept separate: A retrospective reconstruction, B causal latent filtering, C autonomous rollout, D stimulus/schedule-conditioned rollout. All headline negatives below are task C unless explicitly labelled D. No reconstruction result is presented as a forecast.

## 2. What data and splits were used?

Kato 2015 WT_NoStim (5 sessions) for Stages B, C, E, F, G and Phase 6 support comparison; Kato 2015 WT_Stim (7 sessions) for Stage D. NoStim used a chronological 70/15/15 split with 12-frame purge. Stim used 4/4/3 oxygen transitions per session. The test tails are session-disjoint, not animal-disjoint; 21 held-out Stim transitions are one repeated schedule. Low power is explicit.

## 3. What was wrong with the old stability claim, and was it fixed exactly?

The old path scaled the VAR coefficient matrix once, which does not guarantee the actual companion-matrix radius. The new path builds the companion matrix, iteratively scales only the coefficient block, preserves the identity block, and verifies the realized radius. `StableVARModel` records requested cap, realized radius, iterations, and scale. Tests cover the cap and realized radius. This does not retroactively alter frozen Phase 6 checkpoints.

## 4. What did the q90/q95/q99 support audit find?

Phase 6 official q99 was preserved. Mean fraction of starts with maximum distance inside q99: VAR 0.977, stochastic VAR 0.977, switching VAR 0.860. At q90/q95 the switching family is 0.777/0.795. Context latent session 1 is much worse: only 0.093 of starts have maximum distance inside q99 despite a high per-step fraction. The audit is conservative because Phase 6 stored only per-start maxima.

## 5. How was provenance separated?

Raw-byte SHA256 and canonical-text SHA256 (UTF-8, CRLF/CR → LF) are separate fields with declared normalization. 37 text artifacts have both hashes; only one was byte-identical. 10 binary artifacts (5 PNG, 5 NPZ) are raw-only. All 47 manifest records verified. Frozen Phase 6 reports were not rewritten.

## 6. How were latent models selected?

Train-only PCA; ridge latent transition; exact cap; dimensions 5/10/15; causal filter; selection on validation H20 skill across five temporal holdouts. Selected: dimension 15, ridge 0.01, cap 0.995. Test remained untouched until selection. Stable VAR selected lag 1, ridge 1.0.

## 7. Did the latent LDS beat persistence?

No. Median H20 skill −0.911 (mean −1.616); median H100 −0.164 (mean −0.715). One-step mean skill −2.408. Per-session H20: −1.972, −4.704, −0.378, −0.911, −0.118. Only session 2 at H100 was positive (+0.244, 95% CI [0.186, 0.303]). Gate G5 fails.

## 8. Did the corrected stable VAR improve on the latent LDS?

Only marginally and not enough. Stable VAR median H20 −0.770, H100 −0.061; still below persistence. A less-negative model is not a validated simulator.

## 9. Was dLDS reproduced?

No. The published mathematical form was reimplemented independently because the official Python repository had no LICENSE file. It is not a reproduction of Yezerets et al. (2025), and no official code was copied. The implementation uses causal coefficients, stable operators, and no oracle future coefficients.

## 10. Did compositional operators add held-out value?

No. Validation selected 2 operators (median H20 −0.577). Test H20 mean −1.771, median −1.087; all five sessions were worse than the latent LDS. H100 mean −0.808, median −0.343. Coefficient entropy was 0.25–0.28 nats against a 0.693 maximum, and policy RMSE was 0.375–0.393: coefficient/operator-policy collapse. Gate G6 fails.

## 11. Was there any future-observation or oracle-coefficient leakage?

No. PCA is train-only; the filter prefix test requires exact equality between a short prefix and the same prefix of a longer pass; the observation covariance uses training-only initialization; the compositional coefficient policy receives only the current encoded state and current known input; oxygen forecasts receive only the known schedule for the scored window.

## 12. What did the oxygen audit show?

Seven sessions, 12 metadata transitions each, 4%/21% levels, first 360 s at 21%. Actual transition frame times vary (e.g. 360.000 vs 360.178 s); exact 30 s alignment is not claimed. Default split has 4/4/3 transitions in train/val/test in session 0.

## 13. Did measured oxygen improve prediction?

No. H20 mean skill: A no input −0.998; B measured O2 −1.010; C compositional O2 −1.210; D lagged O2 −1.010; D time-since-transition −0.996; D time index −1.562; E stimulus-only −1.679. B−A paired MSE difference +0.008 (worse); B better in 1/7 sessions; C worse in 7/7. These are known-schedule forecasts, not autonomous rollouts. Oxygen is environmental sensory input, not targeted neuronal stimulation.

## 14. Did shared latent dynamics transfer across sessions?

No. Leave-one-session-out H20 mean skill: shared zero-shot −2.822; shared calibrated −5.174; local same-calibration −3.818; local full-training reference −1.924. H100: −0.980, −7.068, −1.618, −0.515 respectively. Calibration adaptation was harmful and some rollouts left the q95 envelope. Session-disjoint, not animal-disjoint.

## 15. Did branching reproduce observed local futures?

No. Four of five sessions were evaluable. Mean TV to observed local outcomes: fitted stochastic latent 0.580; persistence 0.548; marginal 0.317; Markov 0.360. Mean simulated entropy 0.520 vs observed 0.652. This is fitted process covariance, not arbitrary injected noise, but it is misspecified. No biological bifurcation claim.

## 16. Are perturbations causal or dose-equivalent, and what are Gates 7/8?

No. All six requested targets were present in all five recordings; 600 events were run with matched same-amplitude controls. AVBL dynamical input had the largest mean excess (0.075), but every target and control moved the linear model, and ± amplitudes produced identical shift norms. These are model sensitivities. Gate 7 = NOT_EVALUABLE (no matched experimental intervention); Gate 8 = NOT_EVALUABLE (no compatible intervention dataset).

## 17. What does the dashboard show, and are labels safe?

A “CONTEXT-AWARE NEURAL SIMULATOR” tab with Overview, Latent model, Oxygen context, Branching, Perturbations. All prior tabs are preserved. The view distinguishes measured oxygen, hypothetical oxygen, mathematical neuron perturbation, modelled input, and actual stimulation data; it shows NOT_EVALUABLE and missing reports. AppTest renders all 16 top-level tabs and 5 context sub-tabs without exception.

## 18. Tests, reproduction, gates, and limits

- Full suite: 310 passed (after updating the two tab-count regression tests); 116 unit/data/simulator tests passed separately earlier.
- Ruff: all changed files pass; `git diff --check` clean.
- MkDocs strict build: passed.
- Reproduce: `python scripts/context_support_audit.py`, `context_latent_baseline.py`, `context_compositional.py`, `context_oxygen_audit.py`, `context_oxygen_models.py`, `context_shared_dynamics.py`, `context_branching.py`, `context_perturbations.py`, `context_figures.py`, `context_manifests.py`.
- Gates: G1 PASS, G2 PASS, G3 PASS, G4 CONDITIONAL, G5 FAIL, G6 FAIL, G7 NOT_EVALUABLE, G8 NOT_EVALUABLE.
- Main limitations: sessions are not animals; oxygen schedule is repeated and collinear with time-since-transition; linear models; PCA shared basis is not identifiable; support distance is not a likelihood; no intervention data. The strongest defensible output is the corrected methodology plus a clearly reported negative result, not a generative simulator.
