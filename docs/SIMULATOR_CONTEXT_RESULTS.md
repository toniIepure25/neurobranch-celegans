# Results — context-conditioned generative dynamics

All values are held-out test results unless stated otherwise. “Skill” is `1 − MSE / persistence MSE`; positive means better than persistence. Means/medians are across recording sessions, not independent animals.

## 1. Data and power

- WT_NoStim: 5 sessions, strict chronological 70/15/15 split with 12-frame purge.
- WT_Stim: 7 sessions, 4/4/3 oxygen transitions in train/validation/test per session; 21 held-out transition frames total, but one repeated schedule.
- Oxygen audit SHA256: `96ffc39afbc90b66ff1b4c98de199f8424391450d4a8050cbe2b65b3a2fdaa72`.
- NoStim and Stim are session-disjoint at best; they are not five or seven independently identified animals.

## 2. Stage A — stability, support, provenance

Exact companion-matrix capping and verification passed the stability tests. The frozen Phase 6 reports were not modified.

Support audit, Phase 6 per-start maximum distance inside the calibration threshold:

| Family | q90 mean | q95 mean | q99 mean |
|---|---:|---:|---:|
| var | 0.937 | 0.951 | 0.977 |
| stochastic_var | 0.912 | 0.936 | 0.977 |
| switching_var | 0.777 | 0.795 | 0.860 |

These are conservative maximum-distance audits, not full per-step distributions. The Phase 6 official q99 choice is preserved; the switching-VAR q99 coverage is materially below 1.0 and is reported as a support limitation.

Context latent LDS, fraction of starts whose **maximum** support distance is inside the threshold:

| Session | q90 | q95 | q99 |
|---|---:|---:|---:|
| 0 | 0.504 | 0.693 | 0.932 |
| 1 | 0.021 | 0.049 | 0.093 |
| 2 | 0.707 | 0.739 | 0.799 |
| 3 | 0.857 | 0.915 | 0.959 |
| 4 | 0.989 | 1.000 | 1.000 |

Session 1 is a clear support failure even though its mean per-step q99 fraction is high. Support is not forecast accuracy.

Dataset and artifact manifests were generated; the raw and canonical-text hashes are separate records. A manifest self-consistency check is recorded in `artifact_manifest.json` and `support_audit.json`.

## 3. Stage B — latent LDS and stable VAR (task C, autonomous)

Selected: latent dimension 15, ridge 0.01, cap 0.995; stable VAR lag 1, ridge 1.0.

| Model | Horizon | Mean skill | Median skill |
|---|---:|---:|---:|
| latent LDS | 20 | −1.616 | −0.911 |
| latent LDS | 100 | −0.715 | −0.164 |
| stable VAR | 20 | −1.481 | −0.770 |
| stable VAR | 100 | −0.612 | −0.061 |
| latent one-step | 1 | −2.408 | −2.428 |

Per-session latent H20 skills were −1.972, −4.704, −0.378, −0.911, −0.118. Only session 2 at H100 was positive (+0.244, 95% CI [0.186, 0.303]).

**Conclusion:** the autonomous-rollout gate fails. The corrected stable VAR is slightly less bad than the latent LDS but still fails. One positive cell out of ten is not a replicated generative result.

## 4. Stage C — compositional operators (tasks C/D diagnostics)

Validation selected **2 operators** (median H20 skill −0.577), ahead of 1 (−0.594) and 3 (−0.646). Operators were stable (maximum realized radii 0.991–0.995). Coefficients were not literally uniform, but their mean entropy was only 0.25–0.28 nats against a maximum of 0.693, and the causal coefficient policy RMSE was 0.375–0.393.

Test results for the selected 2-operator model:

| Horizon | Mean skill | Median skill |
|---|---:|---:|
| 20 | −1.771 | −1.087 |
| 100 | −0.808 | −0.343 |

Per-session H20 skills: −2.070, −4.853, −0.477, −1.087, −0.366. All five are negative and all are worse than the selected latent LDS at H20 (mean −1.616). At H100, only session 2 is positive (+0.181), and the mean is worse than the latent model (−0.808 vs −0.715).

Coefficient trajectories are causal encoded-state coefficients, not an oracle path. **Conclusion:** the compositional model does not add held-out value and shows coefficient-policy collapse. G6 = FAIL. This is not evidence for or against the biological existence of compositional dynamics; it is a failure of this estimator on this data.

## 5. Stage D — oxygen input (task D, known schedule)

All seven sessions, H20, 40 evenly spaced held-out starts per session. Oxygen is measured environmental input, not targeted neuronal stimulation.

| Model | Mean skill | Median skill |
|---|---:|---:|
| A no input | −0.998 | −0.687 |
| B measured O2 | −1.010 | −0.702 |
| C compositional O2 | −1.210 | −0.884 |
| D lagged O2 (12 frames) | −1.010 | −0.711 |
| D time since transition | −0.996 | −0.691 |
| D time index | −1.562 | −1.140 |
| E stimulus-only | −1.679 | −1.422 |

Measured O2 did **not** improve on no input: B − A mean paired MSE difference was +0.0081 (positive = worse), and B was better in only 1 of 7 sessions. The compositional O2 model was worse than B in all 7 sessions (mean paired MSE difference +0.155). A non-specific time-since-transition control was numerically similar to O2, which is exactly the confound the control was designed to expose.

**Conclusion:** no evidence that measured oxygen conditioning improves free-running H20 prediction in this low-power design. Task D forecasts are explicitly known-schedule forecasts, not autonomous forecasts.

## 6. Stage E — shared dynamics

Leave-one-session-out on the common alphabetic named panel; test skill vs persistence:

| Model | Horizon | Mean skill | Median skill |
|---|---:|---:|---:|
| shared zero-shot | 20 | −2.822 | −1.598 |
| shared zero-shot | 100 | −0.980 | −0.341 |
| shared calibrated | 20 | −5.174 | −2.451 |
| shared calibrated | 100 | −7.068 | −7.860 |
| local, same calibration prefix | 20 | −3.818 | −1.972 |
| local, same calibration prefix | 100 | −1.618 | −0.570 |
| local, full training reference | 20 | −1.924 | −0.857 |
| local, full training reference | 100 | −0.515 | −0.139 |

Zero-shot shared dynamics is less bad than the local model fitted on the same short calibration prefix, but it is still far worse than persistence at H20 and is not better than the local full-training reference. Calibration adaptation made things dramatically worse (H20 −5.17; H100 −7.07), and several calibrated H100 rollouts left the q95 support envelope (session 0: 0.264; session 1: 0.211; session 2: 0.132).

**Conclusion:** no evidence for useful shared latent dynamics. The result is session-disjoint, not animal-independent. The shared coordinate is not a biological common coordinate.

## 7. Stage F — branching

Four of five sessions had evaluable reversal states (session 1 had no REV candidate frames). Across evaluable sessions:

| Comparator | Mean total variation to observed local outcome |
|---|---:|
| fitted stochastic latent LDS | 0.580 |
| persistence | 0.548 |
| marginal first-exit | 0.317 |
| first-order Markov | 0.360 |

The latent stochastic ensemble was worse than persistence and much worse than the marginal/Markov comparators. Its mean simulated entropy (0.520) was below observed local entropy (0.652). The process covariance was finite and small (trace 5.3–10.1), so this is not an arbitrary noise injection; it is a misspecified fitted-uncertainty model.

**Conclusion:** the model does not reproduce observed local branching statistics. No biological bifurcation claim is made.

## 8. Stage G — bounded perturbations

All six requested targets were verified present in all five recordings. There were 600 target/control events (5 frames × 6 targets × 2 protocols × 2 directions × 5 sessions; each event has a target and a same-amplitude control).

Largest mean excess final shift over the matched control:

| Target / protocol | Mean target shift | Mean control shift | Mean excess |
|---|---:|---:|---:|
| AVBL dynamical | 0.270 | 0.195 | 0.075 |
| AVBR dynamical | 0.205 | 0.150 | 0.055 |
| RIVL dynamical | 0.199 | 0.146 | 0.053 |
| RIVR dynamical | 0.268 | 0.234 | 0.034 |
| AVBL initial | 0.054 | 0.037 | 0.016 |
| AVBR initial | 0.040 | 0.029 | 0.011 |

Positive and negative amplitudes produce identical shift norms because the fitted model is linear; direction is not an independent contrast. Support q95 was 0.981 for all target events. These are model sensitivities, not experimental effects, and the matched random-neuron control also moves the model. Gate 7 is NOT_EVALUABLE and Gate 8 is NOT_EVALUABLE.

## 9. Gates

| Gate | Status |
|---|---|
| G1 data integrity | PASS after manifests |
| G2 stability | PASS |
| G3 causality/leakage | PASS |
| G4 support audit | CONDITIONAL |
| G5 autonomous rollout | FAIL |
| G6 compositionality | FAIL |
| G7 perturbation validation | NOT_EVALUABLE |
| G8 intervention compatibility | NOT_EVALUABLE |

## 10. Artifact and test status

- Support audit: `phase6_support_event_audit.csv`, `phase6_support_summary.csv`, `context_latent_support_event_audit.csv`, `context_support_thresholds.csv`.
- Dataset/artifact manifests: `dataset_manifest.json`, `artifact_manifest.json`, `reproducibility_manifest.json`. The artifact manifest contains 47 records; all 47 verified against their declared mode. 37 text artifacts have separate raw and canonical-text hashes (only one happened to be byte-identical), and 10 binary artifacts (5 PNG figures and 5 `.npz` trajectory files) are raw-only.
- Figures: `figures/autonomous_skill.png`, `oxygen_models.png`, `branching_tv.png`, `support_audit.png`, `operator_coefficients.png`, and the interactive `figures/latent_rollout_3d.html`.
- Unit/data/simulator suite: **116 passed** in 19.39 s.
- Ruff: all checked files passed.
- Streamlit smoke test: no exceptions; all 16 top-level tabs and all 5 context sub-tabs rendered.
- Phase 6 frozen reports were not modified.

