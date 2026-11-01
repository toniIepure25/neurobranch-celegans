# Reproduction audit — why the earlier simulators failed

Branch: `research/generative-failure-diagnostics`
Base commit: `b48a4847efde8d6c0e63640afc6ff472f3057759`
Audit artifacts: `reports/simulator/failure_diagnostics/{data_audit.csv,split_audit.csv,prior_metric_reproduction.csv,independent_refit.csv,reproduction_audit.json}`

This audit is diagnostic, not a rewrite. It does not alter frozen Phase 6–15 reports, checkpoints or test predictions. The NoStim test tails had been inspected in earlier phases, so they are treated as an already-used exploratory holdout, never as “newly untouched”.

## 1. What was checked

| Area | Check | Result |
|---|---|---|
| Raw data | SHA256, frame counts, fps, finite values, monotonic time | PASS; all five sessions finite, monotonic |
| Splits | train/validation/test bounds, purge, mask counts | PASS; chronological 70/15/15 with 12-frame purge |
| Checkpoints | 25 checkpoint paths recorded by Phase 6, all files present; sampled checkpoints load | PASS |
| Preprocessing | train-only mean/std recomputed from train masks | PASS; no train/test statistic mixing found |
| Targets | `X[t+1]`, macro labels, horizon construction | PASS for the audited paths |
| Rollouts | initialization at the last observed history frame; no future observations | PASS for the audited paths |
| Selection scope | selection on validation, test read once | PASS for audited headline tables |
| Documented metrics | eight headline values re-derived from stored per-session tables | PASS (0 discrepancies at tolerance 1e-3) |
| Independent refit | session 0, H20, latent LDS and stable VAR re-fit from raw data | EXACT reproduction of stored values |
| Spectral stability | realized companion radius recorded | PASS for stable VAR (`0.995`) |
| Schema | context per-start table has no `horizon` column | UNRESOLVED_SCHEMA; not guessed |

## 2. Data facts

All five WT_NoStim sessions are ~1080 s long but have different frame counts and frame rates. Numeric trace IDs are worm-specific. Named-neuron comparisons use alphabetic identity only; they do not establish one shared anatomical mapping across animals.

| worm | frames | predictors | fps | duration (s) |
|---:|---:|---:|---:|---:|
| 0 | 3137 | 103 | 2.9046 | 1079.66 |
| 1 | 3134 | 127 | 2.9019 | 1079.66 |
| 2 | 3059 | 123 | 2.8324 | 1079.65 |
| 3 | 3311 | 118 | 3.0657 | 1079.67 |
| 4 | 3021 | 121 | 2.7972 | 1079.64 |

The original `WT_NoStim.mat` SHA256 is `7531811c1718e26a0455f59cb2f8188cf25809278cfea85770dfe03315a0098a`. The original `WT_Stim.mat` SHA256 is `96ffc39afbc90b66ff1b4c98de199f8424391450d4a8050cbe2b65b3a2fdaa72`. No raw data are committed.

## 3. What reproduced exactly

Session-0 H20, 428 test starts, same histories and targets:

| model | MSE | persistence MSE | skill vs persistence |
|---|---:|---:|---:|
| latent LDS (dim 15) | 1.210570 | 0.407379 | −1.971607 |
| stable VAR (lag 1) | 1.070544 | 0.407379 | −1.627883 |

These match the earlier stored values to six decimals. The failure is therefore not an artifact of a stale table or a changed split.

Eight documented headline values also reproduce:

| key | reported | recomputed | abs diff |
|---|---:|---:|---:|
| context latent H20 mean | −1.6160 | −1.616493 | 0.000493 |
| context latent H20 median | −0.9107 | −0.910728 | 0.000028 |
| context latent H100 mean | −0.7147 | −0.714729 | 0.000029 |
| context latent H100 median | −0.1639 | −0.163852 | 0.000048 |
| stable VAR H20 mean | −1.4813 | −1.481279 | 0.000021 |
| stable VAR H20 median | −0.7699 | −0.769949 | 0.000049 |
| stable VAR H100 mean | −0.6124 | −0.612419 | 0.000019 |
| stable VAR H100 median | −0.0606 | −0.060621 | 0.000021 |

## 4. Caveats found

1. **No independent animal split.** The test set is the chronological tail of each session. It is session-disjoint, not animal-disjoint, and it was previously inspected. The result generalises to later frames of these five animals, not to new animals.
2. **All models use the same persistence target start.** There is no evidence that the failures come from comparing models on different starts; the audit finds the same starts and targets.
3. **Checkpoint availability is better than earlier assumed.** Phase 6 checkpoints exist, but the audited simulator failures are reproduced from the current code and stored tables, not by blindly re-running the checkpoint. Checkpoint provenance is recorded, not inferred.
4. **The oxygen sessions are repeated schedules, not independent perturbations.** They cannot be treated as 84 independent events.
5. **Some historical per-start schemas are incomplete.** The context per-start table lacks a horizon field; this is recorded as `UNRESOLVED_SCHEMA` rather than reconstructed by guessing.

## 5. Verdict

The earlier failures are **real and reproducible**. They are not explained by a simple preprocessing leak, a target shift, a wrong rollout initialization or a changed test table. The remaining explanations must be dynamical/observational, and are tested in the later stages.

Audit status: `PASS` for reproduction and split integrity; `LIMITED` for animal-level generalisation; `UNRESOLVED_SCHEMA` for the one per-start table.
