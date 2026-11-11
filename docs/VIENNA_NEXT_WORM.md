# Sixth-worm ingestion gate and pre-registration

Status: **ready in code, blocked on data**. The five published worms remain the
only evidence base. This document defines exactly what is needed to add a sixth
worm without changing the evaluation unit or tuning on the new animal.

## 1. Files

Place the released artifacts here:

```text
data/processed/bundlenet_sim/bundlenet_consistent_embedding_worm_5.npz
data/processed/ncmcm/nc_mcm_model_worm_6.npz          # optional, for Tier 2
```

The BunDLe-Net file must contain:

| key | shape | meaning |
|---|---|---|
| `x` | `(T, 3)` float | aligned consistent embedding, same 3-D coordinate system |
| `b` | `(T,)` integer | behaviour label in `0..7`, canonical order below |

Canonical order: `Dorsal turn, Forward, No state, Reverse-1, Reverse-2,
Sustained reversal, Slowing, Ventral turn`.

The NC-MCM file must follow the published six-array layout documented in
`neurobranch.bundlenet_sim.ncmcm`; its labels are mapped to the canonical order
by behaviour name, never by raw integer.

## 2. Gate

Run:

```powershell
python scripts/validate_new_worm.py --worm-id 5
```

The gate fails if the file is absent, the shape is wrong, labels are outside
`0..7`, or the worm cannot be loaded. It prints frame counts, per-behaviour
counts, discovered worm ids, and an NC-MCM frame-count warning when the interval
convention has not been established.

Only after the gate passes:

```powershell
python scripts/validate_new_worm.py --worm-id 5 --run-benchmark
```

This writes a **new** report; it does not overwrite the five-worm report.

## 3. Pre-registration (freeze before looking at results)

- **Unit of analysis:** worm, not frame. Leave-one-worm-out; the sixth worm is a
  held-out fold, never a training worm in its own score.
- **Primary metric:** 10-frame latent MSE and CRPS for `manifold_walk`.
- **Secondary metrics:** 90% interval coverage (raw and nested-calibrated),
  behaviour accuracy/NLL, transition JS at 25 frames.
- **Baselines:** constant velocity, persistence, training-mean; no removing a
  baseline after seeing the new worm.
- **No test-time tuning:** do not change `horizon`, `stride`, seeds, neighbours,
  calibration target, or model family after seeing the sixth-worm score.
- **Interval convention:** establish the frame offset between the neural trace
  and behaviour events before scoring Tier 3; record it, do not search for the
  best offset.
- **Decision rule:** report the result even if it is worse than the five-worm
  mean. A sixth worm is an external-validity test, not a leaderboard entry.

## 4. Why this matters

Five worms give five held-out folds; a sixth adds one independent animal and
makes worm-level uncertainty less dependent on a single split. It does **not**
create an intervention, independent eight-behaviour labels, a digital twin, or a
causal claim. Those remain separate blockers.

## 5. What not to do

- Do not duplicate or resample a published worm and call it a sixth worm.
- Do not impute missing behaviour frames with model predictions.
- Do not align NC-MCM and BunDLe-Net by raw index without checking the release's
  interval/offset convention.
- Do not report a sixth-worm score without the same preprocessing and the same
  frozen evaluation code.
