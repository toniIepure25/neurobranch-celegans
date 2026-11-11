# Tier 1 interval calibration — design and result record

Status: implemented in benchmark schema v3. This is a **marginal diagnostic**, not a posterior
calibration claim.

## Pre-run hypothesis

Raw ensemble 90% intervals under-cover because they are finite-seed quantiles around a fitted
median, not a predictive distribution. Widening them by a scalar estimated on held-out training
worms should move empirical coverage toward nominal without touching the test worm.

## Design

- Target: 90% marginal coverage over starts and latent dimensions; 50% is reported as a secondary
  diagnostic.
- For each outer test worm, use the four training worms only.
- Nested loop: fit all three models on three training worms, score the fourth, pool the normalised
  absolute residual `|true − median| / halfwidth`.
- Scale factor: `max(1, quantile(residual_ratio, target))`; no shrinking.
- Apply the scale to the outer test intervals; report raw and calibrated coverage side by side.
- No test frame, test start or test label enters the scale estimate.

## Decision criterion

A calibration run is useful if it improves held-out coverage without changing point forecasts. It
is **not** called calibrated unless coverage is close to nominal with uncertainty reported. If it
remains below nominal, the UI must say so and keep the bands descriptive.

## Result

| model | raw 90% | calibrated 90% | scale × |
|---|---:|---:|---:|
| manifold walk | 0.707 | 0.854 | 1.67 |
| quadratic | 0.681 | 0.870 | 2.08 |
| switching | 0.752 | 0.801 | 1.16 |

Calibration improves all three but remains below 90%. Under-coverage is worm-dependent and the
conformal exchangeability assumption is not satisfied by five distinct animals. The result is
reported as a partial repair, not a guarantee.

## Implementation

`src/neurobranch/bundlenet_sim/benchmark.py`:

- `_fit_bundle` — shared model fitting and train-residual noise.
- `_calibration_scales` — nested leave-one-training-worm-out residual pooling.
- `_score_predictions` — raw and scaled coverage, with scale fields.
- `benchmark_models` — schema v3, calibration block, per-fold scale fields.

Run:

```powershell
python -m neurobranch.bundlenet_sim.benchmark `
  --out reports/vienna_workbench/tier1_benchmark.json
```

Approximate runtime: 2 minutes locally for 5 folds, 25-frame horizon, 16 seeds, including 20 nested
calibration fits.
