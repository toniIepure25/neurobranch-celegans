# IC1-v2 Monte Carlo calibration

The [frozen specification](IC1_V2_ESTIMAND_SPEC.md) set 13 synthetic fixtures, two independent estimator modes, 256 replicates per fixture and mode at each of 128, 512 and 8192 trajectories per fitted row. For the occupancy mode the corresponding number of independently drawn start pairs per occupied cell equals the number of trajectories per row. That gives 19,968 evaluator runs. The second pass used the same seeds to check reproducibility and produced identical bytes. [Frozen grid, seed keys, per-cell results and example response counts](IC1_V2_ACCEPTANCE_RESULTS.json)

## Acceptance

All 13 analytic targets and all deterministic diameter estimates matched within 1e−12. All eight prespecified convergence checks passed. There were zero invalid or missing estimates. At 8192, the largest global absolute bias, RMSE and mean interval width across both modes and all cases were **0.008375**, **0.009655** and **0.089776** TV, against thresholds 0.025, 0.03 and 0.12. Simultaneous intervals covered every occupied cell and the global value in 256/256 replicates in each of the 78 fixture × mode × sample-size regimes. The Wilson 95% lower bound of each observed rate is 0.9852. These are conservative intervals. The all-hit result gives no evidence of near-nominal sharpness.

| Fixture and mode | Oracle | Bias at 128 / 512 / 8192 | RMSE at 128 / 512 / 8192 |
|---|---:|---:|---:|
| Ring L1, support | 0.70 | +0.03611 / +0.01873 / +0.00490 | 0.04820 / 0.02398 / 0.00641 |
| Ring L1, weighted | 0.35 | −0.00218 / +0.00098 / −0.00024 | 0.02124 / 0.01099 / 0.00260 |
| Ring asymmetric, support | 0.85 | +0.03712 / +0.02016 / +0.00509 | 0.04206 / 0.02266 / 0.00576 |
| Ring asymmetric, weighted | 0.425 | +0.00070 / +0.00101 / +0.00002 | 0.01791 / 0.00894 / 0.00217 |
| Coincident stochastic null, support | 0 | +0.06848 / +0.03305 / +0.00837 | 0.07643 / 0.03795 / 0.00966 |
| Coincident stochastic null, weighted | 0 | +0.03452 / +0.01729 / +0.00423 | 0.03959 / 0.01954 / 0.00479 |
| Three-category alternative, support | 0.50 | +0.01883 / +0.01096 / +0.00234 | 0.04985 / 0.02685 / 0.00666 |
| Three-category alternative, weighted | 0.25 | +0.01016 / +0.00437 / +0.00139 | 0.03422 / 0.01796 / 0.00441 |

Plug-in TV is a convex absolute-difference functional of histograms, so noise raises its expectation at the stochastic null. Taking a support-wide maximum adds another selection effect. The drop in null bias agrees with the expected concentration of histogram differences as trials increase. The intervals combine bounded event-frequency concentration with the triangle inequality and, for occupancy, concentration over IID start pairs. They cover the true error including start-pair randomness under the declared design. The repeated-run calibration limits apply to these finite synthetic sizes and alphabets. Even at 8192, individual uncertainty can be too broad for subtle scientific contrasts.

The rare-start fixture gives exact diameter 1 and weighted error 1.99999998e−8. The weighted sampler omitted the 1e−8 state in all 256 replicates at each grid point. Its extreme rarity gives a nonobservation probability of about 0.999836 even at 8192 pairs. The support sampler included the state in every replicate and recovered diameter 1. The explicitly trimmed 1e−6 sensitivity excluded probability mass 1/100000000 and returned diameter 0. The weighted estimate cannot stand in for the full-support diameter.

## Execution and reproduction

Local Windows Python 3.13.5 and NumPy 2.3.1. Use the [source fingerprints and runtime metadata](IC1_V2_ACCEPTANCE_RESULTS.json) with the pinned [specification fingerprint](IC1_V2_PROTOCOL_FINGERPRINT.md). The first run took 38.09 seconds, and the replay took 37.00 seconds. Both exited 0. Result SHA-256: `c2b67a20abc3d7876b7854a2b1c1960dc8015d3166d9f88cc9a8168055a91f68`.

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$nbReplay = Join-Path $env:TEMP ('nb-ic1-v2-replay-' + [guid]::NewGuid().ToString('N') + '.json')
python scripts/ic1_v2_acceptance.py $nbReplay
Get-FileHash -Algorithm SHA256 -LiteralPath $nbReplay,research/ic1_v2/IC1_V2_ACCEPTANCE_RESULTS.json
```

The runner requires a new destination and retains a failed calibration as JSON. Each grid cell records checks, errors/quantiles, omissions and an example histogram. Coverage here concerns the instrument on independently simulated paths with known labelled checkpoints. A future learned-model comparison needs its own uncertainty over training and evaluation blocks. Kubernetes was not used.
