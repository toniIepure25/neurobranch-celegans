# JOINTDYN reproduction

**Result: G1 PASS.** The historical overdispersed result, the corrected lead-1 population result and persistence reproduce at machine precision for energy and full-neuron CRPS, and at better than 1e-8 relative difference for full-neuron coverage.

## 1. What was reproduced

Three frozen comparators were rerun on all 175 test landmarks with the original split, 100 members, the original seeds and the fixed 10-component training PCA:

1. probabilistic persistence (`P0_persistence` -> `J0_persistence`);
2. corrected population lead-1 bootstrap (`P6_adaptive_lead1` -> `J5_population_lead1_boot`);
3. historical overdispersed population block model (`P6_adaptive_block` -> `P6_historical_block`).

The historical model is retained only as a diagnostic. It admits multi-step forecast errors into the innovation pool and must never be used as a candidate generative model.

## 2. Exact reproduction table

| New | Historical | Metric | Historical | Rerun | Relative difference |
|---|---|---|---:|---:|---:|
| J0 | P0 | H20 energy | 2.950334 | 2.950334 | 0 |
| J0 | P0 | H20 full-neuron CRPS | 0.452406 | 0.452406 | 1.2e-16 |
| J0 | P0 | coverage 50 | 0.493781 | 0.493781 | 0 |
| J0 | P0 | coverage 80 | 0.790664 | 0.790664 | 0 |
| J0 | P0 | coverage 95 | 0.936477 | 0.936477 | 0 |
| J5 | P6 lead-1 | H20 energy | 2.700150 | 2.700150 | 1.6e-16 |
| J5 | P6 lead-1 | H20 full-neuron CRPS | 0.413370 | 0.413370 | 0 |
| J5 | P6 lead-1 | coverage 50 | 0.587781 | 0.587781 | 0 |
| J5 | P6 lead-1 | coverage 80 | 0.848665 | 0.848665 | 0 |
| J5 | P6 lead-1 | coverage 95 | 0.946998 | 0.946998 | 1.2e-16 |
| P6 block | P6 block | H20 energy | 4.608757 | 4.608757 | 5.5e-9 |
| P6 block | P6 block | H20 full-neuron CRPS | 0.649156 | 0.649156 | 2.4e-15 |
| P6 block | P6 block | coverage 50 | 0.909960 | 0.909960 | 0 |
| P6 block | P6 block | coverage 80 | 0.992070 | 0.992070 | 0 |
| P6 block | P6 block | coverage 95 | 0.998903 | 0.998903 | 0 |

The largest relative difference is `5.5e-9`, far below the frozen 2% tolerance.

## 3. Reproduction of the overdispersion

The historical model reproduces its pathological behaviour:

| Quantity | Historical | Rerun |
|---|---:|---:|
| H20 energy | 4.6088 | 4.6088 |
| H20 full-neuron CRPS | 0.6492 | 0.6492 |
| Coverage 50 | 0.9100 | 0.9100 |
| Coverage 80 | 0.9921 | 0.9921 |
| Coverage 95 | 0.9989 | 0.9989 |
| Trajectory energy H20 | 3.2109 | 3.2109 |

The corrected lead-1 pool removes the catastrophic spread:

| Quantity | Historical block | Corrected lead-1 |
|---|---:|---:|
| H20 energy | 4.6088 | 2.7001 |
| H20 full-neuron CRPS | 0.6492 | 0.4134 |
| Coverage 50 | 0.9100 | 0.5878 |
| Coverage 80 | 0.9921 | 0.8487 |
| Coverage 95 | 0.9989 | 0.9470 |
| Trajectory energy H20 | 3.2109 | 2.0600 |

This is a diagnostic correction, not a newly independent confirmation. The corrected model was examined after earlier test results had been observed.

## 4. Discrepancies investigated

An initial comparison appeared to disagree on CRPS and coverage because the new joint evaluation reports per-PCA-component and per-neuron quantities in different tables. The reproduction endpoint is the **full-neuron** quantity, matching the previous report. Once matched:

- full-neuron CRPS is exact;
- full-neuron coverage is exact;
- 10-D PCA energy is exact.

No discrepancy remains after matching the evaluation space.

## 5. Data and split verification

- Five sessions and 103 predictors (one session 102) were confirmed.
- Train/validation/test masks and 12-frame purge gaps were unchanged.
- 175 test landmarks were generated with `evaluation_landmarks(..., horizon_max=100, stride=10)`.
- PCA was fitted on training frames only.
- Seeds were unchanged.
- No raw recordings or large arrays are committed; only derived scores, diagnostics and compressed projected examples.

## 6. Reproduction commands

```powershell
python scripts/jd_reproduce_historical.py
python scripts/jd_evaluate.py --members 100 --stride 10
python scripts/jd_summarize.py
python scripts/validate_jointdyn_reports.py
```

The first command reproduces the historical invalid comparator. The second reproduces the corrected and baseline comparators as part of the full factorial evaluation.

## 7. Artifacts

- `reports/simulator/joint_dynamics/reproduction_check.csv`
- `reports/simulator/joint_dynamics/historical_block_scores.csv`
- `reports/simulator/joint_dynamics/scores_by_landmark.csv`
- `reports/simulator/joint_dynamics/score_summary.csv`
- `reports/simulator/joint_dynamics/gates.json`
