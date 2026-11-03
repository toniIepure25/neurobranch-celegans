# Kato final audit: clean end-to-end reproduction

## Status

```text
FULL_END_TO_END_REPRODUCTION_PASSED
```

This status is based on a complete, independently executed run of the official
entry point, not on unit tests, manifest verification, post-processing or a
resumed run.

## 1. Frozen starting state

| Item | Value |
| --- | --- |
| Branch before audit | `research/kato-common-state-controls` |
| Starting HEAD | `358107a7a55470d48f8092b2a2dc7f704dbd37c1` |
| Starting worktree | clean |
| Audit branch | `research/kato-final-reproducibility-audit` |
| Python | 3.13.5 (MSC v.1943, 64-bit) |
| Platform | Windows 10.0.19045 |
| NumPy / pandas / SciPy | 2.3.1 / 2.3.3 / 1.16.0 |
| scikit-learn / Matplotlib | 1.9.1 / 3.11.0 |
| pytest / Ruff | 8.4.1 / 0.15.21 |

Full package lock: `reports/simulator/kato_final_audit/environment/pip_freeze.txt`.

## 2. Dataset and split verification

- File: `data/raw/kato2015/WT_NoStim.mat`
- SHA256: `7531811c1718e26a0455f59cb2f8188cf25809278cfea85770dfe03315a0098a`
- Sessions: 5
- Frames: 15,662
- Native dimensions: 103, 127, 123, 118, 121
- H20 landmarks: 175 (35, 35, 34, 38, 33)
- Train bounds: `0:2195`, `0:2193`, `0:2141`, `0:2317`, `0:2114`
- Test bounds: `2689:3137`, `2687:3134`, `2623:3059`, `2837:3311`, `2591:3021`
- Purge: 12 frames
- PCA: training-only, 10 components, `random_state=17`
- PCA-10 variance: 0.709861, 0.621639, 0.567138, 0.709508, 0.546713

The complete split manifest is in `preflight.json`.

## 3. Exact execution

The analysis source is the committed pipeline. The only change to the entry point
is an optional `--out-dir` argument; no model, split, seed or metric was changed.

```powershell
python -X utf8 scripts/kato_common_state_main.py `
  --out-dir reports/simulator/kato_final_audit/clean_rerun
```

- Started: `2026-11-03T16:41:55+02:00`
- PID: 28552
- Completed: `kato common-state complete 2005.9s`
- Reported elapsed: 2,005.9 s
- stdout: 36 bytes, one completion line
- stderr: 0 bytes
- Output: 21 files, 1,687,845 bytes
- Resume script: not used
- Historical directory: not overwritten

Provenance: `full_rerun_status.json`; logs in `logs/clean_rerun.stdout.log` and
`logs/clean_rerun.stderr.log`; exit provenance in `full_rerun_status.json`.

## 4. Comparison with the committed artifacts

Tolerance: `1e-8` absolute for deterministic float64/CPU outputs. The historical
`1e-6` G1 tolerance was not used as the clean-reproduction criterion.

| Artifact | Rows | Max absolute difference | Status |
| --- | ---: | ---: | --- |
| model predictions (`model_per_landmark.csv`) | 2,275 | 0.0 | identical |
| model summary | 7 | 1.11e-16 | identical |
| paired contrasts | 30 | 4.44e-16 | identical |
| decomposition | 6 | 1.11e-16 | identical |
| session effects | 5 | 1.11e-16 | identical |
| horizon profile | 24 | 4.44e-16 | identical |
| factor variance | 20 | 0.0 | identical |
| representation sensitivity | 4 | 8.33e-17 | identical |
| stage-A reproduction | 15 | 0.0 | identical |
| surrogate draws | 4,200 | 0.0 | identical |
| surrogate paired | 12 | 4.44e-16 | identical |
| surrogate summary | 4 | 0.0 | identical |
| synthetic audit | 15 | 0.0 | identical |
| synthetic summary | 5 | 0.0 | identical |
| gates | 8 | status match | identical |
| synthetic verification | — | exact JSON | identical |
| reproduction summary | — | exact JSON | identical |

Maximum absolute difference across all compared numeric values:
`4.44e-16`. Maximum relative difference at the worst column: `9.84e-16`.
Values above `1e-8`: zero.

The stochastic stages are also **seed-reproducible exactly**, not merely
statistically comparable: 4,200/4,200 surrogate draws and all 15 synthetic rows
match. This is expected because the seeds and CPU/float64 environment are fixed;
it does not establish distributional robustness to a different environment.

### Primary metrics

| Metric | Committed | Clean rerun | Difference |
| --- | ---: | ---: | ---: |
| B3 H20 energy | 2.7001497504411485 | 2.7001497504411485 | 0.0 |
| B5-10 H20 energy | 2.9645877431095045 | 2.9645877431095045 | 0.0 |
| B3 advantage (`B5−B3`) | 0.2644379926683563 | 0.2644379926683563 | 0.0 |
| Block-5 interval low | 0.0749929195790352 | 0.0749929195790351 | 9.71e-17 |
| Block-5 interval high | 0.4031164915680882 | 0.4031164915680882 | 0.0 |
| Fraction B3 better | 0.64 | 0.64 | 0.0 |
| Pairs | 175 | 175 | 0 |

The gate statuses match exactly:

```text
G1 PASS, G2 PASS, G3 PASS, G4 PASS,
G5 PASS, G6 PASS, G7 FAIL, G8 NOT_EVALUABLE
```

## 5. What this does and does not prove

It proves that the committed analysis can be regenerated from the entry point on
the same machine, environment, dataset and seeds, with all deterministic and
seeded-stochastic outputs agreeing to floating-point precision.

It does **not** prove:

- independence from the original operator/environment;
- animal-level generalization;
- that the test tails are fresh;
- that the ten-factor comparison is capacity-matched;
- that the sign rule was prespecified;
- that raw ensemble trajectory arrays exist (the pipeline preserves per-landmark
  scores and metrics, not raw ensemble trajectories);
- any biological mechanism.

The historical run was interrupted and post-processed; that admission is not
erased. The clean rerun is the reproduction evidence, and it is recorded
separately from the historical artifact production.

## 6. Commands to audit

```powershell
git rev-parse HEAD
Get-FileHash data/raw/kato2015/WT_NoStim.mat -Algorithm SHA256
python -X utf8 scripts/kato_final_audit_preflight.py
python -X utf8 scripts/kato_common_state_main.py --out-dir reports/simulator/kato_final_audit/clean_rerun
python -X utf8 scripts/kato_final_audit_compare.py
python -X utf8 scripts/kato_final_audit_uncertainty.py
python -X utf8 scripts/kato_final_audit_ten_factor.py
python -X utf8 scripts/kato_final_audit_comparability.py
python -X utf8 scripts/kato_final_audit_claims.py
python -X utf8 scripts/validate_kato_final_audit.py
```
