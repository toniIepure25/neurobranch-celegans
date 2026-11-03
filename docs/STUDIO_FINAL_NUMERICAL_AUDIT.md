# Final numerical audit

**Date:** 2026-11-04  
**Starting HEAD:** `ec27dd87a816cbce10dab70491c31f13dc59e3e2`  
**Branch:** `research/neurobranch-final-quality-audit`

## 1. Scope

This audit resolves the reconstruction discrepancy reported for
`scripts/master_review/independent_model_check.py` and checks the B3/B5
H20 PCA-10 energy arithmetic against the frozen rows. It does not rerun the
175-landmark study, select a new model, or promote the exploratory claim.

Source of truth remains:

1. `reports/simulator/kato_common_state/model_per_landmark.csv`
2. `reports/master_review/statistical_audit.json`
3. `reports/scientific_studio/kato_trajectories.npz`

The independent check is a reconstruction, not a replacement for those files.

## 2. Reported discrepancy and root cause

The historical documentation reported:

| Landmark | Historical independent B3 | Frozen B3 |
| --- | ---: | ---: |
| `(session 0, t=2750)` | `1.473899` | `1.360753` |
| `(session 0, t=2690)` | `2.410819` | `2.498644` |

**Root cause: the historical reference drew 20 innovations instead of 100.**
The J5 moving-block sampler draws `ceil(horizon / block)` random block starts
per member. A 20-step request therefore consumes a different random path than
the first 20 steps of a 100-step request; the short draw is **not** a prefix of
the long draw. The official pipeline always rolls 100 steps and scores step 20,
even though only 20 steps are displayed.

The negative control reproduces both historical numbers from the same model,
PCA, and scoring code:

| Landmark | Official H100 reference | 20-step draw | Historical (6 dp) |
| --- | ---: | ---: | ---: |
| `(0, 2750)` | `1.360753653911` | `1.473899352733` | `1.473899` |
| `(0, 2690)` | `2.498643590923` | `2.410818931729` | `2.410819` |

The six-decimal historical values agree with the 20-step control. They do not
indicate an error in the frozen B3 model, the official ridge fit, the PCA
projection, or the energy score.

## 3. Stage-by-stage comparison

The stage comparison at the two disputed landmarks:

| Stage | `(0, 2750)` | `(0, 2690)` |
| --- | ---: | ---: |
| Transition count | 499 | 499 |
| First / last transition | 2251 / 2749 | 2191 / 2689 |
| `max abs(A_reference - A_official)` | `2.52e-13` | `3.25e-13` |
| `max abs(b_reference - b_official)` | `1.20e-13` | `3.84e-13` |
| Reference radius | `0.990315626666` | `0.995000000000` |
| H100 B3 vs frozen | `3.23e-7` | `4.17e-7` |
| H20 negative control vs frozen | `1.13e-1` | `8.78e-2` |

The first divergent stage is **innovation sampling**, not model fitting or
scoring. The operator differences are at floating-point noise scale.

## 4. Extended fixed-landmark check

Rule, fixed before running: take the first and last frozen H100 landmark of
each recording, plus the errata landmark `(0, 2750)`. This is 11 cases and does
not use a performance criterion, tuning, or model selection.

Results (independent NumPy reference vs frozen rows):

| Session | Landmark | B3 abs diff | B5 abs diff |
| ---: | ---: | ---: | ---: |
| 0 | 2690 | `4.17e-7` | `3.21e-7` |
| 1 | 2688 | `7.71e-8` | `1.98e-7` |
| 2 | 2624 | `1.14e-7` | `5.78e-7` |
| 3 | 2838 | `9.82e-8` | `1.14e-7` |
| 4 | 2592 | `6.56e-8` | `1.69e-8` |
| 0 | 3030 | `1.79e-8` | `4.47e-10` |
| 1 | 3028 | `1.13e-7` | `1.36e-7` |
| 2 | 2954 | `9.87e-8` | `1.93e-8` |
| 3 | 3208 | `3.90e-9` | `3.71e-8` |
| 4 | 2912 | `1.33e-7` | `1.20e-7` |
| 0 | 2750 | `3.23e-7` | `3.53e-7` |

Maximum B3 difference: `4.17e-7`; maximum B5 difference: `5.78e-7`; tolerance
`2e-5`. The tolerance is for float32 artifact storage and PCA sign/rounding,
not an inferential tolerance.

**Status: PASS.** Evidence:
`reports/scientific_studio/final_audit/independent_model_check.json`.

## 5. What was independently implemented

The checker reimplements:

- train-only PCA by SVD (`pca_reference`);
- ridge VAR by normal equations (`ridge_var`);
- B3 and B5 rollouts (`b3_reference`, `b5_reference`);
- the energy score (`energy`).

It reuses only the historical J5 innovation sampler, because a common random
path is required for a paired comparison. A synthetic fixture
(`tests/test_innovation_horizon_contract.py`) proves that a 20-step moving-block
draw is not a prefix of a 100-step draw. A third duplicate implementation was
not needed after the divergence was localized; the official operator and the
independent operator already agree to `~1e-13`.

## 6. Limitations

- The check uses the same raw preprocessing and J5 residual pool as the
  official run. It tests arithmetic, not a new data pipeline.
- It is a code/math cross-check at 11 landmarks, not a full 175-landmark rerun.
- No new animal, split, model, or endpoint was introduced.
- The primary result remains exploratory and horizon-dependent.
