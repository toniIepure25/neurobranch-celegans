# Final PCA and 3D variance audit

**Date:** 2026-11-04

## 1. The projection

For each recording, the Studio fits
`PCA(n_components=10, random_state=17)` on the training split only and transforms
all frames of that recording. The 3D view plots the first three scores. It is a
population-state summary under the training covariance, **not** anatomy, a
connectome, or a physical brain space.

The four quantities that were previously conflated are:

| Quantity | Definition |
| --- | --- |
| Original feature variance | Sum of per-feature training variances, `ddof=1` |
| PCA-10 total explained | Sum of the first ten `explained_variance_ratio_` values |
| PC1–PC3 original variance | Sum of the first three ratios |
| Fraction of PCA-10 visualized | PC1–PC3 ratio sum divided by PCA-10 ratio sum |
| Remaining within PCA-10 | `1 - fraction visualized` |
| Outside PCA-10 | `1 - PCA-10 ratio sum` |

`sklearn.decomposition.PCA.explained_variance_ratio_` always divides by the
**total training feature variance**, including components beyond the requested
ten. It does not divide by the retained PCA-10 variance. The old statement that
"48% is outside the visible three axes within the ten-dimensional PCA" used the
total-variance denominator but described it as a PCA-10 remainder. For
recording 0 the correct PCA-10 remainder is **26.75%**, not 48%.

## 2. Session table

| Session | Original variance | PCA-10 explained | PC1–PC3 / original | PC1–PC3 / PCA-10 | PC4–PC10 / PCA-10 | Outside PCA-10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 103.04696655 | 73.14903259 | 0.51994640 | 0.73246199 | 0.26753810 | 0.29013872 |
| 1 | 127.05793762 | 78.98416138 | 0.42377472 | 0.68170547 | 0.31829438 | 0.37836099 |
| 2 | 123.05745697 | 69.79055786 | 0.39796349 | 0.70170504 | 0.29829487 | 0.43286216 |
| 3 | 118.05096436 | 83.75811768 | 0.47814870 | 0.67391574 | 0.32608426 | 0.29049188 |
| 4 | 121.05725098 | 66.18359375 | 0.36729366 | 0.67182148 | 0.32817841 | 0.45328683 |

For recording 0 the four requested statements are:

- original feature variance: `103.04696655`;
- variance explained by PCA-10: `73.14903259` (`0.70986110` of original);
- original variance explained by PC1–PC3: `0.51994640`;
- fraction of PCA-10 retained variance that is visualized: `0.73246199`;
- remaining variance within PCA-10: `0.26753810`;
- original variance outside PCA-10: `0.29013872`.

Full precision is in
`reports/scientific_studio/final_audit/pca_variance_audit.json`.

## 3. What changed in the product

- The 3D panel now states "of total training variance", then gives the
  PCA-10-relative percentage and the unshown PCA-10 fraction.
- Axis titles read `PC1 · 26.0% total`, not an unqualified percentage.
- The 2D fallback uses the same qualified axis titles.
- The Models caption distinguishes total variance from retained PCA-10
  variance.
- `TrajectoryStore.case()` exposes `explained_10d`,
  `explained_3d_of_10d`, `remaining_10d_of_10d`, `remaining_10d_of_total`, and
  `outside_10d_of_total` so no section has to recompute the denominator.
- `docs/STUDIO_3D_SCIENTIFIC_MAPPING.md` and the supervisor demo now use the
  correct denominator.

## 4. Geometry and scale checks

- All three axes are scores from the same fitted PCA; no axis is rescaled by a
  different standard deviation.
- `aspectmode="data"` preserves relative data extent. The axes are not forced
  to equal visual length, and Euclidean screen distance is not anatomical
  distance.
- Axis ranges are computed from the displayed history, origin, truth, and model
  means with 10% padding; changing one model does not silently rescale another.
- Camera is fixed to the same eye/center and `uirevision` unless the user
  orbits.
- Forecast origin is observed frame `t`; observed continuation is
  `t+1 … t+20`; model step 1 is `t+1`. No future observation is an input.
- The deterministic B3/B5 mean in the artifact was recomputed from the frozen
  model code at `(0, 2690)`: maximum absolute difference `1.19e-7` (B3) and
  `1.16e-7` (B5), with the last fitted transition at `2689 < 2690`.

**Status: PASS** for PCA semantics and trajectory/axis integrity. Visual
readability is a separate decision in
[STUDIO_FINAL_VISUAL_QA.md](STUDIO_FINAL_VISUAL_QA.md).
