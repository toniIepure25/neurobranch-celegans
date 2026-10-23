# Branching statistical audit (Phase 2)

**Status:** corrected analysis, executed on all five Kato recordings.
**Artifacts:** `reports/phase2a_scientific_audit/branching/`
**Code:** `src/neurobranch/branching/stats.py`, `src/neurobranch/branching/synthetic.py`,
`src/neurobranch/branching/__init__.py`
**Reproduce:** `python scripts/phase2a_branching_audit.py`

> This is an analysis of **local outcome diversity in a projection**, not a
> bifurcation analysis. A low or high entropy value does not identify a
> parameter-dependent change in the vector field.

## 1. Why the Phase 1 p-value was misleading

Phase 1 reported `p = 1.0` for the one-sided test `observed entropy > shuffled
null`. That is correct for the *excess* direction, but it was easy to read as
"there is no structure". The corrected audit reports:

- `p_excess`: fraction of null statistics at least as large as observed;
- `p_deficit`: fraction at least as small as observed;
- `p_two_sided = 2 * min(p_excess, p_deficit)` (capped at 1).

The important result is a **deficit**, not an excess: observed local outcome
entropy is lower than expected when outcomes are decoupled from latent geometry.
That is evidence that nearby states have more consistent futures than chance, not
evidence for branching.

## 2. Definitions

For held-out test frames in macro-state REV:

1. Fit a 3-D PCA on **training frames only**; project all frames.
2. For each candidate frame, take its `k = 40` nearest candidates in the
   projection, excluding frames within `±12` frames in time.
3. Outcome = first macro-state different from the current state within a
   horizon of 20 frames.
4. Statistic = mean Shannon entropy (bits) of the kNN outcome distribution.

Candidate frames overlap heavily in time. The mean-entropy interval is therefore
a **moving-block bootstrap over candidate frames** (block size 18), not an iid
interval over independent states.

## 3. Null specifications

| Null | What it preserves | Why it matters |
|---|---|---|
| iid permutation | outcome counts | baseline used in Phase 1 |
| block permutation (10/30/60) | local temporal runs of outcomes | tests whether temporal autocorrelation explains the result |
| circular shift | outcome sequence, autocorrelation | strongest temporal control; 500 shifts |

The primary null is block permutation. The circular-shift result is reported as a
sensitivity analysis, not replaced by the iid result.

## 4. Real recordings

`M = mean observed entropy`; `N = null mean`; entries are p-values. All tests use
500 permutations (iid/block) or 500 shifts. Full 25-row table:
`branching_all_worms_long.csv`.

| Worm | Candidates | M [95% block-bootstrap CI] | iid N | block-10 N | p_deficit iid | p_deficit block-10 | min p_excess (all nulls) | shift p_two_sided |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 328 | 0.578 [0.448, 0.697] | 0.961 | 0.905 | 0.00498 | 0.00498 | 0.985 | 0.0398 |
| 1 | 248 | 0.625 [0.504, 0.743] | 0.978 | 0.933 | 0.00498 | 0.00498 | 0.965 | 0.0796 |
| 2 | 383 | 0.780 [0.698, 0.854] | 0.880 | 0.847 | 0.00498 | 0.0448 | 0.821 | 0.368 |
| 3 | 226 | 0.889 [0.830, 0.937] | 0.924 | 0.860 | 0.0249 | 0.677 | 0.0896 | 0.338 |
| 4 | 412 | 0.603 [0.509, 0.692] | 0.948 | 0.887 | 0.00498 | 0.00498 | 0.955 | 0.0995 |

Key readings:

- **4/5 worms show a significant entropy deficit** under iid and block-10 nulls
  (p ≈ 0.005; worm 2 weaker under block-10, p = 0.045).
- **No worm shows a significant excess** under any null. The smallest one-sided
  excess p is 0.0896 (worm 3, block-30), two-sided 0.179; it does not survive the
  shift null (p = 0.338).
- **Worm 3 is not significant under block nulls** (block-10 p = 0.677), so the
  deficit is not universal. Worm 2 also loses significance under longer blocks
  and shift.
- The circular-shift null is the most conservative. Only worms 0 and 4 are below
  0.05 two-sided; worm 1 is 0.080, worm 2 0.368, worm 3 0.338.

The defensible statement is: *in this projection, local outcome diversity is
often lower than a geometry-preserving null, but this is projection- and
worm-dependent and is not evidence of branching.* The Phase 1 "p = 1.0" should
not be cited as a negative result about structure.

## 5. Projection-width sensitivity

Spearman correlation of per-frame entropy between the 3-D projection and wider
PCA projections (common frames):

| Worm | 8-D | 16-D | 32-D | 3-D variance explained |
|---:|---:|---:|---:|---:|
| 0 | 0.765 | 0.707 | 0.722 | 53.1% |
| 1 | 0.783 | 0.810 | 0.778 | 43.1% |
| 2 | 0.376 | 0.350 | 0.301 | 42.0% |
| 3 | 0.340 | 0.210 | 0.151 | 51.1% |
| 4 | 0.757 | 0.733 | 0.713 | 38.6% |

Worms 2 and 3 change substantially with projection width. Any frame-level
"branching score" for those worms is not stable enough to interpret
biologically.

## 6. Synthetic ground-truth checks

These are generated fixtures, not recordings. They test whether the statistic
does what it claims. Full table: `synthetic_branching_audit.csv`.

| Scenario | Truth | 3-D observed | iid null | p_excess (iid / block-10) | Interpretation |
|---|---|---:|---:|---:|---|
| deterministic | one outcome | 0.000 | 0.000 | 1.00 / 1.00 | no false excess; null also degenerate |
| stochastic | 3-way random | 1.469 | 1.423 | 0.140 / 0.066 | not detectable as geometry-driven branching |
| rare | dominant + rare | 0.270 | 0.307 | 0.724 / 0.688 | no excess |
| hidden | unobserved binary state | 0.970 | 0.949 | 0.143 / 0.123 | unobserved state produces high entropy but no excess |
| projection (3-D) | two high-D states collapsed | 0.924 | 0.950 | 0.890 / 0.880 | projection destroys separability; no excess |
| projection (8-D) | same, full width | 0.000 | 0.950 | 1.00 / 1.00 | high-D recovers deterministic outcomes |

The fixture suite shows the test has low power for genuine stochastic branching:
when outcomes are not predictable from geometry, shuffling outcomes does not
change expected local entropy much. A null result therefore cannot distinguish
"no branching" from "branching not represented in this projection". This is a
core limitation, not a bug to tune away.

## 7. What this does and does not establish

**Supported**

- The Phase 1 one-sided excess test does not support branching.
- In 4/5 worms, local outcome entropy is *lower* than a geometry-preserving null
  under iid/block-10 nulls; this is local predictability, not branching.
- The result is sensitive to temporal null, projection width, and worm.

**Not supported**

- A claim that the system has no branching.
- A claim that entropy deficits imply a bifurcation or causal mechanism.
- A claim that the 3-D projection is a state space with biological coordinates.
- Generalization beyond Kato's five worms, AVA/SMD/RIB-derived labels, and the
  blocked test block.

## 8. Caveats

1. Candidate frames overlap; the effective sample size is much smaller than the
   frame count. Block bootstrap is a partial correction.
2. Outcomes are macro-behaviour labels derived from AVA, SMD and RIB activity.
   The label geometry and the predictor geometry are not independent.
3. The blocked split is interpolation. The strict-forward split is reported
   separately and collapses for most models.
4. Only 2 outcome classes occur among REV candidates (REV/TURN), so entropy is
   bounded by 1 bit and depends strongly on class balance.
5. `p` values are not corrected across five worms and multiple nulls; the
   per-worm pattern, not a single p, is the evidence.
