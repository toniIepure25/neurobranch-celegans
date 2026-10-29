# Corrections

Corrections are recorded here when a previously reported number, status or
interpretation changes. Each entry states what was claimed, what is correct,
the evidence, and the impact on conclusions.

## C1. Phase 7 normalised scores above 1

**Was claimed:** some Phase 7 models exceeded the reliability ceiling, with
normalised scores above 1.

**What happened:** the normalisation divided by a split-half ceiling that had
been truncated at zero. A truncated denominator is not a ceiling, and a ratio
above 1 is arithmetically possible only because the denominator was invalid.

**Correct:** normalised scores are withdrawn. Compare raw correlations to the
ceiling instead of dividing by it.

**Evidence:** `docs/PHASE7_RESULTS.md`, `docs/MEASUREMENT_LIMITS.md`.

**Impact:** no Phase 7 ranking depended only on the normalised score; the raw
correlations are unchanged. Any statement that a model "exceeded the ceiling"
is withdrawn.

## C2. "CRO reaches 64% of the ceiling"

**Was claimed:** the CRO model's pair Spearman of 0.0984 was 64% of the 0.152
split-half ceiling.

**Correct:** the split-half ceiling was biased downward. The variance-corrected
single-animal ceiling is 0.473, so 0.0984 is about **20.8%** of the ceiling.

**Evidence:** `reports/phase9/reliability_summary.json`,
`docs/PHASE9_RESULTS.md`.

**Impact:** the model is farther from the measurement limit than previously
stated. This correction is unfavourable to the project and is reported anyway.

## C3. Phase 8 H1 pair status

**Was claimed:** the H1 pair gate passed, and the pair-level advantage was
treated as supported.

**Correct:** the interval [+0.0080, +0.0901] excludes zero, but the design's
reconstructed MDE is 0.0587 against a smallest effect of interest of 0.020.
Under the Phase 9 decision rule the claim is `underpowered`, not `supported`.

**Evidence:** `reports/phase9/claims.json`;
`src/neurobranch/benchmark/power.py`.

**Impact:** the Phase 8 conclusion "H1 not supported overall" stands, but for a
stronger reason: the pair component alone is not resolved at the declared
margin. The result is not evidence of no effect; it is evidence that the design
cannot distinguish the effect of interest from zero.

## C4. Reliability estimator replacement

**Was claimed:** the split-half Spearman-Brown estimate (mean full reliability
0.152) was the atlas ceiling.

**Correct:** the variance-corrected estimates are 0.243 for repeat stimulation
and 0.223 for single-animal generalisation, with ceilings 0.493 and 0.473.

**Evidence:** `src/neurobranch/benchmark/variance.py`,
`tests/benchmark/test_variance.py`, `reports/phase9/`.

**Superseded by C7:** the 0.223 and 0.473 values used a denominator that
ignored per-responder animal counts and double-counted the interaction term.

**Impact:** the ceiling roughly triples. This correction is favourable to the
project's models in absolute terms but, combined with C2, still leaves them far
from the limit.

## C5. Bootstrap intervals for per-target reliability

**Was attempted:** animal-cluster bootstrap intervals for each target's
reliability.

**What happened:** resampling animals with replacement placed the same animal
in both split halves, inflating split-half agreement. The bootstrap upper
limits averaged 0.86 against a point estimate of 0.08. The intervals were
discarded rather than published.

**Correct:** per-target intervals are not reported. Aggregate intervals are
target-level normal approximations and are labelled descriptive.

**Evidence:** git history of `src/neurobranch/benchmark/variance.py`;
`docs/PHASE9_ANALYSIS_PLAN.md` (deviation 1).

**Impact:** no published number depends on the discarded intervals.


## C6. Phase 9 applied the single-animal ceiling to pooled metrics

**Was claimed:** the Phase 6-8 pair-level metrics are bounded by the
single-animal ceiling of 0.473, so the CRO of 0.098 reaches about 21% of its
limit.

**What happened:** Phase 6-8 pair Spearman is computed on pair means pooled
over the roughly 22 animals in a test fold, not on individual animals. A pooled
mean has a different attenuation ceiling, and that ceiling depends on the
number of animals pooled. The single-animal ceiling applies only to per-animal
claims.

**Correct (Phase 10):** the metric-aligned pooled ceiling at 22 animals is
0.833. The CRO of 0.098 is therefore about **11.8%** of its ceiling, not 21% and
not 64%. The single-animal ceiling is 0.403 and applies to individual-level
claims only.

**Superseded by C8:** the 0.833 value is itself coverage-sensitive. Requiring
responders to be observed in at least 80% of animals lowers the pooled ceiling
to 0.560, so the CRO is about 17.6% of the coverage-aware ceiling.

**Evidence:** `docs/PHASE10_RESULTS.md`, `reports/phase10/ceiling_curve.csv`,
`reports/phase10/reliability_summary.json`.

**Impact:** the old 64% claim remains withdrawn. The Phase 9 replacement (21%)
is also withdrawn because it used the wrong estimand. This correction is
unfavourable to the project: against its actual metric, the model is farther
from the measurement limit than Phase 9 stated. Phase 8 H1 remains
`underpowered`; the correction does not upgrade it.

## C7. Sparse-responder sampling variance and double-counted interaction

**Was claimed:** single-animal reliability is 0.223 with ceiling 0.473.

**What happened:** two defects in the animal facet:

1. the pooled sampling variance divided by the total animal count squared
   instead of the per-responder count squared, so responders measured in fewer
   animals were under-corrected and the signal component was inflated;
2. the signal component `V - S` still contained `interaction/A`, and the
   reliability denominator added `interaction/A` again, counting the
   animal x responder variance twice.

**Correct:** with per-responder counts and the interaction correction,
single-animal reliability is **0.162** (95% target-level interval
[0.134, 0.191]) with ceiling **0.403**. The repeat facet is unchanged at 0.243
(ceiling 0.493) because it does not pool across animals.

**Evidence:** `docs/PHASE10_RESULTS.md`,
`reports/phase10/profile_components.csv`,
`tests/benchmark/test_variance.py`.

**Impact:** the individual-level measurement limit is lower than Phase 9
reported. The conclusion that the atlas is measurement-limited for
single-animal claims is strengthened; the pooled-metric conclusion is not
affected in direction, only in magnitude.


## C8. The Phase 10 pooled ceiling depends on sparse responder coverage

**Was claimed:** the pooled reliability ceiling at 22 animals is 0.833, and the
CRO reaches about 11.8% of it.

**What happened:** the Phase 10 estimator included every responder, including
neurons observed in a single animal. A responder measured once contributes to
the pooled profile but has no between-animal sampling variance to subtract, so
apparent shared signal is inflated. Phase 11 repeated the estimate while
requiring each responder to be observed in a minimum fraction of the target's
animals:

| Minimum coverage | Pooled reliability | Ceiling | Targets |
|---:|---:|---:|---:|
| 0.0 (Phase 10) | 0.693 | 0.833 | 126 |
| 0.5 | 0.414 | 0.643 | 126 |
| 0.8 | 0.314 | **0.560** | 126 |
| 1.0 | 0.207 | 0.455 | 79 |

**Correct:** the defensible descriptive ceiling is **0.560** at 80% coverage;
the Phase 10 value 0.833 is a sparse-coverage historical estimate. The CRO of
0.098 is about **17.6%** of 0.560. The Phase 11 extrapolation gate also failed
(MAE 0.210, bias +0.016), so the curve beyond the observed animal count is not
validated.

**Evidence:** `docs/PHASE11_RESULTS.md`,
`reports/phase11/coverage_sensitivity.csv`,
`reports/phase11/extrapolation.json`,
`reports/phase11/summary.json`.

**Impact:** the headline pooled ceiling is withdrawn as a general estimate.
This is unfavourable to the project: the coverage-aware measurement limit is
lower, and the model is closer to it than Phase 10 reported. No claim is
upgraded; Phase 8 statuses are unchanged.
