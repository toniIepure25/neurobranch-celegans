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
