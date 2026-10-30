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
the Phase 10 value 0.833 is a sparse-coverage historical estimate. The coverage
filter also changes which responders are included, so 0.560 is the ceiling for
the broadly observed responder population, not a bias-corrected 0.833. The CRO of
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

## C9. The Phase 12 hierarchical interval is not calibrated

**Was attempted:** replace the Phase 11 moment bound with a hierarchical
measurement model and a calibrated posterior interval for pooled reliability and
the attenuation ceiling.

**What happened:** the model reproduced the point estimate but failed every
interval/calibration gate that matters:

- SBC 22-animal coverage was **0.923**, below the [0.93, 0.97] gate; rank
  uniformity failed (KS p = **1.4e-06**, mean rank 0.568);
- held-out animal predictive coverage was **0.896** with median absolute error
  **0.191** (gates: >= 0.90 and <= 0.10);
- permuting responder labels within each animal still gave median
  **R(22) = 0.176** and q95 = 0.525 (gates: <= 0.05 and <= 0.15);
- prior sensitivity passed (range 0.003), but that cannot repair the other
  failures.

**Correct:** the hierarchical point estimate is **0.304** (ceiling **0.551**),
within 0.05 of the Phase 11 moment estimate (0.314 / 0.560), but it is a
model-based sensitivity result, not a calibrated interval. The **0.560**
coverage-aware descriptive ceiling remains the headline. No target ranking and
no posterior animal-budget guarantee are published.

**Evidence:** `docs/PHASE12_RESULTS.md`, `reports/phase12/summary.json`,
`reports/phase12/sbc.json`, `reports/phase12/heldout_ppc.json`,
`reports/phase12/negative_control.json`.

**Impact:** unfavourable to the project: a more elaborate model did not earn a
better uncertainty statement. The Phase 8/9/10/11 claim statuses are unchanged.
The Phase 12 benchmark cluster analysis also found no configuration supported
at the 0.05 SESOI (largest delta +0.189, MDE 0.238, Holm p 0.185).

## C10. The Phase 12 calibration failure stands, but not because of a variance-loss bug

**Was claimed in the Phase 12R plan:** a post-report audit had found that
`_cells_from_template` discarded the known cell sampling variance, so the Phase
12 SBC and negative control fit a model with `v=0`.

**What happened:** that diagnosis was tested directly before any Phase 12R
result was produced and was **not reproduced**. `fit_hierarchical_cells()` takes
per-cell summaries directly and preserves `var / n`; the hypothetical `v=0`
path was created by testing `cell_summaries()` on a one-event frame, not by the
SBC path. The real defects were:

1. simulated templates were already animal-centered and standardized, then
   centered again by the fit, while the SBC truth used the uncentered formula;
2. the prose said `min_events=2`, `min_animals=2`, and two responders, while the
   executed primary used `min_events=1`, `min_animals=6`, and three responders;
3. 79% of observed cell variances are exactly zero because most cells contain
   one event, so the fit's `1e-8` floor carries most of the likelihood.

Phase 12R aligned the simulation, fit, and truth and repeated the frozen gates.

| Quantity | Phase 12 | Phase 12R |
|---|---:|---:|
| Pooled R(22) | 0.3035 | 0.3035 |
| Ceiling | 0.5509 | 0.5509 |
| SBC coverage at 22 animals | 0.923 | **0.933** (pass) |
| SBC rank KS p | 1.38e-06 | **1.43e-06** (fail) |
| SBC fit-integrity failure fraction | not reported | 0.000 (pass) |
| Maximum grid edge mass | 0.0069 | 0.0392 (pass) |
| Held-out PPC coverage | 0.896 | **0.904** (pass) |
| Held-out PPC median absolute error | 0.191 | **0.207** (fail) |
| Negative-control median R(22) | 0.176 | **0.120** (fail) |
| Negative-control q95 R(22) | 0.525 | **0.594** (fail) |
| Prior-sensitivity range | 0.0031 | 0.0031 (pass) |

The aligned rerun improves nominal coverage but does not make the interval
calibrated: rank uniformity still fails decisively, predictive error is over
twice the gate, and the negative control still returns substantial apparent
reliability. The point estimate is unchanged and remains within 0.05 of Phase
11. The `min_events=2` sensitivity gives **0.199** with 36 eligible targets, so
the 0.304 estimate is sensitive to the event filter.

**Correct:** the Phase 12 conclusion is unchanged: **the hierarchical interval
is not calibrated and 0.560 remains the only descriptive ceiling**. The reason
is model/coverage misspecification and measurement-variance floor behavior, not
a variance-loss coding bug. No target ranking, animal-budget guarantee, or
calibrated interval is published. The variance floor is an explicit
approximation, not a measured variance.

**Evidence:** `research/phase12r_protocol.md`,
`research/phase12r_deviations.md`, `reports/phase12r/summary.json`,
`reports/phase12r/sbc.json`, `reports/phase12r/heldout_ppc.json`,
`reports/phase12r/negative_control.json`,
`reports/phase12r/min_events_sensitivity.csv`.

**Impact:** unfavourable and corrective. It removes a false bug narrative,
documents that the earlier calibration evidence was misaligned, and shows that
the negative conclusion survives the correction for different reasons. Phase
8/9/10/11 claim statuses are unchanged.

## C11. The Phase 12 animal-cluster benchmark comparison is invalid, not merely underpowered

**Was claimed:** the largest animal-cluster benchmark delta was +0.189 for
`global_mean`, with a 95% interval [+0.017, +0.345], Holm p 0.185, MDE 0.238;
all configurations were labelled `underpowered`.

**What happened:** the old merge paired baseline and model rows by
`dataset_id, target, neuron` aggregates rather than by the same events. In the
saved Phase 8 predictions, baseline and model `y_true` differ for **1334 of
1334** paired rows (the baseline table also has a different event coverage).
An animal-cluster interval computed on mismatched truths cannot support a
paired delta, even a negative or underpowered one.

**Correct:** withdraw the benchmark cluster claim entirely. The Phase 12
benchmark table is a historical diagnostic, not evidence of an underpowered
comparison. A valid comparison must pair on `event_id` before resampling
animals. `paired_cluster_inference` now fails closed on unaligned truth by
default; the old behavior is available only through the explicit
`require_aligned_truth=False` reproduction flag.

**Evidence:** `reports/phase12r/inference_alignment_audit.json`,
`reports/phase12/benchmark_cluster.json`,
`src/neurobranch/benchmark/inference.py`,
`tests/benchmark/test_inference.py`.

**Impact:** no benchmark superiority claim, and no “underpowered but positive”
framing, is permitted. Leaderboard values remain descriptive only. This is
unfavourable to the project but prevents an invalid statistical claim.

## C12. The Phase 12R 0.304 estimate was dominated by the variance floor, not centering

**Was claimed in the Phase 13 plan:** the Phase 12R hierarchical reliability
estimate could be repaired by correcting the animal-centering projection and by
replacing the `1e-8` one-event variance floor with an event-level model.

**What happened:** Phase 13 separated the two corrections. The projection-only
model M1 gives pooled R(22) = 0.3138 (ceiling 0.5602), within 0.0002 of the
Phase 11 reference 0.314; the centering algebra was therefore not the main driver
of the Phase 12R point estimate. The event-variance model M2 gives
R(22) = 0.2003 (ceiling 0.4475), a change of 0.1032. The median event variance is
0.6906 on the standardized scale, and 78.9% of observed cells have one event, so
the old floor treated most cells as almost noiseless.

**Correct:** the Phase 12R estimate 0.3035 is floor-sensitive and must not be
presented as a stable measurement limit. The lower M2 value is **not** a
calibrated replacement: M2 fails SBC coverage (0.9244) and rank (KS p = 0.00306),
both null controls (negative median 0.1485, q95 0.4680; sign-flip median 0.1341,
q95 0.5796), and the Phase 11 agreement gate. The held-out PPC was not run
(deviation D4). The decision is `not_adopted_gate_failures`; 0.560 remains the
only descriptive coverage-aware ceiling.

**Evidence:** `reports/phase13/summary.json`, `docs/PHASE13_RESULTS.md`,
`research/phase13_deviations.md`.

**Impact:** no calibrated interval, target ranking or animal-budget statement is
published. The Phase 12R negative result stands and is now better diagnosed: the
floor was the dominant approximation, while the centering correction was nearly
neutral for the point estimate. This correction is unfavourable to the project
and is reported anyway.

## C13. Phase 13 G6 compared two different one-event variance treatments

**Was claimed in the Phase 13 protocol:** G6 tested whether M2 agreed with the
Phase 11 reference `0.314 / 0.560` within 0.05.

**What happened:** the Phase 11 reference uses `cell_summaries`, which fills a
one-event cell variance with `0.0`. M2 uses `omega^2 / n`. G6 therefore compared
two different variance treatments, not only two estimators. Phase 14A reproduced
V0 exactly and replaced the one-event zero with a robust global event variance
(`omega^2 = 1.7629`). V1 gives `R(22)=0.3006` and ceiling `0.5482`; the difference
from V0 is 0.0131, below the 0.05 audit threshold. V1h gives `0.3139 / 0.5603`.
The historical `0.560` is therefore robust to this assumption under a robust
estimator.

**Correct:** the G6 mismatch did not inflate `0.560` substantially. Phase 13's
negative decision still stands for independent reasons: SBC rank, null controls,
and the unrun PPC. The `0.560` bound remains population-specific (126 targets,
80% coverage); the `min_events=2` estimate `0.1754` is a different 36-target
estimand, not a correction.

**Evidence:** `reports/phase14/reference_audit.json`,
`docs/REFERENCE_AUDIT.md`, `research/phase14_deviations.md` D1.

**Impact:** Phase 13 G6 is reclassified as a mismatched reference test, not as
evidence that the reference was wrong. The Phase 14 model is still rejected.

## C14. An exact zero spike does not repair the reliability null bias

**Was proposed:** adding an exact point mass at `sigma = 0` would make the model
assign zero reliability under a responder-permutation null and pass calibration.

**What happened:** M4 removes the median null bias (median `R(22)=0`), but the
upper tail remains too heavy: negative-control q95 = 0.3535 and sign-flip q95 =
0.4509. The spike is also poorly calibrated when the truth is zero (median
`P(z=0)=0.3561`), SBC ranks are non-uniform (KS p = 3.35e-192), slab edge mass is
0.0810, held-out PPC coverage is 0.7697 with MAE 0.3998, prior range is 0.1220,
and the real-data median `0.0543` is 0.2463 away from the V1 reference. Eight of
twelve frozen gates fail.

**Correct:** the decision is `not_adopted_gate_failures`. No calibrated interval,
target ranking, animal budget, causal effect, or digital-twin claim is supported.
No prior, grid, seed, or threshold was tuned after the full run. Phase 14 is
published as a negative result; `v1.3.2` is the appropriate release.

**Evidence:** `reports/phase14/summary.json`, `reports/phase14/sbc.json`,
`reports/phase14/heldout_ppc.json`, `reports/phase14/negative_control.json`,
`docs/PHASE14_RESULTS.md`.

**Impact:** a zero median is not sufficient evidence of a valid null model. The
false-positive tail and spike calibration must be reported alongside it. This is
unfavourable to the proposed repair and is reported anyway.
