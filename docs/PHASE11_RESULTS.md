# Phase 11 results: coverage sensitivity and failed uncertainty gates

Phase 11 audits the Phase 10 ceiling rather than adding a new biological claim.
The headline is negative: **the Phase 10 pooled ceiling is not robust to
responder coverage**, and none of the three interval/shrinkage procedures passed
its pre-declared gate.

## 1. Coverage sensitivity

The Phase 10 estimator uses every responder, including neurons observed in a
single animal. Requiring a responder to be present in a minimum fraction of the
target's animals changes the estimate monotonically:

| Minimum responder coverage | Targets | Mean responders | Pooled reliability at 22 | Pooled ceiling |
|---:|---:|---:|---:|---:|
| 0.0 (Phase 10) | 126 | 207.1 | 0.693 | 0.833 |
| 0.3 | 126 | 104.0 | 0.524 | 0.724 |
| 0.5 | 126 | 61.5 | 0.414 | 0.643 |
| **0.8** | **126** | **21.1** | **0.314** | **0.560** |
| 1.0 | 79 | 7.1 | 0.207 | 0.455 |

At coverage 0.8 the pooled ceiling is **0.560**, not 0.833. The difference
(0.273) is far beyond the 0.10 sensitivity gate, so G1 fails. Sparse coverage was
manufacturing apparent shared signal: a responder measured once contributes to
the pooled profile but has no between-animal sampling variance to subtract.

Phase 10's `phase10-fold-ceiling` is therefore a **coverage-sensitive historical
estimate**, not a headline. This is recorded as correction C8.

## 2. Extrapolation gate

Animal halves, coverage 0.8, 15 splits, 74 eligible targets:

| Quantity | Value |
|---|---:|
| median observed cross-half correlation | 0.184 |
| median predicted `sqrt(rel_A rel_B)` | 0.058 |
| median absolute error | 0.210 |
| median bias | +0.016 |
| gate (MAE <= 0.10, abs bias <= 0.05) | **fail** |

The components over-predict the observed cross-half correlation by about 0.21.
The additive model does not describe the held-out half profiles well enough to
justify extrapolating the curve beyond the observed animal count.

## 3. Shrinkage gate

Empirical-Bayes logit shrinkage at coverage 0.8 collapses the per-target
estimates toward the population mean (the between-target variance is small
relative to the approximate sampling variance). The half-split correlation
becomes undefined because the shrunk values are constant; 0 valid splits.

**Gate: fail.** No target ranking is published. This is consistent with the
Phase 10 finding that rankings do not replicate (median raw half-split Spearman
0.209 in this run, versus approximately zero in the Phase 10 analysis; neither
supports a recommendation).

## 4. Two-way bootstrap gate

The percentile bootstrap resamples animals and responders with replacement.
At coverage 0.8, 300 replicates, 126 targets:

| Quantity | Value |
|---|---:|
| point estimate | 0.314 |
| bootstrap mean | 0.540 |
| 95% percentile interval | [0.497, 0.583] |
| bias | +0.226 |
| point inside interval | **no** |

Resampling animal rows with replacement duplicates them, which shrinks the
estimated between-animal interaction variance and inflates reliability. The
interval is **not calibrated** and must not be reported as a confidence
interval. This is a methodological failure, not a result to be rescued by
choosing a different percentile.

The synthetic coverage study is reported separately (`coverage_study.json`): 100
replicates, true reliability 0.917, coverage **1.000**, mean width 0.185, bias
−0.009. That is conservative on the additive simulation, but it does not rescue
the real-data interval: the real bootstrap bias (+0.226) comes from resampling
sparse responder rows, a feature the balanced simulation does not contain. The
synthetic result is not used to certify the real-data interval.

## 5. What survives

- The **direction** is stable: requiring broader responder coverage lowers the
  ceiling.
- A defensible descriptive bound at coverage 0.8 is a pooled ceiling of **0.560**
  at 22 animals, with the caveat that the extrapolation gate failed.
- The Phase 10 single-animal conclusion (low reliability for one animal) is
  unchanged in direction; coverage reduces it further (0.162 → 0.034 mean).
- No ranking, no calibrated interval and no causal claim are supported.

## 6. Decision

The benchmark card defaults to the coverage-aware ceiling (0.560), not 0.833.
Phase 10 artifacts remain frozen and are labelled superseded for headline use.
The next analysis must use a hierarchical model with explicit responder
coverage, or a prospective design with balanced coverage, before any interval is
called calibrated.

## 7. Reproduce

```bash
python scripts/phase11_sensitivity.py
python scripts/phase11_benchmark.py
python scripts/phase11_manifest.py
python scripts/audit_claims.py
python scripts/check_reproducibility.py --reports reports/phase11
```

Artifacts are in `reports/phase11/`; the input table is not redistributed (see
[`LICENSE_AUDIT.md`](LICENSE_AUDIT.md)).
