# Phase 10 results: metric-aligned measurement ceilings

**Status:** complete for the ceiling correction and robustness audit; the
design interval calibration failed and is reported as a limitation.
**Input:** `reports/phase3/randi_trial_table_phase3.csv.gz`
(237,639 trials, 110 animals, 259 targets, 305 neurons; SHA-256 in
`reports/phase10/manifest.json`).
**Code:** `src/neurobranch/benchmark/variance.py`,
`src/neurobranch/benchmark/robustness.py`,
`src/neurobranch/design/`.
**Artifacts:** `reports/phase10/`.

## Headline

Phase 9 was right that this atlas is measurement-limited for **individual
animals**, and wrong to apply that limit to **pooled** metrics. With the
estimator corrected for sparse responders and for the interaction term, the
single-animal ceiling is **0.403**, while the ceiling for a profile pooled over
the 22 animals of a Phase 6-8 fold is **0.833**.

The practical consequence is narrow and important: claims about one animal are
severely bounded; claims about a pooled population mean are not. The Phase 6-8
models are therefore not near a measurement wall - they are far below a
relatively high ceiling.

| Facet | Targets | Mean reliability | 95% interval | Ceiling |
| --- | ---: | ---: | --- | ---: |
| Repeat stimulation (within animal) | 152 | 0.243 | [0.206, 0.279] | 0.493 |
| Single animal (n = 1) | 126 | 0.162 | [0.134, 0.191] | 0.403 |
| Pooled over 22 animals (fold) | 126 | 0.693 | [0.653, 0.734] | 0.833 |
| Pooled over all observed animals | 126 | 0.873 | [0.837, 0.909] | 0.934 |

Only 3.2% of targets reach single-animal reliability 0.5; 13.2% reach repeat
reliability 0.5. The pooled values at 10 and 55 animals are 0.554 and 0.814.

Intervals in the table are target-level normal approximations and are
descriptive: targets share animals. The animal jackknife for the fold-level
mean is reported separately below.

## The ceiling curve

`reports/phase10/ceiling_curve.csv` and `ceiling_curve.png` show reliability as
a function of the number of animals pooled:

| Animals pooled | Mean reliability | Ceiling | Targets beyond observed count |
| ---: | ---: | ---: | ---: |
| 1 | 0.162 | 0.403 | 0 |
| 2 | 0.254 | 0.504 | 0 |
| 5 | 0.418 | 0.646 | 0 |
| 10 | 0.554 | 0.744 | 26 |
| 22 | 0.693 | 0.833 | 87 |
| 55 | 0.814 | 0.902 | 125 |
| 110 | 0.873 | 0.934 | 126 |

Values with `extrapolated = True` exceed the observed animal count for that
target and are descriptive only. Extrapolation is the main reason the
right-hand side of the curve is smooth: the model, not the data, supplies the
shape.

## What changes relative to Phase 9

Two defects were found while aligning the metric with the ceiling (C7):

1. the pooled sampling variance used the total animal count squared rather than
   the per-responder count squared, inflating the signal for sparse responders;
2. the signal `V - S` still contained `interaction / A`, which the reliability
   denominator then added again.

Correcting both lowers the single-animal estimate from 0.223 to 0.162 and the
single-animal ceiling from 0.473 to 0.403. The repeat facet is unchanged.

The old Phase 7 claim that CRO reaches 64% of the ceiling remains withdrawn.
Phase 9 replaced it with 21%; that replacement used the single-animal ceiling
for a pooled metric and is also withdrawn. Against the metric-aligned fold
ceiling, CRO's 0.0984 is about **11.8%** of the ceiling. This is worse for the
model than either previous statement, and it is the correct comparison for the
metric that was actually computed (C6).

## Multiplicity check

The two Phase 8 inferential claims form a family. Their unadjusted normal
p-values are 0.035 (pair advantage) and 0.121 (EIG selection); after Holm
adjustment they are 0.070 and 0.121. The pair advantage therefore does **not**
survive multiplicity control at 0.05. Its status was already `underpowered`
because the MDE exceeds the SESOI, so no status changes; the adjusted value
makes the weakness explicit.

## Robustness

- **Response summary.** `response_mean` and `response_integral` give identical
  reliability because the estimator is scale-free. `response_residual` gives a
  fold ceiling of 0.833 with target-level Spearman 0.934 against
  `response_mean`. The ordering of targets is therefore only mildly
  summary-dependent.
- **Animal-half replication.** Splitting the 110 animals into two halves 25
  times gives a median Spearman of -0.005 (95% interval [-0.135, 0.182]) between
  per-target single-animal reliabilities. **Target-level reliability rankings
  do not replicate across animal samples.** The aggregate ceiling is stable;
  the per-target ranking is not. This is why no target prioritisation is
  offered.
- **Animal jackknife.** Delete-one-animal jackknife for the fold-level mean
  gives a standard error of 0.270 and a 95% interval of [0.165, 1.000]. The
  delete-one mean is 0.634 against a full-sample estimate of 0.693. The
  aggregate is therefore **fragile to single-animal deletion**, not precisely
  estimated. This is a negative result and is not hidden.

## Design budget

Using the fitted components, the number of animals needed for a target
reliability of 0.8 has a median of **25** and a 90th percentile of **104**;
for 0.5 the median is 6. Twenty-two animals per fold - the Phase 8 design - is
close to the median for 0.8 but far below the p90. This is the concrete reason
the Phase 8 MDE was large.

`python scripts/phase10_design.py` reproduces the table.

## Calibration failure

The simulation coverage study does **not** validate the jackknife interval as a
confidence interval for the population ceiling. On synthetic atlases with known
components, nominal 95% intervals covered the truth in roughly 20-60% of
replicates, depending on the number of targets, responders and animals. The
interval captures animal resampling but not responder sampling or component
estimation bias, and it is too narrow.

**Consequence:** the jackknife interval is reported as a design diagnostic, not
as a calibrated confidence interval. No claim in this repository uses it as an
inferential interval. A two-way (animal x responder) resampling scheme is the
next methodological step; it is not implemented here.

## Limitations

- The estimator is a moment approximation, not a likelihood or REML fit.
- The pooled curve assumes exchangeable animals and stable components;
  extrapolation beyond observed counts is model-driven.
- Target-level reliability rankings are not reproducible across animal halves.
- The jackknife interval is not calibrated; see above.
- The design calculator is a plug-in calculation, not a validated prospective
  design tool.
- No causal claim, no digital-twin claim and no foundation-model claim is made.

## Reproduction

```powershell
python scripts/phase10_reliability.py
python scripts/phase10_figure.py
python scripts/phase10_robustness.py
python scripts/phase10_claims.py --check
python scripts/phase10_design.py
python scripts/validate_reports.py --strict
python -m pytest -q
```
