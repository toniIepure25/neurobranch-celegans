# Phase 12 results: hierarchical model, failed calibration, retained 0.560

Phase 12 asked whether an explicit hierarchical measurement model can replace the
Phase 11 moment bound with a calibrated interval. The headline is negative: the
model reproduces the point estimate but **fails the calibration gates**, so the
coverage-aware **0.560** ceiling remains the only defensible descriptive bound.

## 1. Headline

| Quantity | Phase 11 moment | Phase 12 hierarchical |
| --- | ---: | ---: |
| Pooled reliability at 22 animals, 80% coverage | 0.314 | **0.304** |
| Attenuation ceiling | **0.560** | 0.551 |
| Single-animal reliability | 0.034 | 0.019 |
| Eligible targets | 126 | 126 |
| Calibrated interval? | no | **no** |

The difference in the pooled point estimate is **−0.010**, within the
pre-declared 0.05 reproduction margin. That is the only gate the model passes.
Agreement of point estimates is not evidence that the posterior interval is
calibrated.

The coverage curve is non-monotone in target population because the filter
changes which responders and targets are eligible. At 80% coverage the model
gives 126 eligible targets; at 100% it gives 79. The 0.359 estimate at full
coverage is therefore not a stricter version of the same estimand.

## 2. Calibration gates

| Gate | Requirement | Observed | Result |
| --- | --- | ---: | --- |
| SBC 22-animal coverage | 0.93--0.97 | **0.923** | fail |
| SBC rank uniformity | KS p > 0.01 | **1.38e-06** | fail |
| Grid boundary mass | < 0.05 | 0.0069 | pass |
| Held-out coverage | >= 0.90 | **0.896** | fail |
| Held-out median absolute error | <= 0.10 | **0.191** | fail |
| Negative-control median R(22) | <= 0.05 | **0.176** | fail |
| Negative-control q95 R(22) | <= 0.15 | **0.525** | fail |
| Prior-sensitivity range | <= 0.05 | 0.0031 | pass |

SBC used 200 replicates x 3 targets = 600 fits, with real coverage masks and
sampling variances, 1,200 posterior draws on a 50 x 50 grid. Coverage 0.923 is
just below the gate, but the rank test is decisively non-uniform (mean rank
0.568), so this is not a borderline pass that a larger simulation would rescue.
The held-out check used five animal-level splits and 77 predictions; coverage is
below nominal and the median error is nearly twice the allowed value.

## 3. Positive null bias

The negative control permutes responder labels **within each animal**, preserving
each animal's coverage pattern and marginal response distribution while
destroying true cell correspondence. A model with no true shared structure should
return near-zero reliability. It returns:

- median R(22) = **0.176**;
- 95th percentile R(22) = **0.525**;
- maximum R(22) = **0.620**;
- 26 eligible permuted targets.

This is the most damaging result. The hierarchical model can manufacture
apparent reliability from coverage geometry and marginal distributions alone.
The earlier event-level label shuffle gave a smaller but still positive 0.193;
the within-animal profile permutation is the stronger control and is the one
reported.

The prior-sensitivity gate passes, so the failure is not a prior-scale artefact.
Changing the HalfNormal scale from 0.5 to 2.0 moves pooled R(22) only from 0.260
to 0.263. The problem is the likelihood/coverage structure, not the prior width.

## 4. Animal-cluster benchmark inference

Fold-level leaderboard values share animals across folds and cannot support a
generalisation claim. Phase 12 resamples animals, computes within-animal pair
means, and applies Holm adjustment across the five non-baseline configurations.

| Configuration | Paired delta vs static CRO | 95% animal-cluster interval | Holm p | MDE (80%) | Status |
| --- | ---: | --- | ---: | ---: | --- |
| `global_mean` k=0 | +0.189 | [+0.017, +0.345] | 0.185 | 0.238 | underpowered |
| `twin` k=8 random | +0.090 | [−0.074, +0.247] | 1.000 | 0.238 | underpowered |
| `twin` k=8 hybrid | +0.087 | [−0.082, +0.252] | 1.000 | 0.240 | underpowered |
| `twin` k=8 eig | +0.084 | [−0.081, +0.247] | 1.000 | 0.238 | underpowered |
| `twin` k=0 | +0.078 | [−0.074, +0.236] | 1.000 | 0.226 | underpowered |

The largest delta has a lower bound above zero, but its MDE is 0.238 against a
0.05 smallest effect of interest, and its Holm-adjusted p-value is 0.185. Under
the pre-declared rule it is **underpowered**, not supported. No configuration is
ranked or recommended. Fold-level numbers remain descriptive.

The cluster count is 88--96 animals depending on the configuration, not 110,
because within-animal pairing requires both the model and baseline predictions
for the same target/neuron pair.

## 5. Posterior design table

`reports/phase12/design_posterior.csv` gives, for each eligible target, the
minimum number of animals at which the posterior probability of reaching
reliability 0.5 is at least 0.8. Of 126 eligible targets, **31** reach the
criterion within 60 animals; the median requirement among those is **29 animals**
and the maximum is **59**. The other 95 are labelled
`not_reached_within_budget`. This is a planning calculation conditional on a
model that failed calibration, so it is **not** a sample-size guarantee and is
not used in the paper's recommendations.

## 6. Decision

1. Retain **0.560** as the coverage-aware descriptive ceiling.
2. Do not report a calibrated hierarchical interval.
3. Do not rank targets or recommend an animal budget from the posterior.
4. Record the failure as correction **C9** and as a negative result, not as a
   tuning problem.
5. Keep the Phase 11 moment estimate and Phase 8 statuses unchanged.
6. Report the cluster benchmark as underpowered; no model superiority claim.

The model is still useful as a sensitivity analysis and as a documented attempt:
it shows that an additive Gaussian hierarchical model with known sampling
variance does not fix the sparse-atlas calibration problem. The next model should
address the positive null bias directly, for example by modelling responder
detection and coverage as part of the generative process rather than conditioning
on the observed mask.

## 7. What is not claimed

- Not a causal effect, digital twin, foundation model, or mechanism.
- Not a calibrated reliability interval.
- Not a target ranking.
- Not a validated prospective animal budget.
- Not a superior benchmark configuration.
- Not a systematic review; the literature scan is rapid and scoped.

## 8. Reproduce

```bash
python scripts/phase12_hierarchy.py
python scripts/phase12_benchmark.py
python scripts/phase12_manifest.py
python scripts/audit_claims.py
python -m pytest tests/test_phase12_reports.py -q
```

The input table is `reports/phase3/randi_trial_table_phase3.csv.gz`, referenced
by SHA-256 in the manifest and not redistributed. Full commands, seeds and
software versions are in `reports/phase12/manifest.json`. The pre-run contract is
in [`PHASE12_ANALYSIS_PLAN.md`](PHASE12_ANALYSIS_PLAN.md); fitting details are in
[`METHODS_HIERARCHICAL.md`](METHODS_HIERARCHICAL.md).
