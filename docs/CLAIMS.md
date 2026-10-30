# Claims policy and ledger

NeuroBranch distinguishes four claim statuses. The machine-readable ledger is
`reports/phase10/claims.json`, validated against `schemas/claims.schema.json`
(schema v2). Historical Phase 9 ledgers use
`schemas/claims_v1.schema.json`.

## Status definitions

| Status | Meaning |
| --- | --- |
| `supported` | The interval excludes the null and the effect reaches the design's MDE. The claim is empirically supported **within this design**; it is not proof. |
| `not_supported` | The interval includes the null, or the estimate is below the MDE, or a preregistered gate failed. |
| `bounded_null` | The interval lies inside the declared equivalence margin, so effects larger than the margin are ruled out. This is an informative null, not a failure. |
| `underpowered` | The design's MDE exceeds the smallest effect of interest. The study cannot resolve the question it asked, whatever the point estimate. |

A status is always attached to a specific statement, dataset, metric, estimand
and design. It does not transfer to other datasets, response summaries or
species.

## Estimands (schema v2)

Every claim declares the facet it describes:

- `single_animal` - a profile from one animal; includes animal-specific
  deviation.
- `repeat` - repeated stimulation of the same animal.
- `pooled_fold` - a profile averaged over the animals in an evaluation fold.
- `animal_cluster` - a paired model difference whose resampling unit is the
  animal, used by the Phase 12 benchmark inference.
- `superiority` / `equivalence` - a difference between models or conditions.
- `descriptive` - arithmetic from fitted components, with no sampling claim.

`n_eval_animals` records the number of animals pooled. A claim without an
estimand is invalid under schema v2.

## Decision order

For superiority claims:

1. `underpowered` if MDE > SESOI.
2. `supported` if the interval excludes zero and the estimate reaches the MDE.
3. `bounded_null` if the interval lies inside ±SESOI.
4. `not_supported` otherwise.

For bound claims ("reliability is below 0.5"): `supported` iff the interval
lies entirely on the claimed side of the bound.

## Multiplicity

Phase 8 inferential claims form the family `phase8-inferential`. Two-sided
normal p-values are derived from the reported estimates and intervals, and
Holm adjustment is applied within the family. Adjusted values are stored in
`p_holm`; the unadjusted value is retained in `p_value`. Descriptive claims
carry no p-value and are not in a family.

## Current ledger

| ID | Statement | Estimand | Estimate | 95% CI | Status |
| --- | --- | --- | ---: | --- | --- |
| `phase9-single-animal-ceiling-below-half` | Single-animal reliability < 0.5 | single_animal | 0.162 | [0.134, 0.191] | supported |
| `phase9-repeat-ceiling-below-half` | Repeat reliability < 0.5 | repeat | 0.243 | [0.206, 0.279] | supported |
| `phase10-fold-ceiling` | Phase 10 sparse-coverage ceiling is about 0.83; the 0.56 value is historical and extrapolated | pooled_fold | 0.833 | descriptive | not supported (coverage-sensitive; C8/C15) |
| `phase8-pair-advantage` | APT(k=0) beats static CRO at pair level | superiority | +0.0442 | [+0.0080, +0.0901] | underpowered (MDE 0.0587 > SESOI 0.020; Holm p = 0.070) |
| `phase8-trajectory-advantage` | APT(k=0) beats per-responder template | superiority | −0.0000066 | — | not supported |
| `phase8-sequential-rollout` | 3-event rollout beats persistence | superiority | +0.0230 | — | not supported (coverage 0.904 < 0.95) |
| `phase8-eig-selection` | EIG selection beats random at k=8 | superiority | −0.0038 | [−0.0070, +0.0026] | bounded null (±0.01) |
| `phase7-cro-share-of-ceiling` | CRO reaches 64% of the ceiling | descriptive | 0.0984 | — | not supported (superseded; coverage-aware share 17.6%) |
| `phase12-hierarchical-interval` | Hierarchical posterior interval is calibrated for pooled reliability | pooled_fold | 0.3035 | — | not supported (SBC rank KS p = 1.43e-06; PPC MAE 0.207; negative-control median R(22) = 0.120, q95 0.594; C10) |
| `phase13-event-variance-model` | Projection-consistent event-variance model is calibrated for pooled reliability | pooled_fold | 0.2003 | — | not supported (SBC coverage 0.9244; rank KS p = 0.00306; negative-control median 0.1485, q95 0.4680; sign-flip median 0.1341; PPC not run; 0.1137 below reference) |
| `phase14-zero-inflated-model` | Exact zero/spike-and-slab model is calibrated for pooled reliability | pooled_fold | 0.0543 | — | not supported (8/12 gates failed: SBC rank p = 3.35e-192; edge 0.0810; PPC coverage 0.7697, MAE 0.3998; null q95 0.3535/0.4509; prior range 0.1220; reference gap 0.2463; spike calibration 0.3561) |
| `phase15-observed-reliability` | Observed-count reliability is descriptive, not a calibrated bound | R_obs | 0.2606 | [0.1782, 0.3430] | not supported (jackknife coverage 0.990; null median 0.096, q95 0.114; C15) |
| `phase15-fold-counterfactual` | R(22) is counterfactual for most targets, not identified | R_22 | 0.3006 | [0.2087, 0.3924] | not supported (87/126 extrapolated; extrapolation MAE 0.204; null median 0.122; C15) |
| `phase15-identified-subset` | R_id is observed-count reliability on the identified subset | R_id | 0.3501 | [0.2020, 0.4982] | not supported (39 targets; different population; null median 0.107; C15) |
| `phase15-ranking-stability` | Target rankings replicate across animal halves | ranking | 0.126 | [−0.021, 0.247] | not supported (top-10 Jaccard 0.176; no ranking published; C15) |
| `phase12-benchmark-cluster` | A benchmark configuration beats static CRO by >0.05 in the animal-cluster analysis | animal_cluster | — | — | not supported (withdrawn: baseline/model `y_true` mismatch in 1,334/1,334 paired rows; C11) |

## What is never claimed

- That the virtual perturbation is causal.
- That the model is a validated digital twin.
- That the model is a foundation model of *C. elegans*.
- That a supported claim generalises beyond the atlas and response window.
- That a rapid literature scan is a systematic review.
- That the animal jackknife interval is a calibrated confidence interval.
- That the Phase 12 hierarchical posterior interval is calibrated, or that
  0.3035 replaces a 0.560 bound (0.560 was extrapolated; C15).
- That the Phase 13 event-variance model is calibrated, or that 0.2003 is a
  measurement limit. It failed its gates.
- That 0.560 is a bound, or that `R_22 = 0.3006` is identified for all targets.
  It is counterfactual for 87 of 126; use `R_obs = 0.2606` for observed counts
  and `R_id = 0.3501` for the identified subset (C15).
- That the event-variance model's lower estimate proves the true reliability is
  near 0.20; the negative-control median 0.1485 shows positive null bias.
- That any Phase 12 benchmark configuration is superior; the animal-cluster
  comparison is **withdrawn** because baseline and model truth were not paired
  on the same events (C11).

## Regenerating the ledger

```bash
python scripts/phase10_reliability.py
python scripts/phase10_claims.py --check
python scripts/phase12_hierarchy.py
python scripts/phase12_benchmark.py
python scripts/phase12_manifest.py
python scripts/validate_reports.py --strict
```

`--check` fails if an evidence path in the ledger does not exist. CI validates
the ledger against the schema on every push.

## Adding a claim

1. Add the estimate and interval to a report under `reports/`.
2. Declare the estimand and, for differences, the SESOI before looking at the
   result.
3. Use `classify_superiority` or `classify_bound`; do not write a status by hand.
4. Add the claim in `src/neurobranch/benchmark/claims.py` with an evidence path.
5. Regenerate and validate.
