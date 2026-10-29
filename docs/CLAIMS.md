# Claims policy and ledger

NeuroBranch distinguishes four claim statuses. The machine-readable ledger is
`reports/phase9/claims.json`, validated by `schemas/claims.schema.json`.

## Status definitions

| Status | Meaning |
| --- | --- |
| `supported` | The interval excludes the null and the effect reaches the design's MDE. The claim is empirically supported **within this design**; it is not proof. |
| `not_supported` | The interval includes the null, or the estimate is below the MDE, or a preregistered gate failed. |
| `bounded_null` | The interval lies inside the declared equivalence margin, so effects larger than the margin are ruled out. This is an informative null, not a failure. |
| `underpowered` | The design's MDE exceeds the smallest effect of interest. The study cannot resolve the question it asked, whatever the point estimate. |

A status is always attached to a specific statement, dataset, metric and
design. It does not transfer to other datasets, other response summaries or
other species.

## Decision order

For superiority claims:

1. `underpowered` if MDE > SESOI.
2. `supported` if the interval excludes zero and the estimate reaches the MDE.
3. `bounded_null` if the interval lies inside ±SESOI.
4. `not_supported` otherwise.

For bound claims ("reliability is below 0.5"): `supported` iff the interval
lies entirely on the claimed side of the bound.

## Current ledger

| ID | Statement | Estimate | 95% CI | Status |
| --- | --- | ---: | --- | --- |
| `phase9-single-animal-ceiling-below-half` | Single-animal reliability < 0.5 | 0.223 | [0.196, 0.251] | supported |
| `phase9-repeat-ceiling-below-half` | Repeat reliability < 0.5 | 0.243 | [0.206, 0.279] | supported |
| `phase8-pair-advantage` | APT(k=0) beats static CRO at pair level | +0.0442 | [+0.0080, +0.0901] | underpowered (MDE 0.0587 > SESOI 0.020) |
| `phase8-trajectory-advantage` | APT(k=0) beats per-responder template | −0.0000066 | — | not supported |
| `phase8-sequential-rollout` | 3-event rollout beats persistence | +0.0230 | — | not supported (coverage 0.904 < 0.95) |
| `phase8-eig-selection` | EIG selection beats random at k=8 | −0.0038 | [−0.0070, +0.0026] | bounded null (±0.01) |
| `phase7-cro-share-of-ceiling` | CRO reaches 64% of the ceiling | 0.0984 | — | not supported (superseded) |

## What is never claimed

- That the virtual perturbation is causal.
- That the model is a validated digital twin.
- That the model is a foundation model of *C. elegans*.
- That a supported claim generalises beyond the atlas and response window.
- That a rapid literature scan is a systematic review.

## Regenerating the ledger

```bash
python scripts/phase9_reliability.py
python scripts/phase9_claims.py --check
python scripts/validate_reports.py --strict
```

`--check` fails if an evidence path in the ledger does not exist. CI validates
the ledger against the schema on every push.

## Adding a claim

1. Add the estimate and interval to a report under `reports/`.
2. Declare the SESOI (margin) before looking at the result.
3. Use `classify_superiority` or `classify_bound`; do not write a status by hand.
4. Add the claim in `src/neurobranch/benchmark/claims.py` with an evidence path.
5. Regenerate and validate.
