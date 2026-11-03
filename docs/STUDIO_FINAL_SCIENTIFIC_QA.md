# Final scientific QA

**Date:** 2026-11-04  
**Decision:** `SCIENTIFIC_CLAIMS: PASS` for the active Studio surface, with the
exploratory and non-causal boundaries preserved.

## 1. What the evidence supports

The active claim is deliberately narrow:

> Within five Kato 2015 `WT_NoStim` recordings, at the 175 overlapping test
> landmarks, using a training-only PCA-10 and a 100-step J5 innovation rollout,
> the full population VAR (B3) has lower H20 PCA-10 energy than the ten-factor
> common-state control (B5-10) by a weighted paired mean of
> `+0.264438` (9.79% of the B3 mean).

This is an **exploratory methodological reanalysis**. It is not a biological
mechanism, a causal effect, a digital twin, or an animal-level generalization.

## 2. Numerical boundaries

| Quantity | Value | Source |
| --- | ---: | --- |
| Weighted paired mean, B5 − B3 | `+0.2644379926683563` | `statistical_audit.json`; recomputed from `model_per_landmark.csv` |
| Conditional block-5 interval | `[+0.0749929, +0.4031165]` | same |
| Recording-resampling interval | `[-0.0408009, +0.4792966]` | same |
| Session t interval | `[-0.0640587, +0.5873742]` | same |
| Sessions positive | 4 / 5 | same |
| Sign-test one-sided p | `0.1875` | same |
| MSE difference | `+0.0545043`; interval `[-0.1574262, +0.2409041]` | same |
| Marginal 95% coverage, B3 / B5 | `0.9383 / 0.7246` | same; marginal per-dimension, not joint |
| Capacity ratio B3 / B5-10 | `7.9584–9.9486×` | `capacity_audit.json` |
| B5 training variance captured | `54.67–70.99%` | same |
| Ordered B4m split | `50.99%` common / `49.01%` residual | arithmetic, order-sensitive |

The recording-level interval includes zero. The sign test is not significant.
The H1 contrast reverses (`-0.063669` at one step), so the sign is
horizon-dependent. The test tail was reused. These are constraints, not
footnotes.

## 3. What changed in wording

- Capacity text now derives `7.96–9.95×` and the ordered `51/49` split from
  `capacity_audit.json`; the numbers are no longer hard-coded in the UI.
- Landmark counts and positive-session counts derive from `primary` metadata.
- PCA percentages are explicitly fractions of **total** training variance, with
  the PCA-10-relative fraction and the unshown remainder stated separately.
- The perturbation screen now displays the exact required banner:
  `MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.`
- Historical unresolved language in `STUDIO_REPRODUCIBILITY.md` is preserved in
  a marked historical section and followed by the root cause and correction.
- The 17 errata remain surfaced; no historical result file was rewritten.

## 4. Perturbation audit

The perturbation lab edits one native standardized coordinate at one origin,
then rolls B3 and B5 forward with **zero innovation**. It is not stimulation.

| Check | Result |
| --- | --- |
| Input standardization | training mean absolute max `8.5e-7`; training SD `0.99999905–1.00000072` |
| Edit | `+1.0` training z unit to `OLQVR` at `(0, 2690)` |
| First-step edited-neuron response | `0.36780528025472614` |
| Expected `A[j,j]` | `0.3678052802547261` |
| First-step population norm | `1.115644553000671` |
| Expected `||A[:,j]||` | `1.115644553000671` |
| B3 spectral radius | `0.9950000000000021` (cap applied; below 1) |
| B5 factor radius | `0.9927329318516898` |
| Training-marginal OOD at +1 | no |
| H20 B3−B5 disagreement | `0.3491925968779753` |

The exact first-step equality confirms that the edit enters the model at the
selected coordinate and propagates through the fitted `A` column. It does not
confirm a biological effect. The B3/B5 disagreement demonstrates model
dependence; agreement would not have validated the edit.

The B5 path keeps the latent factor trajectory fixed and changes only the
edited own-coordinate state. This is the model's stated form; it is not a
recomputed PCA projection of an intervened observation.

## 5. Claim tiers and unresolved items

The active surface uses only `SUPPORTED_EXPLORATORY`, `LIMITED`, `UNSUPPORTED`,
`INVALIDATED`, and `NOT_EVALUABLE`. There is no `CONFIRMED` tier.

Remaining limitations:

- five recordings from one laboratory/preparation, with an inferred animal
  mapping and no independent cohort;
- 175 overlapping landmarks are not 175 animals;
- the test tail was reused and the split is not a fresh confirmatory test;
- B3/B5 capacity is not matched;
- marginal coverage is not joint calibration;
- the residual mechanism is unidentified;
- virtual perturbations have no ground truth, sham, or dose;
- the Studio is not a digital twin.

## 6. Frozen-manifest validation status

The scientific validators were run after the code changes:

| Command | Result |
| --- | --- |
| `validate_kato_common_state.py` | PASS, 26 hashed artifacts |
| `validate_reports.py --strict` | PASS |
| `validate_kato_final_audit.py` | FAIL on `.gitattributes` size/hash |
| `validate_kato_interactions.py` | FAIL on `app/handoff_demo.py` and `mkdocs.yml` hashes |

These are **frozen-manifest drift failures**, not numerical model failures. The
final-audit manifest predates the Studio's `.gitattributes` rules; the
interaction manifest predates the Studio handoff edit and this audit's MkDocs
nav addition. The frozen manifests were deliberately not rewritten to make the
failures disappear. The active numerical check, the frozen per-landmark rows,
and the Studio artifact agree; the stale metadata manifests remain a
provenance caveat for the supervisor.

**Overall status:**
`READY_FOR_METHODOLOGICAL_SUPERVISOR_REVIEW; NOT_READY_FOR_CONFIRMATORY_OR_BIOLOGICAL_CLAIMS`.
