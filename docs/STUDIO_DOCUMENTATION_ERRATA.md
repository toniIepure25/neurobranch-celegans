# Documentation errata and active-surface treatment

The Master Review found 17 corrections. Frozen historical artifacts are
**preserved**, not rewritten. This page records:

1. what the old surface said;
2. what is correct;
3. where the current Studio stops repeating the error;
4. what scientific limitation remains after the wording is fixed.

The machine-readable source is
`reports/master_review/corrections.json`; the Studio reads it directly and adds
the active-surface mapping in `src/neurobranch/studio/errata.py`.

| ID | Old surface | Correction | Active Studio surface | Status / remaining limit |
| --- | --- | --- | --- | --- |
| X01 | `DATA_PROVENANCE.md` size 128,511,040 | 56,390,975 bytes; SHA256 matches OSF | Overview provenance | corrected; documentation was stale |
| X02 | "five sessions, not five animals" | five distinct acquisition preparations; publication n=5; OSF has no animal key | Overview; handoff demo | corrected with inferred linkage; one-to-one mapping not documented |
| X03 | "175 test landmarks (35 per session)" | 35, 35, 34, 38, 33 | landmark selector and tests | corrected |
| X04 | common test boundary 2689:3137 | session 0 only; per-recording boundaries | activity split overlay | corrected |
| X05 | G4 confirmatory PASS | numerical pass under corrected sign; confirmatory NOT_ESTABLISHED | evidence status | corrected; still no confirmatory claim |
| X06 | 51/49 as a unique partition | ordered B4m→B5→B3 decomposition; 23.5/76.5 vs B1m; 72.4/27.6 vs B0 | evidence capacity | corrected; split is arithmetic |
| X07 | general B3 forecast improvement | H20 PCA-10 energy only; H1 reverses; MSE null | overview, models, evidence | corrected; horizon-specific |
| X08 | 175 landmarks as independent replicates | overlapping windows; ~25 non-overlap proxies; ~7× inflation | evidence inference unit | corrected; effective n is small |
| X09 | test tails untouched | reused across earlier phases; final audit exploratory | overview banner | disclosed; not confirmatory |
| X10 | calibration established | B5 95% marginal 0.725; B3 0.938 | evidence calibration | corrected; not established |
| X11 | block-20 `[+0.092, +0.294]` as robust precision | conditional interval; recording interval `[−0.131, +0.440]` | evidence forest | corrected; intervals separated by unit |
| X12 | residual isolates cross-neuron information | not identifiable from capacity; B3 7.96–9.95× parameters | models; evidence | disclosed; mechanism unidentified |
| X13 | `stateParams` rows align with trace rows | 109/109, 135/107, 131/131, 125/126, 129/129; unused | provenance | corrected |
| X14 | fields `IDs`, `timeVectorSeconds` | actual HDF5 `NeuronNames`, `tv` | provenance | corrected |
| X15 | five recordings are an independent cohort | one laboratory/preparation; no independent cohort | overview footer | corrected |
| X16 | test frames touched once | online refit at every landmark includes validation and earlier test frames; no future leakage | models refit note | disclosed; online-refit estimand |
| X17 | 95% coverage means joint 10-D calibration | average marginal per-dimension coverage | evidence calibration | corrected; not joint |

## Status summary

| Status | Count |
| --- | ---: |
| `CORRECTED_IN_ACTIVE_SURFACE` | 13 |
| `CORRECTED_WITH_INFERRED_LINKAGE` | 1 |
| `DISCLOSED_AS_EXPLORATORY` | 1 |
| `DISCLOSED_AS_UNIDENTIFIED` | 1 |
| `DISCLOSED_AS_ONLINE_REFIT` | 1 |
| **Total** | **17** |

The Studio renders these statuses in **Research Notebook → Documentation
errata**, with the old wording and the correction side by side. A status of
"corrected" means the active surface no longer repeats the error; it does not
mean the underlying scientific question is resolved.

## Unresolved scientific findings (not documentation defects)

These remain true after every wording correction and must travel with the
result:

- **Capacity is not matched.** B3 has 7.96–9.95× the nominal coefficients and
  arbitrary residual coordinates. The residual advantage is not identifiable as
  cross-neuron information.
- **The test tail is reused.** The analysis is exploratory, not confirmatory.
- **The refit is online.** The estimand includes earlier test frames in the
  design; there is no future leakage, but this is not a frozen holdout.
- **Animal-level generalization is not established.** Five recordings from one
  preparation, with no primary animal key.
- **Calibration is not established.** Marginal coverage is not joint
  calibration; B5 is severely undercovered.
- **The 3D space is not anatomy.** PCA components are statistical coordinates.
- **Virtual perturbations are not interventions.** They are model
  counterfactuals with no biological effect size.

## Historical documents

The old values remain in the historical Master Review artifacts and in some
legacy dashboards. They are not silently edited. The current hierarchy is:

1. machine-readable audit artifacts under `reports/master_review/` and
   `reports/simulator/`;
2. the Master Review documents under `docs/MASTER_REVIEW_*`;
3. the Studio documents in this directory;
4. legacy dashboards, which are explicitly labelled Advanced/Legacy.

If a legacy dashboard disagrees with the Studio, the audited artifact wins.
