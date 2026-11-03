# Master Review: Claims Ledger

**Authoritative machine-readable file:** [`reports/master_review/claims_ledger.json`](../reports/master_review/claims_ledger.json).  
**Rule:** no claim is labelled `CONFIRMED`. The five permitted tiers are `SUPPORTED_EXPLORATORY`, `LIMITED`, `UNSUPPORTED`, `INVALIDATED`, `NOT_EVALUABLE`.

---

## 1. Tier semantics

| Tier | Meaning | What it permits |
|---|---|---|
| `SUPPORTED_EXPLORATORY` | Numerically supported in this dataset, under the stated analysis; not confirmatory and not generalized | “In these five recordings…” |
| `LIMITED` | Supported only under stated constraints, or a descriptive/archive fact bounded to this material | “The archive contains…”; “under the B4m ordering…” |
| `UNSUPPORTED` | The available evidence does not establish the claim | “We did not find…” |
| `INVALIDATED` | The claim is contradicted by the audit evidence | “The earlier statement is wrong.” |
| `NOT_EVALUABLE` | The required data or design is absent | “This cannot be tested with these data.” |

`LIMITED` is used for archive facts because the tier vocabulary has no `VERIFIED` label. It means “bounded to this material,” not “uncertain.”

---

## 2. Ledger

### Data and provenance

| ID | Claim | Status | Evidence | Why |
|---|---|---|---|---|
| **D01** | The analysed file is Kato `WT_NoStim` with SHA256 matching the OSF object. | `LIMITED` | `dataset_identity_audit.json`; OSF API | Direct hash/size verification; bounded to the archive. |
| **D02** | Five recordings contain 15,662 frames and 592 behaviour-label-excluded predictors. | `LIMITED` | `dataset_identity_audit.json` | Direct HDF5 count; predictor count excludes eight label-defining names. |
| **D03** | The five entries are five distinct animals, not five repeated sessions of one preparation. | `LIMITED` | `dataset_identity_audit.json`; Kato DOI; NC-MCM DOI | Publication reports five worms; OSF has no animal key; one acquisition per animal is inferred. One cohort, not independent replication. |
| **D04** | The eight behaviour labels are independent external measurements. | `INVALIDATED` | `DATA_PROVENANCE.md`; readme | Labels derive from command/interneuron calcium and behaviour. |
| **D05** | There are 35 H100 test landmarks in every recording. | `INVALIDATED` | `frame_landmark_bounds.json`; `ADAPTIVE_DYNAMICS_MODELS.md` | Actual counts 35, 35, 34, 38, 33. |
| **D06** | All recordings share the test boundary `2689:3137`. | `INVALIDATED` | `frame_landmark_bounds.json`; `HANDOFF_METHODS_AND_LIMITATIONS.md` | Session 0 only; code uses per-recording bounds. |
| **D07** | Every reported forecast target lies inside its recording and test mask, with no padding or interpolation. | `SUPPORTED_EXPLORATORY` | `frame_landmark_bounds.json`; bounds tests | Verified H1–H100; H5 session 1 reaches the last frame with zero margin. |

### Predictive result

| ID | Claim | Status | Evidence | Why |
|---|---|---|---|---|
| **S01** | B3 has lower H20 PCA-10 energy than B5-10 in the frozen 175-landmark sample. | `SUPPORTED_EXPLORATORY` | `statistical_audit.json`; `independent_model_check.json` | Mean `+0.2644`; conditional block-5 interval `[+0.0750, +0.4031]`; 64% positive; arithmetic reproduced. |
| **S02** | B3 is better in four of five recordings. | `SUPPORTED_EXPLORATORY` | `statistical_audit.json` | Session means `+0.300, +0.329, -0.193, +0.407, +0.465`. |
| **S03** | The B3 advantage generalizes across animals. | `UNSUPPORTED` | `statistical_audit.json` | Recording bootstrap `[-0.041, +0.479]`; t interval `[-0.064, +0.587]`; sign test `p=0.1875`. |
| **S04** | B3 improves the conditional mean forecast. | `UNSUPPORTED` | `statistical_audit.json` | PCA-10 MSE `+0.0545`, interval `[-0.157, +0.241]`; neuron MSE `-0.0354`, interval crosses zero. |
| **S05** | The B3 advantage is uniform across horizons and recordings. | `INVALIDATED` | `statistical_audit.json` | H1 reverses (`-0.0637`); session 2 reverses (`-0.1931`). |
| **S06** | The 51% common / 49% residual split is a unique structural partition. | `INVALIDATED` | `capacity_audit.json` | Ordered B4m→B5→B3 decomposition; B1m gives 23.5/76.5; B0 gives 72.4/27.6. |
| **S07** | The fitted off-diagonal operator identifies stable direct neuron-neuron interactions. | `UNSUPPORTED` | `KATO_INTERACTIONS_RESULTS.md`; identifiability audit | Off-diagonal cross-segment correlation `0.052`; G3 fails; common-input synthetic confound. |
| **S08** | A named-neuron virtual perturbation predicts a biological intervention. | `NOT_EVALUABLE` | `KATO_FINAL_AUDIT_RESULTS.md` | Model-dependent +1 SD displacement; response cosine `0.556`; no matched intervention data. |
| **S09** | The ensemble is probabilistically calibrated. | `UNSUPPORTED` | `statistical_audit.json` | B5 95% marginal coverage `0.725`; B3 `0.938`; joint calibration not established. |
| **S10** | G4 is a fully prespecified confirmatory PASS. | `INVALIDATED` | `KATO_FINAL_AUDIT_SIGN_CONVENTION.md`; final results | Sign rule clarified with results; numerical pass under corrected sign; confirmatory `NOT_ESTABLISHED`. |
| **S11** | The residual B3 advantage is information rather than estimator capacity. | `NOT_EVALUABLE` | `capacity_audit.json` | B3 has `7.96–9.95×` more parameters; no capacity-matched model. |
| **S12** | B5 undercoverage is a sufficient explanation of the energy advantage. | `UNSUPPORTED` | `statistical_audit.json` | Plausible contributor; no decomposition establishes sufficiency. |
| **S13** | The Kato test tails were untouched before the final audit. | `INVALIDATED` | Final audit scope; sign-convention doc | Earlier phases inspected the same block. |
| **S14** | Test frames are touched once, only for final metrics. | `INVALIDATED` | `METHODOLOGY.md`; rolling fit code | B3/B5 refit on validation and earlier test frames at every landmark. No future leakage, but online-refit estimand. |
| **S15** | The reported 95% coverage is joint 10-D calibration. | `INVALIDATED` | `coverage_summary` | It is average marginal per-dimension coverage. |
| **S16** | The result is ready for confirmatory publication or biological interpretation. | `UNSUPPORTED` | `quality_gate.json` | Exploratory; reused test; animal-level interval includes zero; calibration/capacity confounds; no intervention. |

---

## 3. Critical issues

| ID | Issue | Severity | Consequence | Required action |
|---|---|---|---|---|
| **I01** | Animal-level interval includes zero | Critical | Generalization claim unsupported | Report recording bootstrap/t interval; do not say “across animals” |
| **I02** | Test reuse and sign clarification | Critical | Confirmatory status not established | Label exploratory; prespecify a new cohort |
| **I03** | Capacity mismatch | High | Residual “information” not identified | Capacity-matched restricted model |
| **I04** | Calibration/spread confound | High | Energy advantage not a mean-accuracy result | Calibration-matched comparison or score decomposition |
| **I05** | H1 reversal / session 2 | High | Effect not uniform | Report horizon and session heterogeneity |
| **I06** | 175 overlapping landmarks | High | Precision overstated | Report effective units and non-overlap proxies |
| **I07** | Off-diagonal instability | High | Interaction/causal claim unsupported | Keep structural claim `UNSUPPORTED` |
| **I08** | No intervention | High | Biological claim not evaluable | Matched stimulation/sham, or no biological claim |
| **I09** | Documentation errors | Medium | Readers get false facts | Correct size, landmark count, boundary, animal wording |
| **I10** | Marginal vs joint coverage | Medium | Calibration overstated | Qualify every coverage number |
| **I11** | Numeric traces unaligned | Medium | Cross-animal cell claims impossible | Use identified subset or NeuroPAL alignment |
| **I12** | `stateParams` row mismatch | Low | Future join errors | Do not use without key verification |

---

## 4. Allowed sentences

These are safe, given the evidence:

1. “The archive is the Kato `WT_NoStim` object; its SHA256 matches OSF.”
2. “The five recordings contain 15,662 frames and 592 predictors after excluding eight behaviour-label names.”
3. “The five acquisition IDs are consistent with five animals from one cohort; the archive has no explicit animal key.”
4. “The code uses per-recording temporal splits with a 12-frame purge; all H1–H100 targets are in bounds.”
5. “In these five recordings, B3 has lower H20 PCA-10 energy than B5-10; the conditional block-5 interval is `[+0.075, +0.403]`.”
6. “The effect is positive in four of five sessions and reverses at H1.”
7. “The effect is not present for mean MSE; the MSE interval includes zero.”
8. “B3 has better marginal 95% coverage than B5, but neither is jointly calibrated.”
9. “B3 has 7.96–9.95× more parameters than B5; the contrast is not capacity-matched.”
10. “A bootstrap that resamples recordings gives an interval including zero.”
11. “The off-diagonal operator is unstable across segments; direct coupling is not identified.”
12. “No biological intervention validation is possible with these data.”

---

## 5. Forbidden sentences

These are not allowed in any abstract, slide, memo or thesis without new evidence:

- “B3 captures cross-neuron interactions.”
- “B3 identifies a functional connectome.”
- “The common factors are command neurons.”
- “The result generalizes across animals.”
- “The result is confirmatory.”
- “The test set was untouched.”
- “The model is calibrated.”
- “B3 improves forecast accuracy.”
- “The 51/49 split shows half the signal is common input.”
- “Virtual perturbation predicts the effect of stimulating neuron X.”
- “The model is a digital twin.”
- “B3 beats BunDLe-Net / NC-MCM.”
- “The five animals are an independent cohort.”

Each forbidden sentence fails at least one of: animal-level unit, capacity control, calibration, causal design, or literature comparability.

---

## 6. Claim-to-evidence trail

Every ledger entry maps to a machine-readable artifact:

| Artifact | Claims |
|---|---|
| `dataset_identity_audit.json` | D01–D03 |
| `frame_landmark_bounds.json` | D05–D07 |
| `provenance_checks.json` | D01, D05, D06 |
| `independent_model_check.json` | S01 |
| `statistical_audit.json` | S01–S06, S09, S12–S15 |
| `capacity_audit.json` | S06, S11 |
| `code_audit.json` | S07, S14, S15 |
| `corrections.json` | D05, D06, S10, S14, S15 |
| `quality_gate.json` | S16 |
| `KATO_INTERACTIONS_RESULTS.md` | S07–S08 |
| `KATO_FINAL_AUDIT_*` | S10, S13 |
| `LITERATURE_REVIEW.md` | literature comparison only |

The JSON ledger is authoritative. This Markdown table is a readable mirror; if they disagree, the JSON wins.

---

## 7. What “exploratory” means here

`SUPPORTED_EXPLORATORY` does not mean “probably true.” It means:

- the number is real in this sample;
- the direction is stable enough to report;
- the design cannot carry confirmatory weight;
- the result is worth a follow-up, not a victory lap.

It is a tier of **permission to discuss**, not a tier of **confidence**.

The project’s earlier `READY_FOR_SUPERVISOR_REVIEW` fits this tier: ready to discuss the evidence and its limits, not ready to assert a mechanism.

---

## 8. What the student must be able to say without notes

1. The five recordings are five animals inferred from the publication, one cohort, not independent replication.
2. The archive has 15,662 frames and 592 predictors; the dimensions are not 103 everywhere.
3. The landmark counts are 35/35/34/38/33, not 35 each.
4. The test boundaries are per-recording, not common.
5. B3 beats B5 on H20 PCA-10 energy by `+0.2644`, conditional interval `[+0.075, +0.403]`.
6. The recording-level interval includes zero; the sign test is not significant.
7. The MSE effect is null; the energy effect is partly calibration/spread.
8. B3 has 7.96–9.95× more parameters; capacity is unresolved.
9. H1 reverses; session 2 reverses; the effect is not uniform.
10. The off-diagonal operator is unstable; no direct coupling.
11. No intervention; biological validity `NOT_EVALUABLE`.
12. The result is exploratory, not confirmatory.

If the student cannot say these twelve sentences, the project is not ready for an oral defense.
