# Master Review: Final Verdict and Readiness

**Branch:** `research/kato-final-reproducibility-audit`  
**HEAD:** `96a8dd333643de0564247e2c01c98a85511dac82`  
**Base:** `358107a7a55470d48f8092b2a2dc7f704dbd37c1`  
**Audit date:** 2026-11-03  
**Overall gate:**

```text
READY_FOR_METHODOLOGICAL_SUPERVISOR_REVIEW
NOT_READY_FOR_CONFIRMATORY_OR_BIOLOGICAL_CLAIMS
```

This is not an automatic “ready for Denis.” It is a statement about **what kind of conversation the work can support**.

---

## 1. The verdict in one paragraph

The archive is the right Kato `WT_NoStim` object. The splits and landmarks are in bounds. The B3/B5 arithmetic reproduces independently. The frozen result is positive in the sample: B3 has lower H20 PCA-10 energy than B5-10 by `+0.2644`, with a conditional block-5 interval `[+0.075, +0.403]`. But the design cannot carry the claim that is usually built on top of it. The five recordings are the biological units; a bootstrap that resamples them gives `[-0.041, +0.479]`. The 175 landmarks are overlapping windows, not independent replicates. The energy gain is null for mean MSE and coexists with a large coverage difference. B3 has `7.96–9.95×` more parameters, and no capacity-matched control exists. The test block was reused, the sign rule was clarified with the results, and the rolling models refit on test-past. The result is a useful exploratory predictive contrast, not a confirmatory, animal-level, or biological finding.

---

## 2. What is verified

These are supported by direct inspection, executable bounds checks, independent arithmetic, or frozen artifacts.

| Item | Evidence |
|---|---|
| File identity | size `56,390,975`; SHA256 `7531811c…a0098a`; OSF hash match |
| Five recordings, 15,662 frames | HDF5 counts |
| 629 traces, 592 predictors, 192 named occurrences, 58 unique names | HDF5 + label exclusion |
| Per-recording split bounds | `frame_landmark_bounds.json` |
| No split overlap, 12-frame purge | exact bounds |
| H1–H100 targets inside recording and test mask | bounds tests |
| No padding/interpolation/resampling | target bounds |
| H100 landmarks 35/35/34/38/33 | recomputed |
| Frozen landmarks match code | exact comparison |
| B3/B5 arithmetic | independent implementation, max abs error `7.46e-7` |
| Positive B3 energy contrast in sample | frozen table, recomputed |
| H1 reversal, session-2 reversal | frozen per-horizon/session table |
| MSE null | frozen table, interval crosses zero |
| Coverage difference | frozen per-landmark table |
| Off-diagonal instability | `r = 0.052` across segments |
| Structural gate failure | G3/G7 audit |
| Clean rerun reproducibility | max abs difference `4.44e-16`, empty stderr |
| Documentation defects | direct comparison to archive |

“Verified” here means **the number or fact is what the archive/code says it is**. It does not mean the scientific interpretation is correct.

---

## 3. What is unresolved

| Question | Why unresolved | What would resolve it |
|---|---|---|
| Do the five acquisition IDs map 1:1 to five publication animals? | No animal key in OSF; publication says five animals; acquisition IDs are distinct | Author metadata, lab notebook, or a keyed release |
| Is the residual advantage information or capacity? | B3 has `7.96–9.95×` more parameters; no matched model | Capacity-matched restricted model |
| Is the energy gain calibration or resolution? | B5 95% marginal coverage `0.725`; no decomposition | Spread-matched comparison / proper-score decomposition |
| Does the effect generalize across animals? | Recording bootstrap `[-0.041, +0.479]`; sign test `p=0.1875` | Independent cohort, animal-level design |
| Is the off-diagonal operator stable? | Cross-segment `r=0.052`; G7 fail | More data, state-stratified fits, identifiability design |
| Is the effect robust to representation? | Neuron energy and MSE intervals cross zero | Prespecified representation, replication |
| Is there a common biological drive? | Ten PCA factors are statistical, not measured | Measured behaviour/sensory/neuromodulatory input |
| Is the model calibrated? | Marginal coverage only; no joint calibration | Joint calibration diagnostics, recalibration |
| Is the result confirmatory? | Test reused; sign rule clarified with results | New preregistered cohort |
| Is there a biological intervention effect? | No stimulation/sham data | Matched intervention experiment |

The first three are the ones that matter most for the scientific claim. The last two are the ones that matter most for a biological claim.

---

## 4. What requires correction

The frozen historical artifacts were not rewritten. The corrections are registered in [`reports/master_review/corrections.json`](../reports/master_review/corrections.json) and summarized here.

### High severity

| ID | Was | Correct |
|---|---|---|
| X01 | `DATA_PROVENANCE.md` size `128,511,040` | `56,390,975` |
| X02 | “Five repeated sessions, not five animals” | Five animals inferred from publication + acquisition IDs; one cohort; no animal key |
| X05 | “G4 fully prespecified confirmatory PASS” | Numerical pass under corrected sign; confirmatory `NOT_ESTABLISHED` |
| X07 | “B3 general forecast improvement” | H20 PCA-10 energy only; H1 reverses; MSE null |
| X08 | “175 independent replicates” | Overlapping windows; ~25 non-overlapping H100 proxies; ~7× inflation |
| X09 | “Test tails untouched” | Reused across earlier phases; exploratory |
| X10 | “Probabilistic calibration established” | Not established; B5 `0.725`, B3 `0.938` marginal |
| X11 | Block-20 interval as robust precision | Conditional; recording bootstrap `[-0.131, +0.440]` |
| X12 | “Residual advantage isolates cross-neuron information” | Not identifiable from capacity |
| X16 | “Test frames touched once” | Rolling fits use validation and earlier test frames; no future leakage, but online-refit estimand |

### Medium severity

| ID | Was | Correct |
|---|---|---|
| X03 | “175 landmarks (35 per session)” | 35, 35, 34, 38, 33 |
| X04 | Common test boundary `2689:3137` | Session 0 only; per-recording bounds |
| X06 | 51/49 as a unique partition | Ordered B4m→B5→B3 decomposition; 23.5/76.5 vs B1m; 72.4/27.6 vs B0 |
| X15 | “Independent cohort” | One laboratory cohort; not independent replication |
| X17 | “95% coverage = joint calibration” | Average marginal per-dimension coverage |

### Low severity

| ID | Was | Correct |
|---|---|---|
| X13 | `stateParams` rows align with traces | 109/109, 135/107, 131/131, 125/126, 129/129; field unused |
| X14 | Fields `IDs`, `timeVectorSeconds` | HDF5 `NeuronNames`, `tv` |

The old documents remain in place because they are frozen research artifacts. They should receive an errata banner or be superseded by the master review. A reader who opens only `ADAPTIVE_DYNAMICS_MODELS.md`, `HANDOFF_METHODS_AND_LIMITATIONS.md` or `DATA_PROVENANCE.md` will still encounter false statements. That is a documentation risk, not a data risk.

---

## 5. What is exploratory

The following are `SUPPORTED_EXPLORATORY` at best:

1. B3 has lower H20 PCA-10 energy than B5-10 in the frozen sample.
2. B3 is better in four of five recordings.
3. The conditional block-5 interval excludes zero.
4. The effect grows with horizon from H1 to H20.
5. The ordered decomposition gives ~51% common / ~49% residual under the B4m baseline.

“Exploratory” means:

- the numbers are real in this sample;
- the direction is stable enough to discuss;
- the design cannot carry confirmatory or animal-level weight;
- the result is a hypothesis for a stronger study, not a finding to publish as settled.

It does **not** mean “probably true.” It means “worth one careful follow-up.”

---

## 6. What requires new experiments

### Existing data, new analysis

1. **Capacity-matched restricted model.** Fit a B3 variant with a rank-`k` residual operator and a parameter budget matched to B5. This directly tests whether the residual advantage is information or capacity.
2. **Calibration-matched ensemble.** Match marginal spread or decompose the energy score into calibration and resolution. This tests whether the energy gain is a spread effect.
3. **Non-overlapping or blocked evaluation.** Use one landmark per non-overlapping H100 window, or a blocked cross-validation that respects dependence. This tests whether the effect survives a dependence-aware design.
4. **State-stratified analysis.** Fit or evaluate within behaviourally homogeneous periods. This tests whether the residual is a behaviour/state effect.
5. **Identifiability-aware operator comparison.** Use the synthetic common-input and interaction scenarios to calibrate what the B3/B5 contrast can and cannot detect.

These are the right next analyses. They use existing data and do not require a new animal.

### New data

6. **Independent cohort.** New `WT_NoStim` animals, animal-level splits, prespecified sign/horizon/metric. Five animals are not enough for a stable animal-level estimate; 10–20 is a reasonable minimum.
7. **Measured behaviour.** Concurrent behaviour or a clean behavioural regressor, so the common-factor control can include it.
8. **Measured common input.** Sensory, neuromodulatory or arousal variable, manipulated or observed.
9. **Intervention with sham.** Cell-specific stimulation/ablation with matched sham and a prespecified target. This is required for any biological claim.
10. **Cross-animal alignment.** NeuroPAL or equivalent, so identified neurons can be compared across animals.

### What is not a new experiment

- More ensemble members.
- More bootstrap replicates.
- More horizons on the same test block.
- A larger VAR on the same five animals.
- A virtual perturbation with a larger displacement.

These can improve precision or description. They cannot create animal-level generalization or causal validity.

---

## 7. Claims appropriate to communicate

### To a supervisor

> The pipeline is reproducible and the B3/B5 contrast is positive in the five Kato recordings. The result is exploratory: the animal-level interval includes zero, the effect is null for MSE, B3 has about eight to ten times more parameters, B5 is undercovered, and the test block was reused. I would like to discuss a capacity-matched and calibration-aware follow-up before deciding whether this is a thesis result or a methods result.

### To a lab meeting

> We have a small, horizon-dependent energy advantage for the full population model over a ten-factor control. It is not a mean-accuracy effect, not animal-level, and not capacity-controlled. The most useful next step is a capacity-matched ablation.

### In a thesis

> In five Kato `WT_NoStim` recordings, B3 had lower H20 PCA-10 energy than B5-10 (`+0.2644`; conditional block-5 interval `[+0.075, +0.403]`; 64% of 175 overlapping landmarks). The effect was positive in four of five sessions but not established at the animal level (recording bootstrap `[-0.041, +0.479]`; sign test `p=0.1875`), was absent for mean MSE, reversed at H1, and was confounded with ensemble calibration and estimator capacity. The result is exploratory and motivates a capacity-matched, animal-level follow-up.

### In a paper abstract

Only the thesis paragraph, with the limitations in the abstract, not just the discussion. If the journal requires a positive claim, this is not the paper.

### To a biological audience

> This is a predictive result, not a mechanism. The data cannot identify direct coupling, a command circuit, or an intervention target. Biological intervention validity is `NOT_EVALUABLE`.

---

## 8. Claims that must not be communicated

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

Each is false at the level of the available evidence, even if a nearby weaker sentence is true.

---

## 9. What the student must learn

1. **A reproducible number is not a validated claim.** The clean rerun, the 457 tests and the `1e-16` match prove arithmetic, not inference.
2. **The unit of inference is the animal, not the landmark.** Five animals cannot be turned into 175 independent replicates by overlapping windows.
3. **A proper score is not a mean-accuracy score.** Energy rewards calibration and sharpness; MSE is the mean metric. They can disagree, and here they do.
4. **A capacity difference is a confound.** A model with `8–10×` more parameters can win without containing the right structure.
5. **A restriction is not an identification.** B4m/B5 are clean restrictions, but they do not isolate a biological mechanism.
6. **An unstable operator is not a connectome.** Cross-segment `r = 0.052` is a warning, not a detail.
7. **A virtual perturbation is not an intervention.** Model-dependent displacement cannot substitute for stimulation/sham data.
8. **Documentation is part of the evidence.** Wrong sizes, wrong boundaries and wrong animal wording change how readers interpret the result.
9. **A negative result can be the most honest result.** The H1 reversal, the MSE null and the failed structural gate are part of the finding.
10. **Say “I do not know” when you do not.** The alternative is inventing a mechanism.

---

## 10. The 12 deliverables

| # | Document | Purpose |
|---:|---|---|
| 1 | [Overview](MASTER_REVIEW_OVERVIEW.md) | Entry point and summary |
| 2 | [Data forensics](MASTER_REVIEW_DATA_FORENSICS.md) | Archive identity, dimensions, bounds |
| 3 | [Research history](MASTER_REVIEW_RESEARCH_HISTORY.md) | How the claim evolved, including dead ends |
| 4 | [Code audit](MASTER_REVIEW_CODE_AUDIT.md) | Equations, causality, findings |
| 5 | [Statistics](MASTER_REVIEW_STATISTICS.md) | Effect size, dependence, intervals, capacity |
| 6 | [Biological interpretation](MASTER_REVIEW_BIOLOGICAL_INTERPRETATION.md) | What can and cannot be said about the worm |
| 7 | [Literature comparison](MASTER_REVIEW_LITERATURE_COMPARISON.md) | Kato, BunDLe-Net, NC-MCM, intervention work |
| 8 | [Claims ledger](MASTER_REVIEW_CLAIMS_LEDGER.md) | 23 claims with allowed statuses |
| 9 | [Peer review](MASTER_REVIEW_PEER_REVIEW.md) | Skeptical reviewer report |
| 10 | [Oral defense](MASTER_REVIEW_ORAL_DEFENSE.md) | 20 questions, traps, rehearsal |
| 11 | [Educational course](MASTER_REVIEW_EDUCATIONAL_COURSE.md) | 16 chapters with exercises |
| 12 | **Final verdict** (this document) | Disposition and next steps |

Machine-readable evidence is in [`reports/master_review/`](../reports/master_review/): 17 JSON/CSV artifacts, a manifest, and logs. The prior audit package in `reports/simulator/kato_final_audit/` and `docs/KATO_FINAL_AUDIT_*` is preserved.

---

## 11. Quality of this audit

### What this audit did

- Verified branch and HEAD.
- Verified file identity, hash and dimensions.
- Recomputed splits and H1–H100 landmark bounds.
- Audited the exact model/metric code path.
- Reimplemented B3/B5 at three landmarks.
- Recomputed effect sizes, intervals, calibration, capacity and decomposition from frozen artifacts.
- Ran a recording-level cluster bootstrap.
- Built a 23-claim ledger and a 17-item correction register.
- Wrote 12 documents and a 16-chapter course.
- Preserved the prior audit package.
- Did not rewrite history, push, or modify frozen artifacts.

### What this audit did not do

- It did not refit all 175 landmarks in a new independent pipeline. It reimplemented three landmarks and recomputed the rest from the frozen per-landmark table.
- It did not obtain the publisher’s full methods text (HTTP 403/CAPTCHA). Cohort and protocol statements rely on the publication record and secondary descriptions.
- It did not run a new biological experiment.
- It did not run a systematic literature review.
- It did not resolve whether the five acquisition IDs map 1:1 to publication animals.
- It did not run a capacity-matched or calibration-matched model.

These are real limits. They are why the verdict is “methodological review,” not “scientific sign-off.”

---

## 12. Final disposition

**If “ready for Denis” means ready for a methodological conversation about a reproducible, honestly labelled exploratory result:** yes. The student can now explain the result, the uncertainty, the confounds and the next experiment.

**If “ready for Denis” means ready for final scientific sign-off, a confirmatory claim, a biological mechanism, or a publication abstract:** no. The animal-level interval includes zero, the capacity and calibration confounds are unresolved, the test block was reused, and there is no intervention.

The honest formulation is:

> The project is ready to be reviewed as a methods-heavy exploratory study. It is not ready to be accepted as a confirmatory or biological result. The next conversation should be about evidential standards and the capacity-matched follow-up, not about whether the current number is “significant.”

That is the verdict. It is narrower than the project once hoped, and stronger than a claim the data cannot carry.
