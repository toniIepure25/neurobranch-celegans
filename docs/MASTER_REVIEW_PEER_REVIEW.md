# Master Review: Skeptical Peer Review

**Manuscript under review:** the Kato common-state predictive audit and its final claims.  
**Reviewer:** independent computational neuroscientist / statistician.  
**Recommendation:** **MAJOR REVISION** for an exploratory methodological report; **REJECT** as a confirmatory mechanistic paper.  
**Confidence:** high on the numerical audit; moderate on the biological literature because the publisher’s methods text was not directly accessible.

---

## 1. Summary

The authors ask whether a full lag-one population VAR (B3) predicts H20 population activity better than an own-history-plus-ten-common-factor control (B5-10) in the Kato `WT_NoStim` data. They report a positive H20 PCA-10 energy contrast of `+0.2644`, a conditional block-5 interval `[+0.075, +0.403]`, and a 64% win rate over 175 landmarks. They are commendably explicit that the result is exploratory, that the test block was reused, and that biological intervention validity is not evaluable.

The numerical work is careful. The code is causal in the no-future sense. The independent arithmetic check passes. The archive identity is verified. The authors have also documented several of their own earlier errors.

The problem is that the inferential claim is still stronger than the design. The five recordings are the independent units; a bootstrap that resamples them includes zero. The 175 landmarks are overlapping windows, not independent replicates. The energy advantage coexists with a large calibration difference and a null MSE effect. B3 has 7.96–9.95× more parameters than B5. The test block was reused, and the sign rule was clarified in the same commit as the results. Under these conditions, the result is a useful exploratory signal, not a demonstrated population-information effect.

---

## 2. Strengths

1. **The archive identity is verified.** SHA256 matches OSF; the local object is the right file.
2. **The split and landmark code is sound.** Per-recording bounds, purge gaps, no padding, all targets in bounds.
3. **The arithmetic is reproducible.** An independent reference implementation matches B3/B5 energies to `7.46e-7`.
4. **The authors report negative results.** H1 reversal, session 2 reversal, MSE null, off-diagonal instability, failed structural gates.
5. **The authors distinguish reproduction from confirmation.** The clean rerun is not presented as a new experiment.
6. **The claims are tiered.** The final audit uses `SUPPORTED_EXPLORATORY`, not `CONFIRMED`.
7. **The correction log is unusually honest.** Several corrections are unfavourable to the project.

These are not cosmetic strengths. Many computational neuroscience papers would fail all seven.

---

## 3. Major concerns

### M1. The animal-level interval includes zero

The primary interval resamples landmarks within the five recordings. It does not resample recordings. A cluster bootstrap that does resample recordings gives:

```text
block 5:  [-0.041, +0.479]
block 20: [-0.131, +0.440]
```

The t interval over the five session means is `[-0.064, +0.587]`; the one-sided sign test is `p = 0.1875`.

The manuscript sometimes acknowledges this and sometimes writes as if the conditional interval were a robustness result. It is not. The correct statement is:

> The effect is positive within these five recordings, but the design does not establish that it would replicate in another five animals.

**Required revision:** report both intervals side by side, label the target of each, and remove every “across animals” or “robust” phrase that is not explicitly conditioned on the five recordings.

### M2. The 175 landmarks are not independent

Adjacent landmarks are 10 frames apart; forecasts span 20–100 frames. The approximate number of non-overlapping H100 windows is 25, not 175. The manuscript reports 175 as if it were a sample size in several places.

The block bootstrap is an attempt to respect dependence, but a 5-frame block is short relative to a 20-frame forecast overlap. Subsampling every 5th landmark gives a larger estimate (`+0.401`), which the authors correctly do not present as confirmation. But the main text still uses “175 landmarks” as a rhetorical precision anchor.

**Required revision:** report effective units, the non-overlap proxy, and the autocorrelation range. Do not describe 175 as independent replicates.

### M3. The energy advantage is confounded with calibration/spread

B3’s marginal 95% coverage is `0.938`; B5’s is `0.725`. B5 is severely undercovered. The energy score rewards calibration and sharpness, not just the mean. The PCA-10 MSE difference is `+0.0545` with an interval `[-0.157, +0.241]`; the standardized MSE effect is `dz = 0.04`.

This is the most important statistical objection after M1. The manuscript reports the energy result as “predictive information,” but a model with a better-calibrated spread can win on energy without a better conditional mean. B5’s undercoverage is a plausible alternative explanation.

The authors do not decompose the energy score into calibration and resolution, and they do not match ensemble spread. They acknowledge the coverage difference but do not let it constrain the claim.

**Required revision:** either (a) run a spread-matched or calibration-matched comparison, or (b) explicitly downgrade the claim to “a proper-score difference whose calibration and resolution components are not separated.” The current wording implies a mean-information result that the MSE null does not support.

### M4. B3 and B5 are not capacity-matched

B3 has `n(n+1)` parameters; B5-10 has `12n+110`. The ratio is `7.96–9.95×`. B3 also has arbitrary residual cross-neuron rows; B5 is a diagonal-plus-rank-10 model. The manuscript acknowledges this in the limitations, but the abstract and the 51/49 sentence still read as if the residual increment were information.

A valid restricted-model contrast is not a capacity-matched causal ablation. The 51/49 split is an ordered decomposition, not a unique structural partition. With B1m as baseline it is 23.5/76.5; with B0 it is 72.4/27.6.

**Required revision:** add a capacity-matched restricted model, or remove “information” from the residual claim. Report the ratio in the abstract, not only the limitations.

### M5. The test block was reused and the sign rule was clarified with the results

The final audit says the test tails were inspected in earlier phases. The only committed protocol was added with the results and contains a “sign-convention clarification.” There is no independent pre-evaluation commit.

This does not invalidate the number. It invalidates the confirmatory label. The final audit downgrades G4 to `PASS_UNDER_CORRECTED_SIGN` and `NOT_ESTABLISHED`, which is the right move. But the common-state results document still says “pre-registered” and “G4 PASS” in places, and a reader who stops there will overread the result.

**Required revision:** make the confirmatory status impossible to miss in the abstract, results, and claims table. Add a dated correction to the common-state results document or a prominent banner pointing to the final audit.

### M6. The rolling fit uses test-past, contradicting “test touched once”

At each test landmark, B3/B5 are refit on all transitions before `t`, including validation and earlier test frames. This is not future leakage. It is online adaptation to the test session.

The generic `METHODOLOGY.md` says “Test frames are touched once, for final metrics.” That is false for this analysis. The final audit’s “causal window” language is accurate, but the generic document is not.

**Required revision:** state the estimand precisely: “online-refit forecast using all observed frames before each landmark,” not “frozen train-only forecast.” Discuss whether B3’s larger capacity can exploit test-past more than B5.

### M7. The off-diagonal operator is unstable; the interaction claim is unsupported

The off-diagonal cross-segment correlation is `0.052`. The structural gate G7 fails. The synthetic common-input scenario produces spurious off-diagonal structure and a worse-than-diagonal forecast. The manuscript’s own limitations say this.

Yet the phrase “cross-neuron information” can be read as “interaction.” It is not. The correct phrase is “residual predictive structure beyond the ten-factor control,” which may be coupling, common input, behaviour, nonlinearity, nonstationarity or capacity.

**Required revision:** replace “cross-neuron information” with “residual predictive structure” throughout, unless a capacity-matched, identifiability-aware analysis is added.

### M8. There is no intervention; the biological claim is not evaluable

The virtual perturbation is a +1 SD displacement in a fitted model. Cross-fit response cosine is `0.556`; some signs reverse; no sham, no stimulation, no rescue. The standing label is correct.

The manuscript generally respects this, but the discussion still speculates about command neurons and intervention targets. Speculation is acceptable if clearly labelled; it is not acceptable in the claims table.

**Required revision:** keep the biological claim `NOT_EVALUABLE`; remove any sentence that implies a validated target.

### M9. Documentation errors undermine trust

Confirmed errors:

- `DATA_PROVENANCE.md` size `128,511,040` vs actual `56,390,975`;
- “35 landmarks per session” vs 35/35/34/38/33;
- common test boundary `2689:3137` vs per-recording bounds;
- “five repeated sessions, not animals” vs five animals inferred;
- field names `IDs`, `timeVectorSeconds` vs HDF5 `NeuronNames`, `tv`;
- `stateParams` row mismatch.

Individually, most are minor. Collectively, they show that the documentation was not independently checked against the archive. A reviewer is entitled to ask what else was not checked.

**Required revision:** either correct the old documents or add a prominent errata banner to each. The correction register is good, but a reader should not need to find it.

### M10. The literature framing overreaches

The manuscript is not a replication of Kato, BunDLe-Net or NC-MCM. It uses the same data but asks a different question. Sentences that imply a benchmark comparison are unsupported: no shared splits, no shared metrics, no retraining.

**Required revision:** add a “relationship to prior work” paragraph that explicitly says “different question, same data,” and remove “outperforms” language.

---

## 4. Minor concerns

1. **Effect size reporting.** `dz = 0.27` is small; the median `+0.186` and IQR crossing zero should be in the abstract, not only the supplement.
2. **Horizon multiplicity.** Four horizons are reported; H20 is primary; no family-wise correction. State that the intervals are not family-wise.
3. **Session 2.** The reversal is large (`-0.193`). A single-session reversal is not a nuisance; it is a heterogeneity result.
4. **Coverage definition.** “95% coverage” is marginal per-dimension, not joint. Say so every time.
5. **B6.** It is a coefficient ablation, not a refitted model. The current text says this once; say it wherever B6 appears.
6. **Factor variance.** The ten factors explain 54.7–71.0% of training variance; report the residual range, not only the high end.
7. **`stateParams`.** Remove it from any future analysis unless row alignment is fixed.
8. **H5 boundary.** Session 1’s H5 target is the last frame. Report the zero-margin condition.
9. **Figure captions.** Any figure showing 175 landmarks should state “overlapping windows.”
10. **Terminology.** “Animal” vs “recording” vs “session” is used inconsistently. Define once and use consistently.

---

## 5. Statistical checklist

| Item | Status | Comment |
|---|---|---|
| Unit of inference stated | Partial | Recordings vs landmarks is inconsistent across documents |
| Animal-level interval | Fail | Recording bootstrap includes zero |
| Effective sample size | Fail | 175 overlapping windows treated as replicates |
| Multiple comparisons | Partial | Exploratory; no family-wise claim, but not always stated |
| Effect size | Partial | `dz` not in abstract; MSE null underemphasized |
| Calibration | Fail | B5 0.725 marginal; no joint calibration |
| Capacity control | Fail | 7.96–9.95× ratio; no matched model |
| Test reuse | Disclosed but not prominent | Exploratory label correct; old docs still say “pre-registered” |
| Sign rule | Disclosed | Confirmatory status not established |
| Negative controls | Good | H1, session 2, MSE, neuron space, surrogates |
| Preregistration | Fail | Protocol added with results |
| Intervention | Not evaluable | No data |

---

## 6. Reproducibility checklist

| Item | Status | Comment |
|---|---|---|
| Archive hash | Pass | Matches OSF |
| Code commit | Pass | Branch/HEAD verified |
| Environment | Pass | Python 3.13, pinned artifacts |
| End-to-end rerun | Pass | Clean rerun, empty stderr |
| Numerical match | Pass | `4.44e-16` max absolute difference |
| Independent implementation | Pass | Three landmarks, `7.46e-7` |
| Split/landmark tests | Pass | Six focused tests |
| Documentation consistency | Fail | Size, landmark count, boundary, animal wording |
| Claims traceability | Pass | JSON ledger + artifacts |

A reproducibility checklist is not a scientific validity checklist. The project passes the former and partially fails the latter.

---

## 7. Questions to the authors

1. What is the unit of inference for the primary claim: landmark, recording, or animal?
2. Why is the primary interval not a recording-level bootstrap?
3. How many non-overlapping H100 windows are there, and what happens to the effect when you use them?
4. What is the MSE effect, and why is it not the primary metric?
5. How much of the energy gap is calibration versus resolution?
6. What is the parameter ratio, and what happens in a capacity-matched model?
7. Why does B3 lose at H1?
8. Why does session 2 reverse?
9. Is the model refit on test-past? If so, what estimand does that define?
10. Was the sign rule written before the results? Show the commit.
11. What exactly does “cross-neuron information” mean, given the off-diagonal instability?
12. What biological intervention would falsify the coupling interpretation?
13. Are the five animals an independent cohort? How were they selected?
14. Why does the documentation say “five repeated sessions” when the publication says five animals?
15. What is the strongest sentence the data support, without the word “information”?

---

## 8. What would change the recommendation

### From “major revision” to “accept as exploratory report”

- Report both intervals with clear targets.
- Add the MSE null and effect size to the abstract.
- Add the capacity ratio to the abstract.
- Replace “information” with “residual predictive structure.”
- Fix or banner the documentation errors.
- Add a calibration/spread limitation paragraph.
- Add a “different question from prior work” paragraph.

### From “major revision” to “accept as confirmatory paper”

This is not achievable with the current data. It would require:

- an independent cohort;
- an animal-level design;
- a capacity-matched control;
- a calibration-matched comparison;
- a preregistered sign/horizon/metric;
- no test reuse;
- an intervention for any biological claim.

No amount of reanalysis of these five recordings can substitute for those.

---

## 9. Decision

**As a confirmatory mechanistic paper:** reject. The design cannot identify the mechanism, the animal-level interval includes zero, and the capacity/calibration confounds are unresolved.

**As an exploratory methodological report:** major revision. The numerical work is solid, the negative results are valuable, and the claims are close to correct. The revisions are about precision, not about rescuing a false result.

**As an MSc thesis chapter:** acceptable for defense **if** the student can state the twelve sentences in the claims ledger without notes and can explain why the animal-level interval includes zero. The thesis should not claim a biological mechanism.

The project is ready for **methodological supervision**, not for **scientific sign-off**.
