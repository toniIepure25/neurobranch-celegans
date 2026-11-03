# Master Review: Supervisor and Oral-Defense Preparation

**Purpose:** help the student defend the work without overselling it.  
**Rule:** never answer a hard question by appealing to tests, commit hashes or “the pipeline passed.” Those establish reproducibility, not validity.

---

## 1. The one-sentence version

> In five Kato `WT_NoStim` recordings, a full population VAR has a modest H20 energy advantage over a ten-factor control, but the effect is exploratory, not animal-level, not capacity-controlled, and not biologically validated.

If you can say that sentence calmly, you can survive the defense.

---

## 2. The 30-second version

> I audited the Kato common-state result end to end. The data and arithmetic reproduce: the OSF hash matches, the splits are in bounds, and an independent implementation matches to below `1e-6`. The scientific result is narrower than the original framing. B3 beats B5 on H20 PCA-10 energy by about `0.26`, but the effect is small, reverses at H1, is null for MSE, and a bootstrap that resamples animals includes zero. B3 also has about eight to ten times more parameters, and B5 is badly undercovered. So I would present this as an exploratory predictive contrast, not a mechanism or a general forecast improvement.

---

## 3. The 2-minute version

> The project asks whether cross-neuron structure in the Kato whole-brain data adds predictive information beyond a strong common-factor control. I verified the archive: `WT_NoStim`, five recordings, 15,662 frames, 592 predictors after excluding eight behaviour-label neurons. The code uses per-recording temporal splits with a purge; every H1–H100 target is inside its recording and test mask. The final audit’s clean rerun reproduces the frozen numbers.
>
> The primary contrast is B3, a full lag-one VAR, versus B5, own history plus ten training-only PCA factors. On H20 PCA-10 energy, B3 is better by `+0.2644`; 64% of 175 landmarks favour B3; four of five sessions are positive. The conditional block-5 interval is `[+0.075, +0.403]`.
>
> But there are four qualifications. First, the 175 landmarks are overlapping windows, not independent samples. Second, a bootstrap that resamples the five recordings gives `[-0.041, +0.479]`, and the session t interval includes zero. Third, B3 has `7.96–9.95×` more parameters than B5, and no capacity-matched model exists. Fourth, the energy gain coexists with a large calibration difference—B3 95% marginal coverage is `0.938`, B5 is `0.725`—while the MSE difference is null.
>
> There are also two protocol facts. The test block was reused, and the sign rule was clarified with the results, so G4 is not confirmatory. And the rolling models are refit on earlier test frames, so this is an online-refit forecast, not a frozen train-only forecast.
>
> My conclusion is that the result is worth discussing as an exploratory methodological finding, but it is not ready for a confirmatory or biological claim. The next step is a capacity-matched, calibration-aware analysis, and ultimately an independent cohort.

---

## 4. The 5-minute structure

Use five slides, not twenty.

1. **Question and data** — Kato `WT_NoStim`; five animals; 15,662 frames; 592 predictors; per-recording splits.
2. **Models** — B3 full VAR; B5 own + ten factors; matched window, ridge, innovations.
3. **Result** — H20 energy `+0.2644`; 4/5 sessions; conditional interval; H1 reversal.
4. **Why it is exploratory** — animal-level interval; capacity; calibration; test reuse; test-past refit.
5. **What next** — capacity-matched model; calibration decomposition; independent cohort; intervention only if biological.

One number per slide. No table with twelve columns.

---

## 5. Numbers to know cold

| Quantity | Value |
|---|---:|
| File size | `56,390,975` bytes |
| SHA256 | `7531811c…a0098a` |
| Frames | 15,662 |
| Predictors | 592 |
| Named occurrences / unique names | 192 / 58 |
| Landmarks H100 | 35, 35, 34, 38, 33 = 175 |
| Non-overlapping H100 proxies | ~25 |
| Mean energy difference | `+0.2644` |
| Conditional block-5 interval | `[+0.075, +0.403]` |
| Recording bootstrap interval | `[-0.041, +0.479]` |
| Session t interval | `[-0.064, +0.587]` |
| Sign test one-sided | `0.1875` |
| Positive landmarks | `64.0%` |
| Paired `dz` | `0.270` |
| MSE difference | `+0.0545` |
| MSE interval | `[-0.157, +0.241]` |
| B3 95% coverage | `0.938` |
| B5 95% coverage | `0.725` |
| Parameter ratio | `7.96–9.95×` |
| H1 difference | `-0.0637` |
| H5 / H10 / H20 | `+0.0835 / +0.1471 / +0.2644` |
| Session 2 mean | `-0.1931` |
| B4m−B5 / B5−B3 | `+0.2751 / +0.2644` (51/49) |

If you remember only five: `+0.26`, `[+0.08, +0.40]`, `[-0.04, +0.48]`, `0.27`, `7.96–9.95×`.

---

## 6. Twenty likely questions

### Q1. “So does the population model capture cross-neuron interactions?”

> No. It captures residual predictive structure beyond ten common factors. That structure could be direct coupling, an unmeasured common input, behaviour, nonlinearity, nonstationarity or estimator capacity. The off-diagonal operator is unstable across segments (`r = 0.052`), so I do not claim an interaction graph.

### Q2. “The interval excludes zero. Why do you say it is not significant?”

> The primary interval is conditional on these five recordings; it resamples landmarks within sessions, not animals. When I resample recordings, the interval is `[-0.041, +0.479]`. The five session means give a t interval `[-0.064, +0.587]` and a sign test `p = 0.1875`. The conditional interval answers “within these recordings,” not “across animals.”

### Q3. “Isn’t 175 landmarks a large sample?”

> No. They are overlapping windows. Adjacent landmarks are 10 frames apart and forecasts span 20–100 frames. There are only about 25 non-overlapping H100 windows. The effective sample size is much closer to 25 than 175, and the biological units are five.

### Q4. “Why is the energy effect significant but MSE not?”

> Energy rewards calibration and sharpness, not just the mean. B5 is severely undercovered: `0.725` at 95%, versus `0.938` for B3. The MSE difference is `+0.05` with an interval crossing zero. So the energy result is partly a spread/calibration contrast. I do not claim a mean-forecast improvement.

### Q5. “Could the whole thing be a capacity effect?”

> Yes, that is unresolved. B3 has `n(n+1)` parameters; B5 has `12n+110`; the ratio is `7.96–9.95×`. B3 also has arbitrary residual rows. I did not run a capacity-matched restricted model, so I cannot separate information from capacity. That is the most important next experiment.

### Q6. “What is the 51/49 split?”

> It is an ordered decomposition: `B4m−B5` is `+0.2751` and `B5−B3` is `+0.2644`; together they sum to `B4m−B3 = +0.5395`. It is not a unique variance partition. With B1m as baseline the shares are 23.5/76.5; with B0 they are 72.4/27.6. I report it descriptively, not structurally.

### Q7. “Why does B3 lose at H1?”

> I do not know. It may be short-horizon persistence, variance propagation, or overfitting. The H1 difference is `-0.0637` with an interval excluding zero, so the reversal is real in this sample. It means the result is horizon-specific, not a general advantage.

### Q8. “Why does session 2 reverse?”

> I do not know. It is a large reversal (`-0.1931`), and it drives the animal-level uncertainty. It could be state composition, preparation quality, nonstationarity, or a genuine individual difference. With five animals, I cannot distinguish these.

### Q9. “Were the test frames untouched?”

> No. Earlier phases inspected the same test block, and the sign rule was clarified in the same commit as the results. The final audit correctly downgrades G4 to `PASS_UNDER_CORRECTED_SIGN` and `NOT_ESTABLISHED`. This is exploratory, not confirmatory.

### Q10. “Do the models see test data during fitting?”

> They see **past** test frames, not future frames. At each landmark, B3/B5 are refit on all transitions before `t`, including validation and earlier test frames. There is no future leakage, but the estimand is online-refit forecasting, not a frozen train-only model. The generic `METHODOLOGY.md` sentence “test frames are touched once” is false for this analysis.

### Q11. “Is the model calibrated?”

> No. B5’s marginal 95% coverage is `0.725`; B3’s is `0.938`. Those are marginal per-dimension coverages, not joint 10-D calibration. The standing label is `PROBABILISTIC CALIBRATION NOT ESTABLISHED`.

### Q12. “Can you identify direct connections from the VAR?”

> No. A VAR coefficient is an effective predictive coefficient at the calcium imaging timescale. The off-diagonal structure is unstable across chronological segments, and a synthetic common-input system produces spurious off-diagonal structure. The structural claim is `UNSUPPORTED`.

### Q13. “Can you predict what happens if I stimulate neuron X?”

> No. The virtual perturbation is a model-dependent +1 SD displacement. Cross-fit response cosine is `0.556`, with sign reversals. Without matched stimulation/sham data, biological intervention validity is `NOT_EVALUABLE`.

### Q14. “Are these five animals independent?”

> They are five distinct animals from one laboratory cohort, inferred from the publication and acquisition IDs. They are not an independent cohort. The archive has no explicit animal key, so the mapping is moderate-confidence.

### Q15. “Why does the documentation say five repeated sessions?”

> That was an error. The publication reports five animals, and the acquisition IDs are consistent with five preparations. I corrected the wording in the master review and logged it as X15.

### Q16. “What is the strongest claim you can make?”

> “In five Kato `WT_NoStim` recordings, B3 has lower H20 PCA-10 energy than B5-10 by `+0.2644`; the effect is positive in four of five sessions but not established at the animal level, is null for mean MSE, reverses at H1, and is confounded with capacity and calibration.”

### Q17. “What would falsify your interpretation?”

> A capacity-matched model that removes the advantage; a calibration-matched comparison that removes it; an independent cohort with an animal-level interval including zero; or an intervention that fails to reproduce the predicted response. Any one would substantially weaken the claim.

### Q18. “Why not just use a nonlinear model?”

> A nonlinear model could improve prediction without identifying mechanism. The synthetic common-input result shows that more flexibility can produce spurious structure and worse forecasting. The question is not “can we fit better?” but “what does the contrast identify?”

### Q19. “What did you learn from the audit?”

> That numerical reproducibility is not scientific validity. The pipeline can pass 457 tests and reproduce to `1e-16` while the inferential claim is still unsupported. The hard part is the unit of inference, the capacity control, and the calibration—not the code.

### Q20. “What should be done next?”

> First, a capacity-matched restricted model. Second, a calibration/spread decomposition. Third, a prospective reanalysis with a preregistered sign and metric. Fourth, an independent cohort at the animal level. Intervention only if the question is biological.

---

## 7. Questions you should not bluff

If asked about:

- the exact Kato dissection/immobilisation protocol;
- the exact strain and transgene;
- the annotation algorithm for the eight states;
- the publisher’s methods text;
- a specific optogenetics result;
- BunDLe-Net’s exact loss weights;

say:

> I did not verify that from the primary methods text. My audit covers the archive and the computational pipeline; I would need to check the paper’s methods before answering.

This is stronger than inventing a detail. The audit’s credibility depends on the distinction between verified and inferred.

---

## 8. The trap questions

### Trap 1: “So it works?”

Do not say yes. Say:

> It reproduces and the contrast is positive in this sample. Whether it “works” depends on the claim: as an exploratory predictive contrast, yes; as an animal-level or mechanistic claim, no.

### Trap 2: “But the p-value is significant?”

> The conditional interval excludes zero; the animal-level interval does not. The sign test is `p = 0.1875`. Significance depends on the unit, and the unit is the animal.

### Trap 3: “You ran 457 tests, so it must be right.”

> Tests establish that the code does what we wrote. They do not establish that the design answers the question. The audit found documentation and inference errors that all tests passed.

### Trap 4: “The reviewer said it’s exploratory—why defend it?”

> Because an exploratory result can still be useful and honest. The defense is about what the data support, not about claiming a discovery.

### Trap 5: “Just add more simulations and it will become causal.”

> No. A virtual perturbation is not an intervention. More simulations of the same model cannot create biological validity.

---

## 9. Slide-by-slide outline

### Slide 1 — Question

- One sentence: does residual cross-neuron structure add predictive information beyond a common-factor control?
- One caveat: “predictive, not mechanistic.”
- One dataset image or schematic.

### Slide 2 — Data and split

- Five recordings, 15,662 frames, 592 predictors.
- Per-recording split table, not one common boundary.
- Purge 12; test tails differ.

### Slide 3 — Models

- B3: `z_{t+1} = A z_t + b + eps`.
- B5: own lag + ten factors.
- Matched window, ridge, innovations.
- Parameter ratio in a corner, not hidden.

### Slide 4 — Result

- H20 energy difference `+0.2644`.
- 4/5 sessions; 64% landmarks.
- Conditional interval `[+0.075, +0.403]`.
- H1 `-0.0637`; session 2 `-0.1931`.

### Slide 5 — Why exploratory

- Animal-level interval `[-0.041, +0.479]`.
- MSE null.
- Coverage `0.938` vs `0.725`.
- Capacity `7.96–9.95×`.
- Test reuse / test-past refit.

### Slide 6 — Next

- Capacity-matched model.
- Calibration decomposition.
- Independent cohort.
- Intervention only if biological.

---

## 10. Opening and closing

### Opening

> This is an audit of a predictive claim, not a discovery talk. I will show what reproduces, what the contrast supports, and why I do not think it supports a biological mechanism.

### Closing

> The pipeline is reproducible and the contrast is real in this sample. The scientific claim is narrower: an exploratory, horizon-specific energy difference that is not animal-level, not capacity-controlled, and not calibrated. I would rather defend that narrow claim than a broad one I cannot support.

---

## 11. Rehearsal checklist

- [ ] Say the one-sentence version in under 30 seconds.
- [ ] Say the 2-minute version without notes.
- [ ] Draw the B3 and B5 equations from memory.
- [ ] Explain why 175 ≠ 175.
- [ ] Explain why the conditional and animal-level intervals differ.
- [ ] Explain why energy can improve while MSE does not.
- [ ] Explain why a VAR coefficient is not a synapse.
- [ ] Explain why a virtual perturbation is not an intervention.
- [ ] State the capacity ratio without looking.
- [ ] State the H1 reversal without looking.
- [ ] State the session-2 reversal without looking.
- [ ] Admit one thing you did not verify.
- [ ] Ask for the next experiment, not for approval.

---

## 12. What “ready for Denis” should mean

It should mean:

> The student can explain the result, the uncertainty, the confounds and the next experiment, and will not overclaim under pressure.

It should not mean:

> The numbers are final and the science is settled.

The first is achievable now. The second is not.
