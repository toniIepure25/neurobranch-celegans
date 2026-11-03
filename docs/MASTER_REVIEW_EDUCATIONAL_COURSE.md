# Master Review: A Graduate Course in Computational Neuroscience Evidence

**16 chapters.** Each chapter has learning goals, a core idea, a worked example from this project, a code pointer, a common misconception, exercises, and five oral questions.  
**Course rule:** no chapter may be passed by citing a test result. The test is whether you can explain the inferential object.

---

## Chapter 1 — What is a scientific claim?

**Learning goals:** distinguish a number, a result, a claim, and a mechanism; identify the unit of inference.

A number is `+0.2644`. A result is “B3 has lower mean H20 energy than B5 in this sample.” A claim is “population models retain predictive information beyond common factors.” A mechanism is “residual cross-neuron coupling causes the improvement.”

Each step adds an assumption. The number needs arithmetic. The result needs a comparison and a sample. The claim needs a unit, a control, and a generalization argument. The mechanism needs an intervention or a strong identification design.

Most computational neuroscience overclaiming happens at the third step: a result is promoted to a claim without naming the unit. In this project the result is within five recordings; the claim is often written as if it were across animals. That is a unit error, not a math error.

**Worked example:** `+0.2644` is the mean paired difference over 175 landmarks. The 175 are overlapping windows from five animals. The number is real; the unit is not 175.

**Code pointer:** [`scripts/kato_common_state_main.py`](../scripts/kato_common_state_main.py) produces the per-landmark table; [`scripts/master_review/build_audit_ledgers.py`](../scripts/master_review/build_audit_ledgers.py) turns it into claims.

**Misconception:** “If the number is reproducible, the claim is established.” Reproducibility is necessary, not sufficient.

**Exercises:**
1. Write the number, result, claim and mechanism for the B3/B5 contrast.
2. Name the unit of each.
3. Find one sentence in the project that jumps from result to mechanism.
4. Rewrite it so the jump is explicit.
5. State what evidence would justify the jump.

**Oral questions:**
1. What is the difference between a result and a claim?
2. What is the unit of inference here?
3. Why is a reproducible number not a validated claim?
4. Give an example of a unit error in the project.
5. What would a mechanistic claim require?

---

## Chapter 2 — Dataset forensics and provenance

**Learning goals:** verify identity, size, hash, field names, and the difference between a file and a cohort.

A dataset is not a filename. It is a specific object with a hash, a size, a schema, a version, and a provenance chain. The audit found:

- actual size `56,390,975` bytes;
- documented size `128,511,040` bytes;
- SHA256 `7531811c…a0098a`;
- OSF hash match;
- HDF5 fields `NeuronNames`, `tv`, `States`, `deltaFOverF_bc`;
- documented fields `IDs`, `timeVectorSeconds` — wrong.

The file identity is verified. The documentation is not. Those are separate findings.

**Worked example:** a size mismatch can mean truncation, a different version, or a stale doc. The hash resolves it: the local file is the OSF object, and the doc is stale. Without the hash, the mismatch would be ambiguous.

**Code pointer:** [`src/neurobranch/data/kato.py`](../src/neurobranch/data/kato.py); [`reports/master_review/dataset_identity_audit.json`](../reports/master_review/dataset_identity_audit.json).

**Misconception:** “The file opens, so it is the right file.” Opening proves parseability, not identity.

**Exercises:**
1. Compute the SHA256 and compare to OSF.
2. List every top-level HDF5 field and its shape/orientation.
3. Find a documented field that does not exist.
4. Explain why a size mismatch is ambiguous without a hash.
5. Write a provenance paragraph that distinguishes verified from inferred.

**Oral questions:**
1. How do you know this is the Kato dataset?
2. What is the difference between size and hash verification?
3. Why does the HDF5 orientation matter?
4. Which provenance claims are inferred?
5. What would make the provenance chain incomplete?

---

## Chapter 3 — Calcium imaging in *C. elegans*

**Learning goals:** relate the measurement to the model; distinguish calcium, spikes, and behaviour.

Kato et al. imaged whole-brain calcium in immobilised worms at ~2.9 Hz. The signal is ΔF/F, not voltage and not spikes. Calcium indicators integrate and decay; the observed time series is a filtered, nonlinear function of neural activity.

A lag-one VAR at ~0.34 s per frame describes an effective transition at the imaging timescale. It is not a synaptic weight, a membrane time constant, or a gap-junction conductance.

The eight states (`FWD`, `SLOW`, `DT`, `VT`, `REV1`, `REV2`, `REVSUS`, `NOSTATE`) are behavioural/motor categories derived from command/interneuron activity and behaviour. They are not independent ground truth.

**Worked example:** if a neuron’s calcium decays over several frames, a VAR coefficient can be large because of indicator kinetics, not because the neuron drives its partners. The model cannot distinguish these without deconvolution or faster imaging.

**Code pointer:** loader and state handling in [`src/neurobranch/data/kato.py`](../src/neurobranch/data/kato.py); behaviour-label exclusion in [`src/neurobranch/preprocessing/__init__.py`](../src/neurobranch/preprocessing/__init__.py).

**Misconception:** “Calcium activity is neural activity.” It is an indirect, slow, nonlinear measurement.

**Exercises:**
1. Convert the frame rate to seconds per frame for each recording.
2. Explain why a VAR lag is not a synaptic delay.
3. List three biological processes that can produce an off-diagonal coefficient.
4. Explain why the state labels are circular.
5. Propose an experiment that would separate indicator kinetics from coupling.

**Oral questions:**
1. What does the calcium signal measure?
2. Why can’t you call a VAR coefficient a synapse?
3. What are the eight states, and how were they derived?
4. Why is behaviour a confound even without a stimulus?
5. What imaging modality would test the timescale objection?

---

## Chapter 4 — Temporal splits and leakage

**Learning goals:** define leakage; distinguish future leakage, test-past adaptation, and test reuse.

Leakage is information in training that would not be available at prediction time. There are three different problems in this project:

1. **Future leakage:** using frames after `t` to predict `t`. The split guard prevents this.
2. **Test-past adaptation:** refitting at `t` on frames before `t`, including earlier test frames. This is causal but changes the estimand.
3. **Test reuse:** inspecting the same test block in earlier phases and again in the final audit. This is a protocol problem, not a code leak.

The first is absent. The second is present. The third is present.

**Worked example:** at landmark `2690`, a 500-frame window starts at `2190`. It includes the end of training (`2195`), validation (`2207–2676`), and one test frame (`2689`). The model has seen test-past. It has not seen the target at `2690+H`.

**Code pointer:** [`src/neurobranch/preprocessing/__init__.py:219-241`](../src/neurobranch/preprocessing/__init__.py#L219-L241); [`src/neurobranch/kato_common_state/models.py:338-396`](../src/neurobranch/kato_common_state/models.py#L338-L396).

**Misconception:** “No future leakage means the test set is untouched.” False. Test-past is still test data.

**Exercises:**
1. For session 0, list the train/val/test ranges and the first landmark.
2. Show which splits the first fit window touches.
3. Classify each touch as future leakage, test-past adaptation, or test reuse.
4. Write the estimand for a frozen train-only model and for the rolling model.
5. Explain why both can be legitimate but not interchangeable.

**Oral questions:**
1. What is leakage?
2. Does the code use future frames?
3. Does it use test-past?
4. Is test-past adaptation cheating?
5. What estimand does the rolling fit define?

---

## Chapter 5 — Landmarks, windows and effective sample size

**Learning goals:** compute valid starts; understand overlap; distinguish nominal from effective `n`.

A landmark is a forecast origin. A window is `[t-h+1, t+H]`. A start is valid only if the whole window is inside the split. With `h=2`, the first test start is one frame after the test boundary; with stride 10, every tenth valid start is a landmark.

The nominal count is 175. The effective count is much smaller because windows overlap. At H100, there are about 25 non-overlapping windows. At H20, about 110. The ratio `175/25 = 7` is the inflation factor for H100.

**Worked example:** session 0 test is `2689:3137`, length 448. Valid H100 starts are `2690..3036`, 347 starts. Every 10th gives 35 landmarks. The last landmark is `3030`; its H100 target is `3130`, inside the recording.

**Code pointer:** [`src/neurobranch/adaptive/evaluation.py:43-47`](../src/neurobranch/adaptive/evaluation.py#L43-L47); [`reports/master_review/frame_landmark_bounds.json`](../reports/master_review/frame_landmark_bounds.json).

**Misconception:** “175 landmarks = 175 independent samples.” They are 175 evaluation origins, not 175 animals or 175 independent windows.

**Exercises:**
1. Compute valid H20 and H100 starts for each session.
2. Count landmarks at stride 10.
3. Compute non-overlapping proxies at H20 and H100.
4. Explain why the first landmark is not the test boundary.
5. Find a document that says “35 per session” and correct it.

**Oral questions:**
1. What is a landmark?
2. Why does history shift the first test start?
3. How many non-overlapping H100 windows are there?
4. Why is 175 not the sample size?
5. What happens to the interval if you use non-overlapping windows?

---

## Chapter 6 — The VAR model

**Learning goals:** write the model, fit it, interpret the operator, and avoid causal overreads.

A lag-one VAR is:

```text
z_{t+1} = A z_t + b + eps_t
```

`A` is an `n x n` effective transition operator; `b` is an intercept; `eps_t` is innovation. The model is linear, Markov in `z`, and time-homogeneous within its fit window.

The ordinary least squares solution is:

```text
A^T = (Z^T Z)^{-1} Z^T Y
b   = mean(Y) - mean(Z) A^T
```

With ridge:

```text
A^T = (Z^T Z + lambda I)^{-1} Z^T Y
```

The intercept is not penalized. The code centers features and targets, solves in standardized coordinates, then back-transforms.

**Worked example:** for `n=103`, `A` has `10,609` coefficients and `b` has `103`, total `10,712`. The model has more parameters than some recordings have independent frames; ridge and the local window matter.

**Code pointer:** [`src/neurobranch/adaptive/var.py:42-100`](../src/neurobranch/adaptive/var.py#L42-L100).

**Misconception:** “A significant VAR coefficient is a direct connection.” It is a conditional predictive coefficient, confounded by common input, behaviour, nonlinearity, and sampling.

**Exercises:**
1. Write the ridge objective with an unpenalized intercept.
2. Derive the parameter count for `n=127`.
3. Explain what `A[i,j]` means and does not mean.
4. Fit a 2-neuron VAR by hand on a toy series.
5. Show why centering makes the intercept invariant to offsets.

**Oral questions:**
1. Write the VAR equation.
2. What is `A`?
3. Why ridge?
4. What does `A[i,j]` not mean?
5. How many parameters for 127 neurons?

---

## Chapter 7 — Ridge, stability and spectral radius

**Learning goals:** understand regularization, eigenvalue stability, and the effect of a spectral cap.

Ridge shrinks coefficients toward zero. It reduces variance at the cost of bias. The penalty `lambda=1.0` is applied to `A`, not `b`.

The spectral radius is:

```text
rho(A) = max_i |lambda_i(A)|
```

If `rho(A) > 0.995`, the code scales `A` by `0.995/rho(A)`. This keeps the autonomous forecast bounded. It also changes the fitted operator; the cap is part of the model, not a cosmetic post-processing step.

**Worked example:** in the independent check, session 0’s B3 raw radius is capped at `0.9950000000000014`; session 4’s is `0.98915`, below the cap. The cap binds in some sessions, not all.

**Code pointer:** [`src/neurobranch/adaptive/var.py:25-39`](../src/neurobranch/adaptive/var.py#L25-L39).

**Misconception:** “The spectral radius is a biological stability measure.” It is a property of the fitted effective operator at the imaging timescale.

**Exercises:**
1. Compute the eigenvalues of a 2x2 matrix.
2. Show that scaling `A` by `c` scales all eigenvalues by `c`.
3. Explain why a cap can improve long-horizon forecasts.
4. Explain why a cap can bias short-horizon forecasts.
5. Find a session where the cap binds.

**Oral questions:**
1. What does ridge do?
2. What is the spectral radius?
3. Why cap at 0.995?
4. Does the cap make the model “stable” biologically?
5. What is lost when the cap binds?

---

## Chapter 8 — PCA and common-factor models

**Learning goals:** fit PCA on training only; interpret scores and residuals; understand the B5 restriction.

PCA finds orthogonal directions of maximum training variance:

```text
f_t = P^T (z_t - mean_train)
z_t ≈ mean_train + P f_t + r_t
```

With ten components, B5 keeps `f_t in R^10` and models:

```text
f_{t+1} = A_f f_t + b_f + eta_t
z_{t+1,j} = b_j + a_j z_{t,j} + beta_j^T f_t + eps_{t,j}
```

This is a diagonal-plus-rank-10 model. It has `12n+110` parameters. The ten factors explain `54.7–71.0%` of training variance.

**Worked example:** for `n=103`, B5 has `1,346` parameters; B3 has `10,712`; ratio `7.96`. The factor model is not a small correction; it is a different hypothesis class.

**Code pointer:** [`src/neurobranch/kato_common_state/models.py:224-292`](../src/neurobranch/kato_common_state/models.py#L224-L292).

**Misconception:** “Ten factors capture the whole population because ten components are enough for the metric.” The metric uses ten components, but the residual dynamics are restricted to one direction per neuron.

**Exercises:**
1. Compute the parameter count for `n=121`, `k=10`.
2. Explain why PCA must be training-only.
3. Derive the residual after ten components.
4. Show why B5 cannot represent an arbitrary residual row.
5. Compare B5 to a factor-analysis model.

**Oral questions:**
1. What is PCA?
2. Why training-only?
3. How many parameters does B5 have?
4. What does B5 restrict?
5. Does the ten-factor metric hide the restriction?

---

## Chapter 9 — Proper scoring: energy, MSE, coverage

**Learning goals:** define a proper score; distinguish calibration, sharpness, and mean accuracy.

The energy score is:

```text
ES(P, y) = E_{X~P} ||X - y|| - 0.5 E_{X,X'~P} ||X - X'||
```

It is proper: the expected score is minimized by the true distribution. Lower is better. It rewards calibration and sharpness.

MSE of the ensemble mean is:

```text
MSE = mean_j (mean_m X_{m,j} - y_j)^2
```

It measures conditional-mean accuracy only. Coverage is:

```text
coverage_95 = mean_j 1[y_j in central 95% marginal interval]
```

It measures marginal per-dimension calibration, not joint calibration.

**Worked example:** B3 energy `2.700`, B5 `2.965`; difference `+0.2644`. B3 MSE `1.7067`, B5 `1.7612`; difference `+0.0545`. B3 coverage `0.938`, B5 `0.725`. The energy gap is larger than the MSE gap and coexists with a coverage gap.

**Code pointer:** [`src/neurobranch/probabilistic/metrics.py`](../src/neurobranch/probabilistic/metrics.py); [`src/neurobranch/kato_common_state/evaluation.py`](../src/neurobranch/kato_common_state/evaluation.py).

**Misconception:** “Lower energy means better mean prediction.” It means a better distributional score, which can be driven by spread.

**Exercises:**
1. Compute the energy score for a 1-D two-member ensemble by hand.
2. Show why the fair estimator uses distinct pairs.
3. Compare energy and MSE on a biased, overdispersed ensemble.
4. Explain marginal vs joint coverage.
5. Construct an example where MSE improves but energy worsens.

**Oral questions:**
1. What is a proper score?
2. Why use energy instead of MSE?
3. What does coverage measure?
4. Why is B5 undercovered?
5. Can energy improve without mean improvement?

---

## Chapter 10 — Ensembles and uncertainty

**Learning goals:** understand how predictive ensembles are generated; distinguish aleatoric, epistemic, and model-dependent uncertainty.

The project uses a shared innovation stream to compare models. Each model propagates the same shocks through its operator. This is a conditional comparison: differences are due to the model, not different random draws.

The ensemble spread is not automatically calibrated. A model can produce a narrow ensemble because its innovation variance is small, because its operator damps shocks, or because it ignores a source of uncertainty. Coverage diagnoses the result; it does not fix it.

**Worked example:** B5’s 95% marginal coverage is `0.725`. Its ensembles are too narrow at the 95% level, even though its covariance error is lower than B3’s. Narrowness and covariance shape are different.

**Code pointer:** rollout code in [`src/neurobranch/kato_common_state/models.py`](../src/neurobranch/kato_common_state/models.py); scoring in [`src/neurobranch/kato_common_state/evaluation.py`](../src/neurobranch/kato_common_state/evaluation.py).

**Misconception:** “More ensemble members means better uncertainty.” More members reduce Monte Carlo error, not model error.

**Exercises:**
1. Explain the shared-innovation design.
2. Distinguish aleatoric and epistemic uncertainty in this model.
3. Construct an overconfident ensemble with the correct mean.
4. Construct a calibrated but uninformative ensemble.
5. Propose a calibration diagnostic beyond marginal coverage.

**Oral questions:**
1. How are ensembles generated?
2. Why share innovations?
3. What is the ensemble measuring?
4. Why is B5 undercovered?
5. Does more members fix calibration?

---

## Chapter 11 — Dependence and effective sample size

**Learning goals:** compute autocorrelation; understand block bootstrap; estimate effective `n`.

Observations in a time series are dependent. Forecast windows overlap. The nominal sample size overstates information.

The autocorrelation of the paired difference is `0.179–0.597` at lag 1. Adjacent forecasts share most of their trajectory. A block bootstrap resamples contiguous blocks to preserve local dependence; the block length trades bias and variance.

**Worked example:** 175 landmarks, 25 non-overlapping H100 windows. Treating 175 as independent inflates precision by about `sqrt(7) ≈ 2.6`. The conditional interval `[+0.075, +0.403]` is narrower than the animal-level interval `[-0.041, +0.479]` for this reason and because it does not resample animals.

**Code pointer:** [`src/neurobranch/kato_interactions/metrics.py:36-76`](../src/neurobranch/kato_interactions/metrics.py#L36-L76).

**Misconception:** “The bootstrap fixes dependence.” It only fixes the dependence structure you specify. A within-session block bootstrap cannot fix the absence of animal resampling.

**Exercises:**
1. Compute lag-1 autocorrelation for a toy series.
2. Explain block bootstrap bias vs variance.
3. Estimate effective `n` for AR(1).
4. Compare block lengths 5, 20, 100.
5. Explain why a longer block is not always better.

**Oral questions:**
1. Why are landmarks dependent?
2. What does a block bootstrap preserve?
3. What does it not preserve?
4. What is the effective sample size?
5. Why does block 20 give a narrower interval than block 5?

---

## Chapter 12 — Confidence intervals and hypothesis tests

**Learning goals:** distinguish interval targets; understand sign tests; avoid p-value overreads.

A confidence interval is a procedure, not a posterior. Its target must be stated: within-session mean, animal-level mean, or conditional mean.

The project reports:

- conditional block-5: `[+0.075, +0.403]`;
- recording bootstrap: `[-0.041, +0.479]`;
- session t: `[-0.064, +0.587]`;
- sign test: `p = 0.1875` one-sided.

These are not contradictory. They answer different questions with different units.

**Worked example:** with five sessions, four positive, the one-sided sign test is `0.1875`. Only 5/5 gives `0.03125`. The design cannot produce a significant sign test with one reversal.

**Code pointer:** [`scripts/master_review/statistical_audit.py`](../scripts/master_review/statistical_audit.py); [`reports/master_review/statistical_audit.json`](../reports/master_review/statistical_audit.json).

**Misconception:** “The interval excludes zero, so the effect is real across animals.” The interval’s target is not animals.

**Exercises:**
1. Compute a sign test for 4/5.
2. Compute a t interval for five session means.
3. Explain why the recording bootstrap is wider.
4. Distinguish confidence and credible intervals.
5. Write a sentence that reports both targets without contradiction.

**Oral questions:**
1. What is the target of the primary interval?
2. Why does the animal-level interval include zero?
3. What is the sign test?
4. Why not just report the smallest interval?
5. What would a preregistered test look like?

---

## Chapter 13 — Capacity, ablation and identifiability

**Learning goals:** distinguish a restriction from a capacity match; understand ordered decompositions; recognize observational non-identifiability.

B3 has `n(n+1)` parameters; B5 has `12n+110`. The ratio is `7.96–9.95×`. A model can win because it has more capacity, not because it has the right structure.

An ablation removes a component. A capacity-matched ablation removes a component **and** matches the parameter budget. B6 zeros B3’s off-diagonals but does not refit; it is a diagnostic, not a clean restriction. B4m and B5 are clean restrictions but not capacity-matched.

The 51/49 split is ordered:

```text
(B4m - B5) + (B5 - B3) = B4m - B3
+0.2751 + +0.2644 = +0.5395
```

It is arithmetic, not a unique causal partition. Different baselines give 23.5/76.5 or 72.4/27.6.

**Worked example:** the synthetic common-input system has no direct coupling but produces spurious off-diagonal structure; a full model can forecast worse than a diagonal model (`MSE ratio 1.556`). Identifiability is not guaranteed by a good fit.

**Code pointer:** [`scripts/identifiability_audit.py`](../scripts/identifiability_audit.py); [`reports/master_review/capacity_audit.json`](../reports/master_review/capacity_audit.json).

**Misconception:** “Ablation proves necessity.” It proves that removing a component from this model changes the score; it does not prove the component is the biological mechanism.

**Exercises:**
1. Compute the parameter ratio for `n=118`.
2. Design a capacity-matched B3.
3. Show why B6 is not a refitted model.
4. Compute the 51/49 shares.
5. Explain why the shares change with baseline.

**Oral questions:**
1. What is capacity?
2. Why is B3/B5 not capacity-matched?
3. What is an ordered decomposition?
4. Why is B6 not a clean ablation?
5. What would identify residual information?

---

## Chapter 14 — Causality and intervention

**Learning goals:** distinguish prediction, intervention, and mechanism; understand why virtual perturbation is not enough.

A predictive model learns `P(Y|X)`. An intervention asks `P(Y|do(X=x))`. These are different objects. A VAR coefficient is a conditional association; it does not identify `do`.

A virtual perturbation displaces the state in a fitted model:

```text
z_0 -> z_0 + delta
```

It answers: “what does this model predict?” It does not answer: “what happens in the animal?”

The project’s cross-fit response cosine is `0.556`; some signs reverse. The virtual response is not stable across fits. There is no sham, no stimulation, no rescue.

**Worked example:** a +1 SD displacement in PC space is not a biological dose. It has no units, no targeting, and no control for network compensation.

**Code pointer:** virtual perturbation code in [`src/neurobranch/kato_interactions`](../src/neurobranch/kato_interactions); standing warning in [`src/neurobranch/adaptive/var.py:15`](../src/neurobranch/adaptive/var.py#L15).

**Misconception:** “If the model predicts it, the intervention will work.” Prediction is not intervention.

**Exercises:**
1. Write `P(Y|X)` and `P(Y|do(X))`.
2. Give a confounded example where they differ.
3. Design a sham-controlled stimulation experiment.
4. Explain why cross-fit stability is necessary but not sufficient.
5. State what would falsify a coupling claim.

**Oral questions:**
1. What is an intervention?
2. Why is a VAR not causal?
3. What is a virtual perturbation?
4. Why is it not a biological dose?
5. What data would make a causal claim evaluable?

---

## Chapter 15 — Reproducibility and software audit

**Learning goals:** verify environment, hash, rerun, and independent implementation; distinguish reproducibility from validity.

Reproducibility means the same inputs and code produce the same outputs. Validity means the outputs support the claim. A project can pass the first and fail the second.

The audit verified:

- branch/HEAD;
- clean rerun, empty stderr;
- max absolute difference `4.44e-16`;
- independent implementation, `7.46e-7`;
- 457 tests, validator pass;
- frozen artifacts preserved.

It also found documentation errors, test reuse, capacity/calibration confounds, and an animal-level interval including zero. None of these is a reproducibility failure.

**Worked example:** the final audit says `FULL_END_TO_END_REPRODUCTION_PASSED`. That is true for numerical reproduction. It is not a scientific validation. The phrase “passed” is doing double duty.

**Code pointer:** [`docs/KATO_FINAL_AUDIT_RESULTS.md`](KATO_FINAL_AUDIT_RESULTS.md); [`reports/master_review/quality_gate.json`](../reports/master_review/quality_gate.json).

**Misconception:** “457 tests passing means the science is right.” Tests check code behavior, not inference.

**Exercises:**
1. Run the focused tests.
2. Compare a clean rerun artifact to frozen output.
3. Write a reproducibility checklist.
4. Write a validity checklist.
5. Find one claim that passes reproducibility but fails validity.

**Oral questions:**
1. What is reproducibility?
2. What is validity?
3. What did the clean rerun prove?
4. What did it not prove?
5. Why preserve the old audit package?

---

## Chapter 16 — Writing, peer review and the next experiment

**Learning goals:** write a claim at the right strength; anticipate reviewer objections; design the next study.

A strong paper does not maximize the claim. It maximizes the ratio of claim to evidence. The strongest defensible sentence here is:

> In five Kato `WT_NoStim` recordings, B3 has lower H20 PCA-10 energy than B5-10 by `+0.2644`; the effect is positive in four of five sessions but not established at the animal level, is null for mean MSE, reverses at H1, and is confounded with capacity and calibration.

A reviewer will ask:

1. What is the unit?
2. Why 175?
3. Why energy and not MSE?
4. Is B5 calibrated?
5. Is B3 capacity-matched?
6. Was the test reused?
7. Is the operator stable?
8. Is there an intervention?

If you cannot answer 1–8, the paper is not ready.

**Worked example:** the peer-review decision is “major revision as exploratory report, reject as confirmatory mechanistic paper.” That is not a failure; it is the correct genre.

**Code pointer:** [`docs/MASTER_REVIEW_PEER_REVIEW.md`](MASTER_REVIEW_PEER_REVIEW.md); [`docs/MASTER_REVIEW_CLAIMS_LEDGER.md`](MASTER_REVIEW_CLAIMS_LEDGER.md).

**Misconception:** “A negative or exploratory result is not publishable.” A well-controlled negative or exploratory result is often more useful than an overclaimed positive.

**Exercises:**
1. Write an abstract in 150 words with the four qualifications.
2. Write a reviewer report with three major concerns.
3. Design a capacity-matched follow-up.
4. Design a calibration-matched follow-up.
5. Design an independent-cohort follow-up.

**Oral questions:**
1. What is the strongest allowed sentence?
2. What is the weakest forbidden sentence?
3. What would a reviewer ask first?
4. What is the next experiment?
5. What would make you abandon the claim?

---

## Capstone assessment

The student passes if they can, without notes:

1. State the number, result, claim and mechanism separately.
2. Name the unit of inference and why 175 is not it.
3. Explain no-future-leakage vs test-past adaptation vs test reuse.
4. Write the B3 and B5 equations and parameter counts.
5. Explain why energy improves while MSE does not.
6. Explain why B5 is undercovered.
7. Explain why the 51/49 split is not unique.
8. Explain why the off-diagonal operator is not a connectome.
9. Explain why virtual perturbation is not intervention.
10. State the strongest allowed claim and one forbidden claim.
11. Propose the capacity-matched, calibration-matched and independent-cohort experiments.
12. Say “I do not know” when they do not.

The course is not a celebration of the project. It is a training in the difference between a reproducible pipeline and a defensible claim. That difference is the whole job.
