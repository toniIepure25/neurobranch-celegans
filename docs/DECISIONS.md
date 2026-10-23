# Design Decisions and Rejected Alternatives

This is an audit trail of choices that materially affect results. Each entry
records the decision, the alternatives considered, the evidence, and the cost.

---

## D1. Treat the Kato labels as behavioural annotations, not ground truth

**Decision.** Exclude the eight label-defining neurons from predictors and state
everywhere that labels are not independent.

**Alternative.** Use all 302 segmented traces, including AVA/SMD/RIB, and report
high decoding accuracy.

**Why rejected.** The labels are derived from calcium in those very cells. Using
them as inputs would be direct label leakage. Even after exclusion, circularity
remains because labels and predictors come from the same recording; this residual
circularity cannot be removed without an independent behavioural assay.

**Cost.** Lower decoding accuracy and fewer candidate perturbation targets.

---

## D2. Report blocked interpolation as primary, strict forward as negative control

**Decision.** Primary benchmark = contiguous middle block held out, train on
earlier and later blocks. Strict forward split = separate negative control.

**Alternative A.** Use only a forward split. This produced catastrophic failures
(e.g. worm 1 MLP R² = −3.67, behaviour accuracy 0.123) that swamp model
comparisons and mostly measure non-stationarity.

**Alternative B.** Use only a random split. Random splits leak temporally adjacent
frames into training and inflate every metric; rejected outright.

**Why blocked.** It gives each model class a fair interpolation test while still
preventing window leakage. It is **not** a forecast. The forward split is retained
precisely so the interpolation result cannot be misread as prospective skill.

**Cost.** The primary number is not a deployable forecasting claim, and we must
keep repeating that caveat.

---

## D3. No smoothing or detrending in the primary suite

**Decision.** `smoothing_sigma_frames = 0`, `detrend_window_frames = 0`.

**Alternatives.** Gaussian smoothing; causal high-pass; zero-phase high-pass
matching upstream.

**Evidence.** Causal high-pass filtering removed slow common-mode drift but also
removed behaviourally relevant slow components and did not improve blocked
held-out prediction. Zero-phase filtering uses future samples and is therefore
inadmissible for a forecast-style evaluation. Aggressive smoothing reduced
variance but blurred transitions.

**Cost.** Common-mode drift remains in the inputs. This is one reason the strict
forward test fails. We prefer an honest failure to a leaky success.

---

## D4. Keep persistence as a first-class model

**Decision.** Include persistence in every table, paired comparison, and figure.

**Alternative.** Compare only against VAR, as is common in the literature.

**Why.** VAR is itself close to persistence on this data. Omitting persistence
would make a +0.004 R² look like progress. Persistence is also the correct
behavioural baseline because behaviour is strongly autocorrelated (label
persistence ≈0.97).

**Cost.** The headline is negative.

---

## D5. Do not tune on test; fix hyperparameters before the final suite

**Decision.** One configuration per regime; seeds vary, hyperparameters do not.

**Alternative.** Per-worm or per-model hyperparameter search scored on test.

**Why rejected.** That would make the already-small differences uninterpretable
and inflate the best model by selection. The suite is a comparison under a fixed,
recorded budget, not a leaderboard.

**Cost.** Some models (especially GRU) may be under-trained; the conclusion is
"under this budget", not "these architectures cannot work".

---

## D6. BunDLe-Net: evaluate official artifacts separately from our reimplementation

**Decision.** Report official leaky metrics and our blocked-holdout metrics in
separate tables, with explicit warnings.

**Alternative.** Report only the official high kNN accuracy, or only our low one.

**Why.** The official artifacts were fitted on all windows; their 0.95–0.98 kNN
accuracy has embedding-level leakage (the encoder saw all frames, including the
test frames). Our reimplementation is a partial reproduction with known
deviations and scores 0.407 with a causal filter (0.624 under the earlier
acausal zero-phase filter, which leaked future frames). Neither number alone is
honest; the drop is reported as a leakage finding.

**Cost.** No clean "we reproduced BunDLe-Net" claim is available.

---

## D7. NC-MCM: verify artifacts and port the test, do not claim full reproduction

**Decision.** Load distributed states, reimplement the Markov test, compare
p-values, and cross-validate the decoder.

**Alternative.** Retrain the full 28-regression + k-means pipeline from raw
calcium and call it a reproduction.

**Why.** Full retraining would require matching undocumented preprocessing and
restart settings; a mismatch would be uninterpretable. Artifact verification plus
an independent test port is a narrower, checkable claim.

**Cost.** We cannot say the NC-MCM training pipeline was reproduced end-to-end.

---

## D8. Branching is defined operationally, with a shuffled null

**Decision.** Local outcome entropy in a stated projection at a stated horizon,
compared with a permutation null and across projection dimensions.

**Alternatives.** Visual inspection of 3-D trajectories; fitting a switching
linear dynamical system and calling its modes "branches"; using the presence of
multiple future behaviours as proof of bifurcation.

**Why.** The first is not quantitative, the second conflates model with mechanism,
and the third ignores that behavioural outcomes are stochastic and
label-autocorrelated. The null test is essential: without it, entropy 0.58 bits
sounds like branching; against a null of 0.96 bits it is *less* diverse than
chance.

**Cost.** The analysis cannot discover a bifurcation; it can only fail to find
excess diversity.

---

## D9. Virtual perturbations are paired and include random-neuron controls

**Decision.** Common random numbers, cluster bootstrap over initial states,
multiple simulation seeds, OOD flag, and a control distribution sampled from
non-candidate named cells.

**Alternatives.** Independent baseline/intervention noise; reporting only the
largest effect; selecting "interesting" neurons post hoc.

**Why.** Independent noise makes small effects unidentifiable. Without a control
distribution, any cell can look special. The first version of the control code
sampled from the full predictor pool **including candidates**; red-teaming caught
this and it was fixed. The corrected controls show the top candidate at the 79th
percentile, p ≈ 0.21, so the candidate effect is not specific.

**Cost.** The headline intervention result is negative and less exciting.

---

## D10. Keep the empirical-validation gate closed

**Decision.** No output is labelled validated without a real, provenance-checked,
controlled intervention dataset.

**Alternative.** Use the AVA-HisCl dataset as if it validated the virtual
perturbations.

**Why rejected.** AVA is excluded from predictors, the HisCl dataset is a
different preparation and perturbation, and matching it to a model edit would
require assumptions we cannot test here. Analysing it could be a *separate*
observational comparison, not validation of this simulator.

**Cost.** The project cannot claim translational or causal impact.

---

## D11. Preserve negative results in the primary documentation

**Decision.** Negative results appear in the README, results, supervisor brief,
and paper prospectus, not in a buried appendix.

**Alternative.** Foreground the attractive 3-D embedding and the top-ranked RIVL
effect.

**Why.** The attractive embedding is upstream-fitted and leaky, and the RIVL effect is
nonspecific. Presenting either as a discovery would be misleading. The negative
results are also the most useful information for the collaboration: they identify
where the methods fail and what experiment would discriminate.

**Cost.** The deliverable is less impressive at first glance and more defensible.
