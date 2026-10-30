# Next Experiments

The highest-priority experiment is the one that can turn the central negative
result into a discriminating test. It is a **real biological intervention**, not
another analysis of the same observational file.

---

## 1. Highest priority: causal test of a model-ranked perturbation

### Question

Does increasing activity in RIVL (and, separately, RIVR) during a spontaneous
reversal causally increase the probability of a reversal-to-turn transition within
4 s, relative to matched sham and random-neuron controls?

This is the cleanest test because:

- RIVL was the top model-ranked candidate (virtual ΔP(TURN)=+0.333).
- RIVR was second (+0.233), giving an internal replication target.
- The effect is **not specific** in the model (4/19 random controls are larger),
  so a negative result is informative about the model, and a positive result must
  clear a high bar.
- Reversal termination is a discrete, time-stamped behavioural event that can be
  scored blind.

### Design

| Element | Specification |
|---|---|
| Organism | *C. elegans* young adults, same temperature and cultivation as Kato et al. |
| Preparation | Animals behaving in a microfluidic arena with head-restrained or tracked whole-brain calcium imaging; behaviour recorded at ≥30 Hz |
| Perturbation | Cell-specific optogenetic activation (red-shifted channelrhodopsin) of RIVL or RIVR, triggered during spontaneous REV bouts; a validated RIV-specific driver to be selected with the Vienna collaborators |
| Conditions | (1) RIVL activation, (2) RIVR activation, (3) random non-candidate neuron activation with the same opsin and light dose, (4) opsin-only sham light, (5) unperturbed trials in the same animal |
| Randomization | Trial order randomized within animal; light pulse timing randomized within the first 2 s of REV; scorer blind to condition and genotype |
| Primary outcome | `P(TURN within 4 s | REV bout)` |
| Secondary outcomes | REV bout duration; turn direction (DT vs VT); latency to turn; calcium response in RIV and downstream cells |
| Sample size | 12 animals per condition, ≥30 REV trials per animal, powered by simulation with animal-level clustering (ICC assumed 0.2); exact N fixed before data collection |
| Blinding | Genotype/promoter identity masked until primary analysis is locked |
| Pre-registration | OSF registration with the model prediction, primary outcome, exclusion rules, and analysis code before data collection |

The model prediction to register is **directional**, not a point estimate:
"RIVL activation increases P(TURN within 4 s | REV) relative to sham and relative
to random-neuron activation." The model's +0.333 is not a dose prediction and
must not be presented as one.

### Controls that make it discriminating

1. **Sham light** controls for heat, light, and handling.
2. **Random-neuron opsin** controls for opsin expression, membrane disruption, and
   general excitation. This is the control the model analysis lacked in its first
   (buggy) version.
3. **Contralateral RIVR** tests anatomical specificity.
4. **Within-animal unperturbed trials** control for baseline turn propensity.
5. **Calcium imaging of the target** confirms the perturbation engaged the cell
   and does not merely trigger a global response.
6. **A second, independent cohort** tests replication; the first cohort is not
   used to tune the analysis.

### Decision rules

| Outcome | Interpretation |
|---|---|
| RIVL effect > sham **and** > random-neuron control, replicated in cohort 2 | Specific causal effect; the model's ranking is supported in direction, not magnitude |
| RIVL effect > sham but **not** > random-neuron control | Nonspecific excitation effect; the model's candidate ranking is not validated |
| No effect vs sham | The model's predicted effect fails; the virtual perturbation is not causally transportable |
| Effect in opposite direction | Model sign error; strong falsification |
| Effect only at high OOD / high light dose | Extrapolation or heat artifact; not supportive |

A negative result is publishable and useful: it would show that a model-ranked
perturbation does not survive a controlled biological test, which is exactly the
kind of evidence the field needs.

### Feasibility and risks

- **Driver availability.** A RIV-specific promoter may not exist or may label
  additional cells. The experiment should begin with a driver characterization
  stage (confocal imaging + calcium) before the behavioral cohort.
- **Optogenetic heat.** Red-shifted opsins reduce but do not eliminate heating;
  sham and random-neuron controls are essential.
- **Immobilization vs behaviour.** Whole-brain imaging in behaving worms is
  difficult. A staged design is acceptable: first a behavioural optogenetics
  cohort (no imaging) with the same trial structure, then an imaging cohort to
  confirm target engagement.
- **Power.** Behavioural turn probability is low and variable; the trial count
  must be based on the observed baseline, not on the model's 0.133.
- **Lab access.** This requires a wet-lab collaboration; NeuroBranch cannot
  perform it. The deliverable here is the protocol, the prediction, and the
  analysis code, not the data.

### Why this is higher priority than more modelling

More modelling on the same five worms cannot distinguish "no dynamics" from
"non-stationary dynamics" or "wrong model class". A randomized intervention can.
It also directly tests the only potentially exciting output of the virtual scan
and is designed to fail convincingly if that output is an artifact.

---

## 2. Second priority: time-varying dynamics

**Question.** Does allowing parameters to vary over time recover forward
prediction?

**Design.** Fit per-block VAR/MLP models, a switching linear dynamical system with
a small number of regimes, or a time-varying state-space model. Evaluate on the
strict-forward split. Compare against persistence and against a model with
time as an explicit input.

**Decision rule.** A model must beat persistence on the final 15% for at least 3/5
worms, with paired CIs, to support the non-stationarity explanation. If no model
does, the "dynamics are absent or unidentifiable" interpretation gains weight.

---

## 3. Third priority: independent behavioural labels

**Question.** Are the negative decoding and branching results caused by label
circularity/coarseness?

**Design.** Re-annotate a subset of the Kato recordings from video or obtain an
independent behavioural tracking dataset. Compare label agreement and repeat the
decoding/branching analyses with the independent labels.

**Decision rule.** If decoding improves above label persistence with independent
labels, the original negative result is partly an annotation artifact. If it does
not, the conclusion is strengthened.

---

## 4. Fourth priority: external replication

**Question.** Do the results transport to another lab, preparation, or stimulus
condition?

**Design.** Apply the exact pipeline to `WT_Stim.mat` (controlled oxygen stimulus)
and to an independent whole-brain dataset. Do not tune on the new data. Report
the same gates.

**Caveat.** `WT_Stim.mat` has four behaviour states, not eight; the macro mapping
must be fixed in advance. This is a replication test, not a validation of the
interventions.

---

## 5. Fifth priority: branching with a better null

**Question.** Is the negative branching result robust to the null construction?

**Design.** Run all five worms with (a) the outcome permutation, (b) a
block-preserving permutation that shuffles futures within behaviour and time
blocks, and (c) a generative null from a fitted first-order Markov model. Compare
observed entropy to all three. Pre-specify the primary null.

**Decision rule.** Excess entropy > 0 with p < 0.05 under the block-preserving
null in at least 3/5 worms would reopen the branching question. Otherwise the
negative result stands.

---

## 6. Sixth priority: dose–response and model-agnostic replication of the scan

**Question.** Is the RIVL effect stable across models and amplitudes?

**Design.** Repeat the virtual scan with VAR, MLP, residual MLP, and latent kNN
checkpoints; amplitudes −3, −2, −1, +1, +2, +3 SD; report the full dose–response
and the random-control distribution for each model.

**Decision rule.** A candidate that is top-ranked in only one model and one
amplitude is not a robust hypothesis. A candidate that is consistently above the
control distribution across models is worth a biological test.

---

## 7. Phase 9 priorities: measure better before modelling more

**Question.** Is the low single-animal reliability a property of the response
summary or of the atlas?

**Design.** For the 20 best-sampled targets, recompute reliability using
(a) response integral, (b) peak latency, (c) baseline-corrected z-score and
(d) a wider post-stimulus window. Report all four against the same animal
filter. Then run a stimulation-budget pilot: five targets with at least five
repeated stimulations per animal in a new recording, and test whether
reliability approaches the predicted value from the design formula in
`docs/MEASUREMENT_LIMITS.md`.

**Decision rule.** If no summary raises single-animal reliability above 0.4,
the atlas is the limit and individual-level modelling should stop. If one
summary does, adopt it and re-run Phases 6-8 under the new ceiling.

**Cost.** Low for the re-analysis; high for new recordings.

---

## 8. Design engine, only after the ceiling is stable

**Question.** Which targets should be stimulated next to maximise information
about a candidate mechanism?

**Design.** Implement the EIG portfolio with simulation-based calibration
(SBC) on synthetic atlases where the true profile is known. Compare EIG,
uncertainty sampling and random selection at matched budget. Report coverage of
the posterior intervals, not only point accuracy.

**Decision rule.** The design engine ships only if SBC shows nominal coverage
and EIG beats random by more than the design MDE at the same budget. A design
optimiser built on an unverified ceiling would optimise noise.

---

## 9. What not to do next

- Do not run more seeds on the same blocked split and call the residual +0.011
  significant. It is within seed noise.
- Do not present the official BunDLe 3-D embedding as a held-out result.
- Do not use the AVA-HisCl dataset as "validation" of these virtual perturbations.
  It is a different perturbation and AVA is excluded from the predictors. It could
  support a separate observational comparison.
- Do not add a 3-D animation and call it a bifurcation.
- Do not tune the intervention target on the test worm and then report a
  confirmatory p-value.


## Phase 10 follow-ups

1. **Two-way resampling.** Implement an animal x responder cluster bootstrap or
   jackknife and test nominal coverage on synthetic atlases. The current animal
   jackknife covers about 60% at nominal 95%.
2. **Prospective design validation.** Use the animal-budget calculator to choose
   a target count for a new cohort, then compare predicted and achieved
   reliability. Until then the calculator is a planning approximation.
3. **Target ranking stability.** The current ranking does not replicate across
   animal halves. Test whether a shrinkage or hierarchical reliability model
   stabilises it before using rankings for prioritisation.
4. **Summary choice.** `response_residual` ranks targets similarly to
   `response_mean` (Spearman 0.934) but not identically; decide a primary
   summary prospectively rather than selecting the best-looking one.
5. **Extrapolation audit.** For the 87 targets with fewer than 22 animals, test
   whether the fitted curve predicts held-out reliability at the observed
   maximum; this is the only honest check of the extrapolated region.


## 10. Coverage-balanced reanalysis (Phase 11 follow-up)

**Question.** Is the pooled reliability ceiling a property of the atlas, or of
the responder-coverage pattern?

**Design.** Fit a hierarchical model with an explicit responder-observation
indicator and animal random effects, then compare its implied ceiling at 80%
coverage with the moment estimate (0.560). Evaluate with a prospective split:
hold out animals, not responders, and check posterior predictive coverage on the
held-out animal profiles. Report the full coverage-sensitivity curve, not one
point.

**Decision rule.** Adopt the hierarchical ceiling only if (a) its synthetic
coverage is in [0.93, 0.97], (b) it reproduces the 0.560 point estimate within
0.05, and (c) the animal-holdout predictive interval contains the observed
profile correlation. If it fails, keep 0.560 as a descriptive bound and do not
rank targets.

**Cost.** Medium: one modelling pass plus a calibration simulation. No new
animals required.

**Kill criterion.** If the model cannot reproduce the coverage sensitivity
without target-specific tuning, stop; the moment estimate remains the
transparent bound.

**Phase 12/12R outcome (2026-10-30).** The aligned Phase 12R model reproduced the
point estimate (0.304 vs 0.314; ceiling 0.551 vs 0.560) but failed the calibration
gates: SBC rank KS p = 1.43e-06, held-out PPC coverage 0.904 with median absolute
error 0.207, and a within-animal responder-permutation control with median
R(22) = 0.120 and q95 = 0.594. No target ranking or calibrated interval was
published; 0.560 remains the bound. The old benchmark cluster claim is withdrawn
because baseline/model truth was not paired on events (C11). See
`docs/PHASE12R_RESULTS.md`, `docs/CORRECTIONS.md` C10/C11.

**Phase 13 outcome (2026-10-30).** The projection-consistent event-variance model
(M2) removed the `1e-8` floor and lowered the pooled estimate from 0.3035 to
0.2003 (ceiling 0.4475), but failed six frozen gates: SBC coverage 0.9244, SBC
rank KS p = 0.00306, held-out PPC not run, negative-control median R(22) = 0.1485
(q95 0.4680), sign-flip median 0.1341 (q95 0.5796), and the Phase 11 agreement
gate (0.1137 away). The centering-only model M1 gave 0.3138 but also failed SBC
rank. No target ranking or calibrated interval was published; 0.560 remains the
bound. See `docs/PHASE13_RESULTS.md` and `research/phase13_deviations.md`.

**Next model (not yet run).** The next protocol must make `sigma = 0` credible
under the within-animal responder-permutation null before fitting real data. The
minimal test is a spike-and-slab or zero-inflated prior on the between-target
variance; the more principled test treats responder detection/coverage as part
of the generative process rather than conditioning on the observed mask. Require
the negative-control median R(22) to be at most 0.05, and the sign-flip q95 at
most 0.15, in a pre-registered run before interpreting any real-data posterior.
A model that fails the permutation control must be rejected even if its point
estimate matches the moment estimator. This is a new protocol, not a Phase 13
retry.
