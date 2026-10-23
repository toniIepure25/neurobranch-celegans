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

## 7. What not to do next

- Do not run more seeds on the same blocked split and call the residual +0.011
  significant. It is within seed noise.
- Do not present the official BunDLe 3-D embedding as a held-out result.
- Do not use the AVA-HisCl dataset as "validation" of these virtual perturbations.
  It is a different perturbation and AVA is excluded from the predictors. It could
  support a separate observational comparison.
- Do not add a 3-D animation and call it a bifurcation.
- Do not tune the intervention target on the test worm and then report a
  confirmatory p-value.
