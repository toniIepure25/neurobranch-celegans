# Master Review: Biological Interpretation

**Standing labels that apply to this entire document:**

```text
PROBABILISTIC CALIBRATION NOT ESTABLISHED
MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.
Biological intervention validation: NOT_EVALUABLE.
```

---

## 1. The biological object

Kato et al. (2015), *Cell* 163:656–669, DOI `10.1016/j.cell.2015.09.034`, recorded whole-brain calcium activity in immobilised *C. elegans* and reported that global brain dynamics embed the motor command sequence. The `WT_NoStim` subset contains five wild-type, unstimulated animals, each recorded for about 18 minutes at roughly 2.8–3.1 Hz.

The archive’s acquisition names include:

```text
lite-1_punc-31_NLS3_..._1mMTet_basal_1080s
```

The `1mMTet` token is consistent with tetramisole, an immobilisation agent; `basal` indicates no stimulus; `1080s` is the duration. The archive does not contain a methods paragraph that independently verifies the immobilisation protocol, so this reading is an inference from the acquisition name, not a primary-methods quotation.

The `States` vector has eight labels:

```text
FWD, SLOW, DT, VT, REV1, REV2, REVSUS, NOSTATE
```

These are motor/behavioural states. The readme states that they are behavioural categories; the project documentation correctly notes that they are inferred from command/interneuron calcium and behaviour. They are not an independent behavioural ground truth.

---

## 2. What the experiment can and cannot see

### 2.1 It is not “no behaviour”

`WT_NoStim` means no optogenetic stimulus. It does **not** mean no behaviour, no sensory input, or no internal state. The animal is immobilised, but the recorded motor command system is still active; the state labels include forward, reversal and turn categories. The brain is not a closed, stimulus-free system.

This matters for the residual B3 advantage. A model that predicts neural activity better after accounting for ten common factors may be capturing:

- a behaviour-related common drive that is not fully represented by ten linear factors;
- a slow motor state;
- a sensory/arousal variable;
- a nonlinearity in the calcium signal;
- a genuine cross-neuron interaction.

The current design cannot separate these.

### 2.2 It is calcium imaging, not voltage

Calcium indicators are slow relative to synaptic and membrane dynamics. At ~2.9 Hz, one frame is ~0.34 s. A lag-one VAR describes an effective transition at the imaging timescale, not a synaptic weight. Calcium buffering, indicator kinetics and bleaching correction all shape the observed signal.

Therefore:

- a VAR coefficient is an **effective predictive coefficient**, not a synaptic strength;
- a spectral radius is a property of the fitted effective operator, not a biophysical time constant;
- a “directed interaction” from the VAR is not a gap junction or chemical synapse;
- a “common factor” is a statistical component, not a measured neuromodulator or command neuron.

### 2.3 It is one laboratory cohort

The five animals are from one laboratory and one experimental protocol. They are not an independent cohort. The publication reports five animals; the archive has no animal key; the mapping is inferred. Even if the five are distinct animals, they do not represent the species or a new cohort.

### 2.4 It has no intervention

There is no matched stimulation, ablation, sham or rescue in `WT_NoStim`. The project’s virtual perturbations are mathematical displacements inside a fitted model. They are not biological interventions.

This is why the standing label is not rhetorical:

```text
Biological intervention validation: NOT_EVALUABLE.
```

---

## 3. What the Kato result actually supports

The strongest biologically relevant statement is:

> In five Kato `WT_NoStim` recordings, a full lag-one population model has a modest H20 PCA-10 energy advantage over a matched own-history-plus-ten-factor control. The advantage is horizon-dependent, absent for mean MSE, and not established at the animal level. It is compatible with residual cross-neuron predictive structure, an unmeasured common drive, behaviour, nonlinearity, nonstationarity, or estimator capacity.

This is a statement about **predictive structure in the recorded population**, not about a mechanism.

### 3.1 What “residual cross-neuron structure” means

B5 predicts each neuron from:

- its own current value;
- ten training-only common factors;
- an intercept.

B3 predicts each neuron from:

- all neurons’ current values;
- an intercept;
- a full `n × n` operator.

The B3 advantage means that the full operator predicts better than the restricted one on this metric. It does **not** mean that the extra coefficients are direct connections. The extra coefficients can encode:

- a shared input not captured by the ten factors;
- a nonlinear function of the common state;
- a behaviour variable;
- a time-varying gain;
- a recording artifact;
- individual neuron heterogeneity;
- overfitting that happens to help on this test sample.

The synthetic identifiability audit makes the point concrete: a common-input system with no direct coupling still produces spurious off-diagonal structure, and a full model can forecast **worse** than a diagonal model (`MSE ratio 1.556`). In the Kato data, the off-diagonal coefficients are unstable across chronological segments (`r = 0.052`). A predictive advantage and a stable interaction graph are different objects.

### 3.2 What the common-factor control means

The ten factors explain `54.7–71.0%` of training variance. They are statistical components, not measured inputs. The common-factor increment is `+0.2751` energy; the residual increment is `+0.2644`. Under the ordered B4m→B5→B3 decomposition, that is `51.0%` common and `49.0%` residual.

This does **not** mean “half the brain’s predictive structure is a common input.” It means:

> Under one ordering and one baseline, ten linear factors account for about half of the population model’s energy gain over the matched own-history model; the other half is residual to those factors.

Different baselines give different shares (`23.5/76.5` vs B1m; `72.4/27.6` vs B0). The split is descriptive, not structural.

### 3.3 What the horizon profile means

B3 is worse at H1 (`-0.0637`), better at H5 (`+0.0835`), H10 (`+0.1471`) and H20 (`+0.2644`). The advantage grows with horizon.

Biologically, this is consistent with a model that captures a slow population state which takes several frames to express. It is also consistent with a model that propagates variance differently, or with a model whose short-horizon predictions are dominated by a strong persistence/own-history component.

The data do not distinguish “slow biological state” from “slow statistical error accumulation.” A horizon profile is a clue, not a mechanism.

---

## 4. The behaviour-label circularity

The eight states are derived from calcium in command/interneuron populations and behaviour. The project excludes eight label-defining names:

```text
AVAL, AVAR, SMDDR, SMDDL, SMDVR, SMDVL, RIBR, RIBL
```

This reduces direct circularity, but it does not make the labels independent:

- the remaining neurons are co-recorded in the same animal;
- the annotators used the worm’s behaviour;
- the states are not external ground truth;
- the excluded list is not present in every recording (session 0 lacks `SMDDL` and `RIBR`).

Therefore:

- decoding the labels is an **observational consistency** check, not independent validation;
- a high decoding score does not prove that the model “knows” the behaviour;
- a low decoding score does not prove the brain is unrelated to behaviour.

The B3/B5 result is a neural forecasting result, not a behaviour-decoding result. But behaviour remains a possible common drive behind the residual advantage. The current analysis does not include a measured behaviour regressor in B5, because the labels are not independent and the archive does not provide a clean external behaviour trace.

---

## 5. What cannot be claimed

The following sentences are not supported by this dataset and analysis:

- “B3 identifies direct neuron-to-neuron interactions.”
- “The off-diagonal operator is a functional connectome.”
- “The common factors are command neurons or neuromodulators.”
- “The model recovers the motor command sequence.”
- “Virtual perturbation of neuron X predicts a biological effect.”
- “The result generalizes across animals.”
- “The result establishes a causal role for population dynamics.”
- “The model is a digital twin.”
- “The model is calibrated.”
- “The five animals are an independent cohort.”

Some of these are stronger than others. The first four are mechanistic overreads; the next two are causal overreads; the last four are evidence-status overreads.

---

## 6. Alternative biological explanations

### H1: direct coupling

The residual structure is a real network interaction: neurons influence each other through synapses or gap junctions beyond the common factors.

**Prediction:** a capacity-matched model with residual cross-neuron rows should retain the advantage; perturbation of a source neuron should produce a target response that replicates across fits and animals.

**Current evidence:** compatible, not established. The operator is unstable; virtual responses disagree across fits (median cosine `0.556`; sign reversals). No intervention.

### H2: unmeasured common input

A third variable drives many neurons; the full operator captures its residual covariance, not direct coupling.

**Prediction:** measuring the common input and including it in B5 should remove the residual advantage; a common-input synthetic system should reproduce the pattern.

**Current evidence:** compatible. The synthetic common-input scenario produces spurious off-diagonal structure. The ten factors may not capture a nonlinear or time-varying common input.

### H3: behaviour / motor state

The residual structure is a behaviour-related state that is not fully captured by ten linear factors.

**Prediction:** adding a measured behaviour regressor or stratifying by state should reduce the advantage; the advantage should concentrate in behaviourally homogeneous periods.

**Current evidence:** compatible. The labels are circular and not used as a clean regressor; no external behaviour trace is in `WT_NoStim`. The session-2 reversal and horizon dependence could reflect state composition.

### H4: nonlinearity / calcium kinetics

The linear VAR captures a nonlinear calcium response or indicator dynamics that the factor model misses.

**Prediction:** a nonlinear model with the same capacity should match or exceed B3; deconvolved spike estimates should change the result; different calcium indicators should change it.

**Current evidence:** compatible. Only linear models were compared. The data are ΔF/F, not spikes.

### H5: nonstationarity / adaptation

The rolling fit adapts to slow changes in the preparation; B3’s larger capacity exploits this more than B5.

**Prediction:** a fixed train-only model should show a smaller advantage; a time-varying parameter model should explain the residual; session-level heterogeneity should be large.

**Current evidence:** compatible. The models are refit on test-past; session 2 reverses; the operator is unstable across segments.

### H6: estimator capacity

B3 has 7.96–9.95× more parameters and arbitrary residual rows. If capacity alone explains the result, the advantage is an estimator effect, not information.

**Prediction:** a capacity-matched restricted model should lose the advantage; a B5 with a local factor basis and matched capacity should close the gap.

**Current evidence:** unresolved and central. No capacity-matched model was prespecified. This is the most important alternative to rule out.

---

## 7. Discriminating experiments

### 7.1 Capacity-matched model (no new data)

Fit a restricted B3 with a rank-`k` residual operator and a parameter budget matched to B5. Fit a B5 with a local factor basis and the same adaptivity as B3. Prespecify the comparison. If the advantage survives, capacity is not sufficient; if it disappears, the current claim is a capacity effect.

This is the cheapest and most direct next step. It uses existing data and does not require a new animal.

### 7.2 Calibration-matched ensemble

Construct B5 and B3 ensembles with matched marginal spread, or evaluate a score that separates calibration from resolution (e.g. a proper score decomposition or a spread-conditioned comparison). If the energy advantage disappears when spread is matched, the result is a calibration effect.

This is also possible with existing artifacts if per-landmark ensemble members are saved. If they are not, a targeted rerun at a subset of landmarks is enough.

### 7.3 Independent cohort

Record or obtain a new cohort of `WT_NoStim` animals, with animal-level splits and a prespecified analysis. Five animals are not enough for a stable animal-level estimate; 10–20 would be a minimum for a meaningful interval.

This is the only way to support generalization.

### 7.4 Measured common input

If the common drive is sensory (e.g. oxygen), measure and manipulate it. If it is behavioural, record behaviour concurrently. If it is neuromodulatory, measure or perturb the relevant system. Include the measured input in B5 and test whether the residual advantage survives.

### 7.5 Intervention with sham

For a causal biological claim, use cell-specific stimulation or ablation with matched sham and a prespecified target. The current virtual perturbation is not a substitute. A +1 SD displacement in a fitted model is not a biological dose.

### 7.6 Cross-animal alignment

Use NeuroPAL or another identity system to align identified neurons across animals, then test whether the residual structure is conserved. The current numeric traces cannot be aligned; the named subset is only 31–47 neurons per recording.

### 7.7 Voltage or spike imaging

Calcium is slow. Voltage imaging or electrophysiology at a faster timescale would test whether the residual structure is a synaptic interaction or a calcium/indicator effect.

---

## 8. What the biological literature does and does not add

Kato et al. established that global brain dynamics embed motor commands. That is a **descriptive** finding about population activity and behaviour. It does not establish a particular interaction graph, and it does not validate a VAR.

BunDLe-Net learns a behaviourally relevant latent manifold from the same data. It is a representation-learning result, not a causal result.

NC-MCM tests whether cognitive states are causally consistent abstractions using Markovianity. It is an observational abstraction test, not an intervention.

The current project’s B3/B5 contrast is closest to a **predictive ablation** of a common-factor model. It is not a replication of Kato, BunDLe-Net or NC-MCM. It uses the same data but asks a different question.

See [Literature comparison](MASTER_REVIEW_LITERATURE_COMPARISON.md) for the detailed table.

---

## 9. The strongest allowed biological sentence

> In five Kato `WT_NoStim` recordings, a full population VAR retains a modest, horizon-dependent H20 energy advantage over an own-history-plus-ten-factor control. The advantage is compatible with residual cross-neuron predictive structure, but the observational design, the unstable off-diagonal operator, the capacity mismatch, the calibration difference, and the absence of intervention data prevent a biological interpretation as direct coupling, a command circuit, or a stimulation target. Biological intervention validity is `NOT_EVALUABLE`.

This sentence is long because the claim is narrow. Shortening it usually makes it false.

---

## 10. What a biologically literate reader should ask

1. Is the animal behaviourally active? If yes, what behaviour was measured concurrently?
2. Is the calcium signal deconvolved? If not, how do indicator kinetics affect the lag-one operator?
3. Are the five animals one cohort? How were they selected?
4. Is the “common factor” a measured input or a PCA component?
5. Does the residual advantage survive a capacity-matched model?
6. Does it survive a calibration-matched ensemble?
7. Does it replicate in a new cohort at the animal level?
8. Is there any intervention, sham, or rescue?
9. Are the named neurons aligned across animals?
10. What would falsify the coupling interpretation?

A paper that cannot answer 5–8 has a predictive result, not a biological mechanism.
