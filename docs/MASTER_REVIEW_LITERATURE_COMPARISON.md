# Master Review: Literature Comparison

**Scope:** methodological orientation, not a systematic review. No PRISMA screening, no exhaustive citation chasing, no meta-analysis.  
**Rule:** every empirical claim below is attributed to the cited source or to a locally verified artifact. The project’s own result is never described as a replication of a paper it did not rerun.

---

## 1. The nearest neighbours

Four literatures matter:

1. **Kato-style whole-brain calcium imaging** — what the data are and what was originally claimed.
2. **Latent-manifold / behaviourally relevant representation learning** — BunDLe-Net.
3. **Causal-abstraction / Markov tests** — NC-MCM.
4. **Intervention and turn-direction circuits** — Leifer-style optogenetics, Kramer/SAAV.

The current project sits between (1) and (2): it asks whether a full population predictor retains information beyond a common-factor control. It is not a latent-manifold method, and it is not a causal-abstraction method.

---

## 2. Comparison table

| Work | Question | Data | Method | Evaluation | Causal status | Relation to this audit |
|---|---|---|---|---|---|---|
| **Kato et al. 2015**, *Cell* 163:656–669, DOI `10.1016/j.cell.2015.09.034`, PMID `26478179` | How do global brain dynamics relate to motor command sequences? | Five immobilised wild-type worms, no stimulus, ~18 min each, ~2.9 Hz | Whole-brain calcium imaging; low-dimensional population dynamics; eight motor states | Descriptive mapping of population states to motor commands | Observational; no perturbation in `WT_NoStim` | The dataset and the biological question. The current B3/B5 result is a reanalysis, not a replication of the paper’s mechanism. |
| **Kumar et al. 2023**, bioRxiv, DOI `10.1101/2023.08.08.551978` | Can a behaviourally relevant latent manifold be learned from population activity? | Kato `WT_NoStim`, five worms | BunDLe-Net: window encoder, residual latent transition, eight-class readout, weighted commutativity | Dynamic prediction error vs autoregressive baseline; behaviour decoding; comparisons to PCA/t-SNE/AE/CEBRA | Representation learning; no perturbation | The closest latent-manifold baseline. It asks a different question: behaviourally relevant latent structure, not a capacity-controlled predictive ablation. |
| **Grosse-Wentrup et al. 2024**, *PLoS Comput Biol*, DOI `10.1371/journal.pcbi.1012674`, PMID `39680605` | Are cognitive states causally consistent abstractions of neuronal activity? | Kato `WT_NoStim`, five worms, 107–131 neurons reported | NC-MCM: pairwise logistic regressions → k-means → Markovianity test | 10-fold CV decoding; Markovianity over k=1…20, 100 restarts, 1,000 surrogates | Observational abstraction; labels from motor-neuron activity; no intervention | The closest causal-abstraction test. It tests a discrete Markov property; the current project tests a continuous predictive ablation. Neither identifies direct coupling. |
| **Atanas et al. 2023**, *Cell*, DOI `10.1016/j.cell.2023.07.035` | How does brain-wide activity encode motor programs in freely moving worms? | >30 recordings, WormWideWeb | Per-neuron encoding models; connectome-class atlas | Cross-validated encoding | Observational | Natural external validation set; not used in this audit. It shows the importance of freely moving behaviour, which `WT_NoStim` lacks. |
| **Leifer et al. 2011**, *Nat Methods*, DOI `10.1038/nmeth.1554` | Can specific neurons be controlled optogenetically in freely moving worms? | Freely moving transgenics | ChR2/NpHR tracking optogenetics | Acute behavioural effects of stimulation/inhibition | Interventional | The empirical standard for a biological intervention claim. The current project has no comparable data. |
| **Kramer et al. 2026**, *Nat Neurosci*, DOI `10.1038/s41593-026-02257-5` | What neural sequences underlie directed turning? | 73 recordings, 56 WT animals for the primary turn-direction set | SAAV / GRU decoders; animal-disjoint evaluation in the project’s reconstruction | Turn-direction AUROC/log loss | Observational; no trial-level optogenetics in the archive | The project’s Phase 4/5 benchmark. It produced a null, not a replication. |

The Kato and NC-MCM descriptions were verified through Crossref/Europe PMC records. The BunDLe-Net description follows the preprint abstract and the project’s literature review; it was not independently retrained. The Kramer description follows the local Phase 4 audit and the published record.

---

## 3. Kato: what is established, what is not

Kato et al. reported that global brain dynamics embed the motor command sequence. The strongest description is:

- population activity is low-dimensional;
- population states map onto motor commands/behaviour;
- the mapping is observable in single trials;
- the eight `WT_NoStim` states are a useful description.

The paper does **not** establish:

- a unique interaction graph;
- that a VAR is the correct dynamical model;
- that a full population operator identifies direct coupling;
- that a virtual perturbation predicts a biological effect;
- that the result generalizes beyond the five animals.

The current project’s B3/B5 contrast is a **predictive ablation** of a common-factor model. It is not a test of Kato’s mechanism. It uses Kato’s data to ask whether residual cross-neuron predictive structure exists after a strong control.

That is a legitimate question. It is narrower than “Kato’s global dynamics.”

---

## 4. BunDLe-Net: the closest latent-manifold baseline

BunDLe-Net’s contribution is a learned latent space in which transitions are approximately Markov and behaviourally relevant. Its losses include a dynamic prediction term, a behaviour readout, and a weighted commutativity term. It compares against autoregressive, PCA, t-SNE, autoencoder and CEBRA baselines.

Differences from the current B3/B5 analysis:

| Dimension | BunDLe-Net | Current B3/B5 |
|---|---|---|
| Object | Learned latent manifold | Fixed PCA-10 evaluation space |
| Transition | Residual latent transition, architectural Markov | Linear VAR, no Markov constraint |
| Behaviour | Eight-class readout | Not used as a primary target |
| Control | Autoregressive and representation baselines | Own-history + ten common factors, matched window/ridge/innovations |
| Evaluation | Dynamic prediction error, decoding | H20 energy, MSE, coverage |
| Causal claim | None | None |
| Data split | Not the same as this audit | Per-recording temporal split, 175 H20 landmarks |

The current analysis is **not** a benchmark against BunDLe-Net. It does not use the same splits, metrics or code. A sentence such as “B3 beats BunDLe-Net” would be unsupported. A sentence such as “B3 retains information beyond a ten-factor linear control” is supported only in the narrow, exploratory sense of this audit.

What BunDLe-Net adds conceptually:

- a reminder that “Markov by construction” is an architectural choice, not a discovered property;
- a reminder that behaviourally relevant latents can be low-dimensional without being causal;
- a baseline for future work if the same splits and metrics are used.

---

## 5. NC-MCM: the closest causal-abstraction test

NC-MCM asks whether cognitive states are causally consistent abstractions. It builds pairwise logistic regressions, clusters them into `k` states, and tests Markovianity with surrogates. It reports decoding and Markov tests across `k=1…20`.

Differences from B3/B5:

| Dimension | NC-MCM | Current B3/B5 |
|---|---|---|
| State | Discrete `k`-state abstraction | Continuous linear population state |
| Transition | Markov transition matrix | VAR operator |
| Test | Markovianity vs surrogates | Predictive energy vs common-factor control |
| Control | Model selection over `k`, restarts | Fixed ten factors, fixed window/ridge |
| Causal claim | Observational abstraction, not intervention | Observational prediction, not intervention |
| Main limitation | Fixed `k`, linear pairwise model, labels from motor neurons | Capacity, calibration, reused test, five units |

The two methods are complementary:

- NC-MCM tests a **discrete Markov property**;
- B3/B5 tests a **continuous predictive residual**.

Neither identifies direct coupling. A positive B3/B5 result does not refute NC-MCM; a negative NC-MCM result does not refute B3/B5. They use different abstractions and different nulls.

The current project’s `G7 structural identifiability` failure is consistent with NC-MCM’s own caveat: an observational abstraction can be predictive without being causal.

---

## 6. Intervention literature: why the biological claim is not evaluable

Leifer et al. established the standard: cell-specific optogenetic stimulation/inhibition in freely moving animals, with behavioural readouts and controls for light/heat/state. Kramer et al. provide circuit-level turn-direction data, but the public archive lacks trial-level optogenetic recordings.

The current project has:

- no stimulation;
- no ablation;
- no sham;
- no rescue;
- no matched behavioural readout;
- only a model-dependent virtual displacement.

A virtual perturbation is a mathematical operation:

```text
z_0 -> z_0 + delta
```

It answers: “what does this fitted model predict if the state is displaced?” It does not answer: “what happens if the neuron is stimulated?” The gap is not a matter of more simulations. It requires biological intervention data.

The standing label is therefore correct:

```text
MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.
Biological intervention validation: NOT_EVALUABLE.
```

---

## 7. Dynamical-systems literature: what would be a stronger model

The current B3 is a linear lag-one VAR. The literature offers several alternatives that would test different hypotheses:

| Class | Example | What it would test | Why it is not a fix by itself |
|---|---|---|---|
| Recurrent switching linear dynamical systems | Linderman et al., arXiv 2016 | Discrete modes with per-mode linear dynamics | Fixed mode count; linear within modes; no causal identification |
| Deep Markov models | Krishnan et al., arXiv 2016 | Nonlinear latent state with explicit confound | Identifiability unestablished; more capacity |
| Neural ODEs | Chen et al., NeurIPS 2018 | Continuous-time vector field | Deterministic; needs timescale assumptions |
| Neural SDEs | continuous-time stochastic models | Divergence surviving noise vs noise-driven | Requires careful calibration; more parameters |
| Markov-switching VARs | classical regime-switching | Time-varying dynamics | Mode count and switching law are assumptions |

A more flexible model can improve prediction without identifying mechanism. The current audit’s synthetic common-input result is a warning: a flexible model can produce spurious off-diagonal structure and forecast worse than a diagonal model. More capacity is not a causal identification strategy.

---

## 8. Where the current result sits

| Question | Kato | BunDLe-Net | NC-MCM | Current B3/B5 |
|---|---|---|---|---|
| Is population activity low-dimensional? | Yes, descriptive | Yes, learned | Yes, discrete | Yes, PCA-10 |
| Is behaviour decodable? | Yes | Yes | Yes | Not primary |
| Is there predictive structure beyond own history? | Not tested this way | Yes, vs AR | Not this metric | Yes, exploratory |
| Is there structure beyond common factors? | Not tested | Not this control | Not this control | Yes, exploratory |
| Is the structure capacity-controlled? | No | No | No | **No** |
| Is it calibrated? | Not claimed | Not claimed | Not claimed | **No** |
| Is it animal-level? | Descriptive | One dataset | One dataset | **No** |
| Is it causal? | No | No | Observational | **No** |
| Is there intervention validation? | No | No | No | **No** |

The current result adds a useful, narrow piece: a common-factor-controlled predictive contrast on the Kato data. It does not replace any of the neighbouring methods.

---

## 9. What a fair literature-facing claim looks like

Allowed:

> On the Kato `WT_NoStim` data, a full lag-one population VAR retains a modest H20 PCA-10 energy advantage over an own-history-plus-ten-factor control. This is a predictive, within-cohort result. It is not a replication of Kato’s mechanism, not a benchmark against BunDLe-Net or NC-MCM, and not a causal identification.

Not allowed:

- “B3 captures the global brain dynamics reported by Kato.”
- “B3 outperforms BunDLe-Net.”
- “B3 refutes NC-MCM’s Markovianity.”
- “B3 identifies the motor command circuit.”
- “B3 is a digital twin.”
- “The result is consistent with direct coupling, therefore direct coupling is likely.”

The last sentence is a common inferential error: consistency with a hypothesis is not evidence for it when many other hypotheses are equally consistent.

---

## 10. What would make a real comparison possible

To compare against BunDLe-Net or NC-MCM fairly, the project would need:

1. the same data split and landmarks;
2. the same evaluation metric and forecast horizon;
3. the same behaviour readout, if behaviour is part of the claim;
4. the published code or a faithful reimplementation;
5. a capacity-matched control;
6. an animal-level interval;
7. a preregistered analysis.

Without these, the correct statement is “a different question on the same data,” not “better than.”

---

## 11. Literature conclusion

The current project is best understood as a **common-factor-controlled predictive ablation** on Kato’s data. Its nearest neighbours are:

- Kato’s descriptive population dynamics;
- BunDLe-Net’s behaviourally relevant latent manifold;
- NC-MCM’s discrete causal abstraction;
- the intervention literature that the project cannot emulate.

The result is exploratory and narrow. Its value is methodological: it shows how to ask whether a population model retains predictive structure beyond a strong common-factor control, and it shows how easily that question can be confused with a mechanism claim.

The literature does not rescue the biological interpretation. Only new data can.
