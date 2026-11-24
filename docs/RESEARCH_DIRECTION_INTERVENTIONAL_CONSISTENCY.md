# Research direction: interventional consistency of neural abstractions

**Status:** research direction, not a protocol and not a result. Written
2026-11-24 on branch `research/interventional-consistency`, based on `547e7b1`.
**Working name:** NeuroBranch-IC.

> **Evidence boundary.** Everything in §1 about P0–P8.5 is the project's *internal*
> record. The public repository is far behind it, so those results are used here to
> design the next stage and are not presented as independently confirmed findings.
> Every external identifier in this document resolved at Crossref *and* its title
> matched the expected fragment on 2026-11-24; the ledger is
> [`../research/interventional_consistency_evidence.csv`](../research/interventional_consistency_evidence.csv).

---

## 1. Where the project actually is

The most valuable output of the local programme is not a score. It is a sequence of
self-corrections, each of which removed a way of being wrong:

| Correction | What it removed |
|---|---|
| Observational accuracy did not predict intervention fidelity (P2/P3) | "good prediction implies good abstraction" |
| A P3 advantage came from operator semantics, not from learning the operator | credit assigned to the wrong component |
| A P3 calcium result was a different estimand | silent estimand substitution |
| Access to `H` in P4 was structural privileged information | "learned representation" claims built on handed-over structure |
| The claimed MOD non-identifiability did not survive the history analysis (P4.5) | an impossibility claim that was really a memoryless-model artefact |
| A P6 cross-animal advantage was a target mismatch | cross-animal gains that were label artefacts |
| ~78.6% of the legacy PCA–NC-MCM difference vanished once the decoder matched (P7) | leaderboard differences that were decoder differences |

Read as a whole, these are one finding: **the field's usual evidence does not
separate a representation from the decoder, the measurement model, the privileged
information, and the evaluation protocol.** That is a methods problem, and it is
exactly the problem this direction attacks.

## 2. The question

> Under which conditions does a coarse-grained representation learned from partial
> neural observations preserve the dynamics and the effects of interventions of the
> underlying system — and when is that impossible from the information the protocol
> permits?

Not "which embedding wins". The object of study is the *commutation* between the
micro-level intervention diagram and its macro-level image.

## 3. Why not another latent model

The representation space is crowded, and the crowding is verifiable:

| Work | Status | What it already occupies |
|---|---|---|
| NC-MCM ([10.1371/journal.pcbi.1012674](https://doi.org/10.1371/journal.pcbi.1012674)) | peer-reviewed | behavioural and dynamic causal consistency, learned from observation |
| BunDLe-Net v5 ([10.1101/2023.08.08.551978](https://doi.org/10.1101/2023.08.08.551978)) | preprint | minimal manifold retaining only behaviourally relevant dynamics |
| PSID ([10.1038/s41593-020-00733-0](https://doi.org/10.1038/s41593-020-00733-0)) | peer-reviewed | explicit behaviourally relevant vs irrelevant subspaces |
| CEBRA ([10.1038/s41586-023-06031-6](https://doi.org/10.1038/s41586-023-06031-6)) | peer-reviewed | consistent embeddings across sessions and animals |
| MARBLE ([10.1038/s41592-024-02582-2](https://doi.org/10.1038/s41592-024-02582-2)) | peer-reviewed | geometry-based comparison of dynamics across animals and systems |
| dLDS in *C. elegans* ([10.1038/s42003-025-08599-3](https://doi.org/10.1038/s42003-025-08599-3)) | peer-reviewed | time-varying operator decomposition on whole-brain worm data |
| NEBULA ([10.64898/2026.06.05.730482](https://doi.org/10.64898/2026.06.05.730482)) | preprint | joint latent model with in-silico steering |

So: predicting behaviour, aligning animals, learning local dynamics and steered
virtual interventions are all taken. What none of them supplies is a **calibrated
measure of whether the abstraction commutes with interventions, with an explicit
statement of the information each method was given**.

Two identifiers named in the project brief — exact and approximate causal
abstraction (Rubenstein et al.; Beckers, Eberhardt & Halpern) — were **not resolved
in this session**. They are recorded as unresolved in the search log and are
deliberately not cited here. That prior art certainly exists and must be resolved
before any novelty sentence is written; until then this document claims only the
gap above, not priority over the causal-abstraction literature.

## 4. The opening NC-MCM itself leaves

NC-MCM's own discussion states the limit precisely: concluding dynamic causal
consistency from observational data "is based on the absence of sufficient evidence
to reject the null hypothesis of Markovianity", and "absence of evidence is not the
same as evidence of absence", so such a model "should be regarded as expressing
experimentally testable predictions ... rather than as direct proof thereof".

The same discussion names the remedy: reconstructing the state with Taken's
time-delay embedding, which they describe as a direction for "more advanced
algorithms" rather than something the paper does. That is the opening. The
programme's question is the operational version of their caveat:

- does finite observation history restore interventional consistency, and at what
  history length;
- when does no permitted history suffice because the intervention-relevant
  distinction is observationally non-identifiable;
- and can a benchmark detect the difference without being fooled by the decoder,
  the measurement model or privileged information.

## 5. Hypotheses

Primary: **A + B + C**. **D** supplies the metric. **E** is explicitly out of scope.

- **A — history-aware sufficiency.** An abstraction built on `C_t = τ(H_t^{(L)})` can
  recover dynamic and interventional consistency where no memoryless `τ(Y_t)` can.
  Not new in general (predictive-state representations), so the deliverable is the
  *characterisation*: for which observation dynamics and mixing does a finite `L*`
  exist, and when does none exist.
- **B — observational ≠ interventional sufficiency, even when the macro process
  looks Markovian.** The deliverable is a minimal counterexample family: two
  abstractions with the same behaviour prediction, the same next-state accuracy and
  the same Markov diagnostic, but radically different intervention error.
- **C — information-contract fairness changes the ranking.** P7 is the local signal;
  the deliverable is a machine-readable contract and a rule that refuses
  "representation quality" comparisons across different contracts.
- **D — commutation error predicts held-out interventional generalisation better
  than reconstruction or prediction error.** Used as the evaluation metric and as a
  testable claim in its own right.
- **E — cross-animal equivalence up to an observation model.** Interesting, high
  risk, dependent on data the project does not have. Not a primary result.

## 6. Formal sketch

Micro state `X_t`, interventions `u ∈ I` with kernels `P_u(X_{t+1} | X_t)`,
observation `Y_t ~ O(· | X_{0:t})`, behaviour readout `B_t ~ Q(· | X_t)`, history
`H_t^{(L)} = (Y_{t-L+1}, …, Y_t)`, macrostate `C_t = τ(H_t^{(L)})`.

Two quantities must commute: the *dynamics* and the *effect of the intervention*.
The operational errors are, for a macro cell `c` and operator `u`,

```
ε_B(c,u) = sup over h,h' with τ(h)=τ(h')=c of  D[ P(B_{t+1} | do(u), H_t=h),
                                                  P(B_{t+1} | do(u), H_t=h') ]
ε_D(c,u) = the same supremum for  P(τ(H_{t+1}) | do(u), H_t=·)
Δ_commute = D[ τ# P_X^u(X_{t+1} | X_t),  P_C^{ω(u)}(C_{t+1} | C_t) ]
```

`D` is total variation for discrete states and a declared Wasserstein distance for
continuous ones. These are working definitions, not theorems. The point of the
worst-case (supremum) form is that a mean hides exactly the aliasing that matters.

## 7. Gap matrix

| Gap | Already exists | What this direction adds |
|---|---|---|
| Formal causal abstraction | exact/approximate abstraction literature, NC-MCM BCC/DCC | operational micro–macro commutation under partial observability |
| Markov sufficiency | NC-MCM uses Markovianity with explicit caveats | separation of observational from interventional sufficiency |
| Behaviourally relevant dynamics | BunDLe, PSID, CEBRA | conditions under which behaviour supervision preserves or destroys interventional structure |
| Virtual interventions | NEBULA and other generative models | quantifying when a virtual intervention is not identifiable at all |
| Benchmark fairness | objective-dependent rankings are known | an information contract that separates representation, decoder, missingness and privileged information |
| Biological validation | Randi optogenetic atlas | a pre-specified prediction to test *if* Stage One passes |

## 8. What "revolutionary" would have to mean here

Three thresholds, in decreasing feasibility. Only one is a realistic MSc target.

1. **Theoretical.** A useful sufficient condition or an impossibility witness linking
   observability/history to BCC/DCC for a non-trivial system class, compared formally
   against existing causal abstraction, not just restated.
2. **Methodological (the target).** Show across several systems and several methods
   that standard evaluations can certify a representation as "good" while micro–macro
   interventions do not commute, then provide a protocol that detects those false
   positives and survives decoder and information confounds.
3. **Empirical.** A learned abstraction, never shown certain interventions, predicts
   the effect distributions of real perturbations on independent animals where
   observational comparators fail. This needs data the project does not have.

A well-formed negative result — "these observational criteria are insufficient;
here are the counterexamples, the conditions, and the protocol that detects the
failure" — clears threshold 2 and is a better contribution than another leaderboard.

## 9. Non-goals

No new architecture family and no hyperparameter search. No "digital twin" language.
No SaaS. No generic cross-animal embedding. No recycling of the P7 power calculation
for a different estimand. No biological claim from synthetic ground truth.

## 10. Decisions for Denis

1. Should the first milestone be the formal/interventional benchmark, or should the
   project be anchored from the start to a specific dataset or lab question?
2. Which intervention family defines the abstraction: instantaneous state
   interventions, controlled inputs, structural/connectivity interventions, or the
   perturbations the group actually performs?
3. Are history-dependent macros `C_t = τ(Y_{t-L:t})` compatible with the intended
   NC-MCM interpretation, or is the abstraction conceptually restricted to an
   instantaneous neuronal state?
4. What would count as sufficient evidence of usefulness: approximate commutation on
   held-out interventions, agreement with real perturbation data, a formal
   guarantee, or a combination?

## 11. Stage One, and what it can and cannot establish

Stage One ([`INTERVENTIONAL_CONSISTENCY_STAGE1.md`](INTERVENTIONAL_CONSISTENCY_STAGE1.md),
protocol in [`../research/interventional_consistency_protocol.md`](../research/interventional_consistency_protocol.md))
builds the instrument on synthetic ground truth. It can establish that the benchmark
separates consistent from inconsistent abstractions, that the reference oracle is
independent of the simulator, and that contract violations are detected. It cannot
establish anything about *C. elegans*. Any biological statement requires independent
experimental evidence, and synthetic causal ground truth is evidence about the
benchmark, not about the worm.
