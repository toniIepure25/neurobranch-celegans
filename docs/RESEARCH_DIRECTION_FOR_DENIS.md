# Research direction for Denis Kovačević

> **Proposal, not a confirmed requirement.** The canonical decision surface is the
> **Scope & decisions** page and [`DENIS_REQUIREMENTS_BRIEF.md`](DENIS_REQUIREMENTS_BRIEF.md).
> Public sources are audited in [`DENIS_PUBLIC_SOURCE_AUDIT.md`](DENIS_PUBLIC_SOURCE_AUDIT.md).

**Prepared:** 2026-10-31. **Branch:** `research/external-validation-readiness`.
This is a proposal for a bounded 6-ECTS project, not a request to launch another
architecture search.

## 1. The original question

Can a free-running neural dynamics model predict an individual *C. elegans*
animal's future population calcium activity from observed history, without
teacher forcing, and can it do so better than a constant-state persistence
baseline on animals that were not used to select the model?

Two follow-up questions must remain separate:

- Does a model fitted on one animal transfer to another animal unchanged?
- Does the model respond correctly to a measured environmental input?

Neither follow-up is answered by the current Phase 6 evidence.

## 2. What the current system can do

- Load Kato NoStim recordings, preserve trace-level labels and build
  leakage-safe temporal/blocked splits.
- Fit VAR, Gaussian stochastic VAR and switching VAR models; run free-running
  rollouts; evaluate H=20/H=100 skill, distributional metrics, branching and
  support-gated virtual perturbations.
- Produce model cards, manifests and a Streamlit/Plotly interface.
- Read Kato Stim and Nichols HDF5 metadata with trace-level labels, time vectors,
  behavioural states and oxygen schedules.

It cannot:

- use an exogenous input channel;
- map neurons one-to-one across animals;
- establish animal independence from OSF filenames;
- measure behaviour independently of the calcium-derived labels;
- support a causal or digital-twin claim.

## 3. Negative findings that define the project

1. No learned family had positive median validation skill in Phase 6. The
   selected VAR was the least-negative family.
2. The exploratory H=20 test skill of the Phase 6 VAR (0.093) was not better than
   the frozen Phase 5 VAR (0.101); the confidence interval was small and
   positive but the baseline-superiority gate failed.
3. Distributional fidelity failed: 0/5 worms improved energy, autocorrelation
   and covariance simultaneously.
4. Stochastic branching failed: TV 0.513 versus Markov 0.264.
5. G6 and G7 were not evaluable; there is no robust multi-family named-neuron
   effect and no biological intervention.
6. The test block was reused; all Phase 6 test/gate claims are exploratory.
7. The Kato Stim external pilot is also negative: 0/7 animals beat persistence
   at H=20 (median skill −0.602; mean −0.593; bootstrap 95% CI −1.024 to
   −0.241). This is a valid negative replication, not a failure to hide.

The honest conclusion is that the current simulator is a measurement-bounded
research prototype, not a validated digital twin.

## 4. Why external validation is the right next step

Continuing to tune architectures on the same five Kato worms would increase
apparent performance without increasing evidence. The binding constraint is not
model capacity; it is independent data, a shared identity space and an
input-aware design.

The current branch therefore asks a narrower, defensible question:

> Can the frozen Phase 6 pipeline predict an untouched future segment of a new
> Kato animal under constant oxygen, after refitting only on that animal's past?

This is protocol B. It avoids the reused NoStim test block, requires no
cross-animal identity mapping, and can be executed now. It does not test oxygen
response or biological intervention.

## 5. Data requirements

For a credible B test:

- independent animals/sessions, with independence documented rather than
  inferred from filenames;
- continuous population calcium traces with time vectors;
- enough frames for chronological train/validation/test and a purge gap;
- stable oxygen and preparation during the evaluated segment;
- finite traces or a pre-specified finite subset;
- the target segment untouched until evaluation.

For C, additionally:

- a one-to-one neuron/ROI mapping with a declared annotation version;
- a pre-specified rule for missing neurons and multi-neuron traces;
- leave-one-animal-out training and untouched held-out animals;
- no pooling of worm-specific numeric IDs.

For an input-response question, additionally:

- a controlled, randomized sensory input with a recorded time series;
- an exogenous-input model, specified before looking at the target test data;
- a behavioural readout independent of the calcium labels;
- a causal design if the claim is about a specific neuron.

## 6. Feasibility of the available datasets

| Dataset | What is available | Verdict |
| --- | --- | --- |
| Kato Stim | 7 sessions, 720 s, constant 21% first 360 s, 107–151 traces, 4 states | **GO for a limited B pilot**; executed; no oxygen-response claim |
| Kato NoStim | 5 sessions, 1080 s, already used | no untouched test; not an external cohort |
| Nichols 2017 | 44 sessions, ~1080 s, lethargus, 10/21% oxygen, quiescence | **LIMITED**; different stage/preparation; only AVAL/RIS common; needs separate protocol |
| dLDS outputs | derived latent fits, not raw independent animals | not a validation dataset for NeuroBranch |

Kato Stim is not a fresh independent cohort in the strong sense: independence is
not documented, and it is the same laboratory and preparation family. It is,
however, a set of sessions not used in this repository's Phase 5/6 selection, and
the first 360 s are constant-oxygen. That supports a limited B replication, not a
confirmatory external validation.

## 7. Proposed 6-ECTS scope (168–180 hours)

| Work package | Hours | Deliverable |
| --- | ---: | --- |
| Freeze/audit Phase 6; verify manifests and support deviation | 20 | audit JSON + `PHASE6_FINAL_SCIENTIFIC_AUDIT.md` |
| External adapter and data QC | 30 | loader, tests, inventory |
| Kato Stim B pilot and uncertainty analysis | 25 | per-animal/per-start reports |
| Protocol, compatibility and claim-boundary writing | 25 | protocol + compatibility docs |
| Reproducibility: tests, Ruff, MkDocs, manifest | 25 | green checks, committed reports |
| Thesis/report writing and supervisor iteration | 43 | final report/chapter |
| **Total** | **168** | |

Explicitly out of scope for 6 ECTS:

- refitting all 44 Nichols sessions;
- developing a neuron-alignment model for C;
- adding an exogenous-input channel and running a sensory-response study;
- a new architecture search;
- causal intervention claims.

If the supervisor wants Nichols or input modelling, the project needs a
different scope or a follow-on thesis.

## 8. Questions for Denis Kovačević

1. **Independence:** Is a same-lab Kato Stim cohort acceptable as “additional
   animals” when OSF provides no animal IDs and non-overlap with NoStim cannot be
   proven? If not, what evidence would satisfy the independence requirement?
2. **Constant-oxygen segment:** Is the first 360 s of Kato Stim (21% O2, no
   switches) an acceptable autonomous-dynamics cohort, or should any oxygen
   change disqualify the session from an autonomy claim?
3. **Protocol B scope:** Is a per-animal refit with frozen Phase 6
   hyperparameters the right 6-ECTS test, or should the project prioritise a
   shared cross-animal model (C) despite the alignment problem?
4. **Neuron identity:** Which annotation/alignment standard should be used for
   C? Is a named-neuron intersection acceptable, or is a latent alignment model
   required before any cross-animal claim?
5. **Nichols:** Should Nichols be excluded because of the developmental stage,
   genotype mixture and only two common named labels, or retained as a separate
   quiescence/arousal case study?
6. **Input question:** Is an oxygen-response analysis in scope? If yes, do you
   accept that it requires a new input-aware model and a randomized design, and
   that it is not a neuronal intervention?
7. **Claim boundary:** Is a negative external validation an acceptable thesis
   result, or does the project need a positive predictive result to be
   defensible?
8. **Assessment:** What would you consider a sufficient 6-ECTS contribution:
   the audit + adapter + seven-animal pilot + docs, or a narrower piece with more
   depth?

## 9. Deliverables and decision points

Deliverables:

- frozen Phase 6 audit with the 95/99 deviation and manifest caveats;
- external compatibility matrix and protocol;
- tested Kato Stim/Nichols adapter;
- seven-animal Kato Stim pilot with per-animal and per-start results;
- final report with explicit negative findings and claim boundaries.

Decision points:

- **Week 1:** approve independence and constant-oxygen assumptions.
- **Week 3:** approve B-only scope; decide whether Nichols is excluded.
- **Week 5:** review pilot result; do not add models to rescue it.
- **Week 7:** freeze claims and write.
- **Week 9:** reproducibility check and submission.

## 10. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Pilot is negative | Report it as a replication failure; do not tune on test |
| Seven sessions too few | Label exploratory; do not claim confirmation |
| Animal independence unproven | State it explicitly; seek cohort metadata |
| Oxygen confound | Hold O2 constant; never call it neuronal stimulation |
| Cross-animal alignment impossible | Keep C blocked; do not pool numeric IDs |
| Scope creep into architecture search | Freeze model family; supervisor sign-off |
| 44 Nichols sessions exceed budget | Exclude or treat as separate follow-on |

## 11. Bottom line

The scientifically defensible 6-ECTS project is an **external-validation
readiness and negative-replication study**, not another decoder or architecture
search. It freezes Phase 6, audits the support-threshold and spectral-cap
limitations, builds a tested adapter, runs one bounded constant-oxygen Kato Stim
B pilot, and states clearly what the data cannot support. That is a coherent
thesis contribution even if every predictive result is negative.
