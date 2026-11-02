# Two bounded research proposals for a 6-ECTS MSc project

**Constraint:** 6 ECTS is approximately 150–180 hours of student work. Neither
proposal promises a validated generative digital twin, a causal whole-brain
model, or a new architecture search.

The two proposals answer different questions. **A** asks whether the one positive
result replicates. **B** asks what an intervention-aware study would require and
whether a minimal falsification test is feasible. They can be combined only by
reducing the depth of both; that is not recommended for 6 ECTS.

---

## Proposal A — Replicating the population-mean advantage

### Working title

**Population coupling beyond one test split: a leakage-safe replication of
low-dimensional neural dynamics in *C. elegans***

### Question

In a fixed, training-only low-dimensional representation, does a population
VAR(1) conditional mean predict held-out joint neural trajectories better than an
independent per-neuron AR(1), in a cohort whose animal independence is documented?

### Why this is the right next question

The current positive result is narrow but specific: H20 energy +0.446
[0.183, 0.660] and trajectory +0.375 [0.218, 0.522] for J5 versus J1 in a fixed
10-D PCA. It is exploratory because the test split was reused and the five
sessions are not independent animals. The decisive test is not another model. It
is the same prespecified contrast in an untouched cohort.

### Design

1. **Cohort:** one external cohort with documented animal/session independence.
   Preferred: a laboratory collaboration or a public release with animal IDs.
   Acceptable fallback: Kato `WT_Stim` first 360 s, only if the supervisor accepts
   same-lab sessions as a limited replication and the independence limitation is
   stated. The raw file must be re-downloaded and hash-verified.
2. **Representation:** fit the 10-D PCA on the new cohort's training frames only;
   freeze it before test evaluation. Do not reuse the old projection.
3. **Models:** J1 independent AR, J2 population diagonal, J5 population lead-1,
   J0 persistence. No new architecture.
4. **Primary endpoint:** H20 joint energy in the fixed 10-D PCA, paired by
   landmark, with animal-level resampling. One primary endpoint only.
5. **Secondary endpoints:** H20 trajectory energy, full-neuron CRPS, all-neuron
   energy, 50/80/95 coverage, covariance error. These are descriptive and
   multiplicity-labelled.
6. **Decision rule:** positive if the animal-level 95% interval for J5/J2 versus
   J1 excludes zero and the direction is consistent in at least 4/5 sessions or
   the equivalent cohort unit. Calibration is reported but is not a rescue
   criterion for the mean result.

### Work packages (168 hours)

| Work package | Hours | Deliverable |
| --- | ---: | --- |
| Data request, licence/provenance audit, animal manifest | 20 | data availability and independence memo |
| Adapter, QC, finite-trace and timestamp checks | 25 | tested loader + QC report |
| Training-only PCA and frozen representation | 15 | projection artifact + hash |
| J0/J1/J2/J5 evaluation | 30 | score tables, paired intervals |
| Animal-level bootstrap and session sensitivity | 20 | uncertainty report |
| Calibration and covariance diagnostics | 20 | calibration report, no rescue tuning |
| Reproduction package and tests | 20 | one-command rerun, manifest |
| Thesis/report and supervision | 18 | final chapter |
| **Total** | **168** | |

### Success and failure criteria

**Success:** a prespecified positive animal-level interval in the new cohort, with
the same direction for J2 and J5, and no calibration claim beyond what the data
show.

**Negative result:** the interval includes zero, or the direction reverses. This
is still a complete 6-ECTS result: it bounds the population advantage and rules
out a general claim. Do not add models to rescue it.

**Inconclusive:** fewer than three independent animals/sessions, undocumented
independence, or too few test frames. Stop and report the data limitation.

### Risks

| Risk | Mitigation |
| --- | --- |
| No independent cohort | Make the data request the first deliverable; do not substitute the reused split |
| Same-lab sessions are not independent | Label as limited replication; do not call it external validation |
| Cohort has different trace panels | Per-animal refit; no cross-animal numeric identity pooling |
| Calibration still fails | Report it; the primary endpoint is mean prediction, not calibration |
| Scope creep into architecture search | Freeze J0/J1/J2/J5 before evaluation |

### Explicitly out of scope

- A new encoder, foundation model or hyperparameter sweep.
- A causal or intervention claim.
- A target ranking.
- A whole-brain digital twin.
- Pooling neurons across animals without a validated identity mapping.

### What the supervisor must decide

- Is the proposed cohort acceptable as "independent" for an MSc?
- Is H20 joint energy the right primary endpoint, or is full-neuron MSE required?
- Is a negative replication an acceptable thesis outcome?
- Is Kato Stim an acceptable fallback, given undocumented independence?

---

## Proposal B — Intervention-aware simulation as a design problem

### Working title

**From observational trajectories to intervention-aware simulation: assumptions,
data audit and a minimal falsification test**

### Question

What would it take to make a model-based perturbation claim testable in
*C. elegans*, and can any current observational model's predicted response
ordering be falsified against existing matched stimulation data?

This is **not** "build a causal simulator in 150 hours". It is a design,
feasibility and falsification study.

### Why this is the right next question

The current perturbation engine edits a model state. It does not implement
optogenetics, a lesion, a dose, or a biological intervention. The repository has
already shown that:

- response reliability in the Randi atlas is low;
- the held-out-target operator misses its effect gate;
- corrected random-neuron controls make the example candidate non-specific;
- no matched neuronal intervention exists for the Kato model;
- observational models can fit the same trajectories and disagree about
  perturbation responses.

The next step is to specify what evidence would count, not to produce another
virtual effect.

### Design

1. **Define the estimand.** A target neuron \(i\), an intervention \(do(X_i=a)\)
   at time \(t\), a post-intervention window, and an outcome \(Y\) (downstream
   calcium response or a behavioural transition). State whether the claim is
   about an individual animal, a target type, or a population mean.
2. **State the assumptions.** Causal sufficiency, correct functional form,
   stationarity/transportability, support overlap, no unmeasured confounding,
   observation model, label validity, and intervention fidelity. For each,
   record whether the current data can test it.
3. **Audit matched data.** Randi 2023 (3,371 events, 110 animals, 259 targets) is
   the strongest available atlas, but it lacks behaviour labels and does not
   match the Kato model. Kato Stim oxygen switches are an environmental input,
   not a neuronal intervention. Baskoylu/Kramer are observational.
4. **Minimal falsification test.** Use held-out animals and held-out targets.
   Compare the model's predicted ordering of downstream responses with measured
   responses under stimulation, using sham/off-target events and a
   target-identity-only baseline. Pre-register the sign and rank correlation.
5. **Decision:** if the predicted ordering does not agree above a prespecified
   threshold, the model is not intervention-ready. If it does, that is a
   hypothesis for a prospective experiment, not a causal validation.

### Work packages (174 hours)

| Work package | Hours | Deliverable |
| --- | ---: | --- |
| Estimand, assumptions and claim-boundary protocol | 25 | preregistration-style protocol |
| Data audit and identity mapping | 30 | matched-data checklist, coverage table |
| Rebuild a minimal held-out-animal/-target benchmark | 35 | event table, splits, baselines |
| Model-response comparison and negative controls | 30 | rank/sign results, uncertainty |
| Power analysis and prospective design | 20 | sample-size and sham schedule |
| Reproduction package | 15 | one-command rerun, manifest |
| Thesis/report and supervision | 19 | final chapter |
| **Total** | **174** | |

### Success and failure criteria

**Success:** a documented, testable estimand and a completed falsification test
with an unambiguous negative or positive answer, plus a prospective design.

**Negative result:** the model's response ordering does not agree with measured
responses, or the data cannot support the test. This is a useful negative result
and a concrete data request.

**Failure mode to avoid:** calling a model-state edit a biological effect, or
using the absence of matched data to justify an unfalsifiable simulation.

### Data checklist for a real intervention study

- [ ] Animal IDs and cohort manifest, not filenames alone.
- [ ] Target neuron identity with an annotation version and laterality rule.
- [ ] Intervention modality, power/dose calibration and timing.
- [ ] Randomized or counterbalanced stimulation versus sham/off-target.
- [ ] Pre-stimulus baseline and post-stimulus window prespecified.
- [ ] Behavioural readout independent of the stimulated neurons.
- [ ] At least two animals per target for a target-level claim; more for a
      population effect.
- [ ] Negative controls and a prespecified analysis plan.
- [ ] Raw data hash and licence.

### Risks

| Risk | Mitigation |
| --- | --- |
| No matched data | Deliver a data request and power analysis; do not simulate a causal effect |
| Identity mapping is ambiguous | Restrict to unambiguous labels; report exclusions |
| Low response reliability | Use animal-level resampling; do not rank targets |
| Behaviour labels are circular | Use an independent behavioural readout or drop the behavioural claim |
| Scope too large | One target family, one outcome, one falsification test |

### Explicitly out of scope

- A full causal digital twin.
- A whole-brain intervention optimizer.
- A new optogenetics experiment without lab access, ethics and safety review.
- A claim that a virtual perturbation predicts a real effect size.
- A neuron-specific stimulation recommendation.

### What the supervisor must decide

- Is the target estimand a downstream calcium response or a behavioural outcome?
- Is there a matched intervention dataset, or should the project start with a
  data request?
- Is a negative falsification result acceptable?
- Would a prospective collaboration provide randomized sham-controlled data?

---

## Choosing between A and B

| Criterion | A | B |
| --- | --- | --- |
| Depends on new independent data | High | High |
| Can produce a clean result with existing data | Medium (Kato Stim fallback) | Medium (Randi reanalysis) |
| Directly addresses the original intervention interest | No | Yes |
| Risk of overclaiming | Low if scoped | High if scoped poorly |
| Best fit for a methods-focused MSc | Yes | Partly |
| Best fit for a neuroscience/experiment-focused MSc | Partly | Yes |

**Recommendation:** choose A if the goal is a defensible replication of the one
positive result. Choose B if the goal is to make the intervention question
scientifically honest and to design the next experiment. Do not combine them into
a single 6-ECTS claim that the current data cannot support.

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**
>
> Biological validation of either proposal is `NOT_EVALUABLE` until matched
> intervention data and an independent cohort exist.
