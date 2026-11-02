# NeuroBranch: Evaluating Generative Neural Population Dynamics and Model-Dependent Perturbations in *C. elegans*

**Supervisor presentation** · target length 9 minutes 30 seconds, plus questions.  
**Audience:** Denis Kovačević and MSc committee.  
**Mode:** focused demo is optional; the argument must stand without it.  
**Rule:** do not call the model a digital twin, a causal simulator, or a validated
perturbation system.

## Slide 1 — Title and disclosure (0:30)

**On slide**

> NeuroBranch: Evaluating Generative Neural Population Dynamics and
> Model-Dependent Perturbations in *C. elegans*
>
> Internal MSc handoff · exploratory reanalysis · November 2026

**Speaker notes**

This is a handoff of what the repository can and cannot support. The original
brief from our conversation is reconstructed, not confirmed. I will separate the
one result I think is worth following from the many claims the data do not
support.

## Slide 2 — The question, and what is provisional (0:55)

**On slide**

- Can a model start from a measured population state and roll forward without
  future observations?
- Do alternative futures depend on a modelled perturbation of a named neuron?
- Would that generate experimentally testable stimulation hypotheses?

**Provisional:** preparation, target circuit, intervention modality, behavioural
endpoint and deliverables were not specified in writing.

**Speaker notes**

The motivating picture was low-dimensional trajectories that appear to branch,
and a virtual perturbation that changes the probability of one future. That is a
good scientific question, but it is not yet an operational specification. The
project deliberately treats the exact assignment as open.

## Slide 3 — What was actually tested (1:00)

**On slide**

| Workstream | What it established |
| --- | --- |
| BunDLe-Net / NC-MCM audit | distributed embedding is transductive; NC-MCM test ported, training not reproduced |
| Kato baselines | persistence is strong; nonlinear free-running models do not robustly beat it |
| Randi intervention atlas | 3,371 events, 110 animals; low reliability; held-out-target effect misses its gate |
| Branching / turn direction | entropy deficit, not excess; no neural increment beyond behaviour |
| Simulator | software contract passes; dynamical and perturbation-specificity gates fail |
| Calibration / joint dynamics | overdispersion corrected; mean signal survives; calibration fails |

**Speaker notes**

This is six workstreams and several negative results. The important point is that
they are connected: each one removes a possible explanation for the next.

## Slide 4 — The evidence ladder (0:55)

**On slide**

1. Representation: leaked artifact geometry is not generalization.
2. Prediction: persistence is hard to beat in full-neuron space.
3. Branching: observed outcomes are concentrated, not broadly branching.
4. Intervention: response reliability is low and target specificity is absent.
5. Generation: free-running rollouts fail distributional fidelity.
6. Calibration: correcting the innovation pool fixes overdispersion, not calibration.

**Speaker notes**

The project did not fail because one model was badly tuned. It failed repeatedly
at different levels. That pattern is why I stopped adding architectures and
started treating the problem as measurement- and design-limited.

## Slide 5 — The central result (1:30)

**On slide**

Fixed 10-D PCA, training frames only. 175 H20 landmarks. 100 members.

| Contrast | Improvement | 95% interval |
| --- | ---: | --- |
| J5 population vs J1 independent AR, energy | **+0.446** | **[0.183, 0.660]** |
| J5 vs J1, trajectory | **+0.375** | **[0.218, 0.522]** |
| J5 vs persistence, energy | +0.250 | [0.020, 0.427] |
| J5 vs J1, full-neuron CRPS | +0.005 | [−0.007, 0.016] |

J2 (parametric diagonal) is effectively identical: energy 2.700, trajectory 2.058.

**Speaker notes**

This is the result I would defend, with qualifications. The population
conditional mean carries joint information that independent AR misses. It is not
a marginal per-neuron gain. It is not a covariance gain. It is a joint
trajectory gain in one fixed representation. J2 reproduces it, so it is not an
artefact of one bootstrap.

## Slide 6 — Mean, not covariance (0:55)

**On slide**

- Mean main effect: **+0.221** energy, **+0.230** trajectory.
- Full-covariance main effect: **−0.086** energy, **−0.071** trajectory.
- Full covariance coverage: 0.767 / 0.957 / 0.992; covariance error 0.762.
- Interpretation: the identifiable signal is the conditional mean; the
  103-dimensional innovation covariance is not stable enough to help.

**Speaker notes**

This is the most useful scientific decomposition. It tells the next study what
to replicate: the mean operator, not a more elaborate noise model. It also
explains why adding covariance made things worse rather than better.

## Slide 7 — What fails (1:10)

**On slide**

- 50% coverage: **0.588** full-neuron, **0.546** in 10-D PCA.
- Neural covariance error: **0.604** against a 0.50 gate.
- Student-t correction: energy **2.829**, worse than 2.700.
- Parameter ensemble: energy **2.807**, worse.
- Session 1 negative; session 2 ambiguous.
- Block-10 interval for J5 vs persistence includes zero.
- Test split reused; five sessions are not five animals.

**Stop label:** `PROBABILISTIC CALIBRATION NOT ESTABLISHED`.

**Speaker notes**

The model over-covers at 50%, and its covariance is wrong. That is not a cosmetic
problem: it means the ensemble cannot be read as a calibrated distribution of
alternative futures. The test reuse and session structure make even the positive
result exploratory.

## Slide 8 — Perturbation: model edit versus biology (1:00)

**On slide**

- The intervention engine edits a **model state**: additive, clamp, zero, scale.
- It is a valid mathematical counterfactual inside the fitted model.
- It is not optogenetics, a lesion, a dose, or a prediction of a wet-lab effect.
- Corrected random-neuron controls made the example candidate non-specific
  (empirical one-sided p ≈ 0.21).
- No matched neuronal intervention exists for this model.

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

Biological validation: `NOT_EVALUABLE`.

**Speaker notes**

I want to be very explicit here. A virtual perturbation can be useful for
generating hypotheses, but it cannot validate them. The model's causal graph is
assumed, not identified. Observational data alone cannot tell us what would
happen under a real stimulation.

## Slide 9 — Decision and two next studies (1:10)

**On slide**

**Do not** run another architecture search on the reused test split.

**A. Replicate the population mean.** Independent cohort; fixed representation;
H20 joint energy; J1 and persistence baselines; animal-level resampling.
150–180 h.

**B. Intervention-aware design.** Specify target, timing, sham and outcome;
audit matched data; run a minimal held-out-animal/-target falsification test.
Deliver a protocol, not a causal simulator.

**Speaker notes**

The choice depends on what you want the MSc to establish. A is a clean
replication study with a prespecified endpoint. B is a design and feasibility
study that could fail early for lack of data, but it asks the intervention
question honestly. I would not promise a full causal simulator in 150 hours.

## Slide 10 — Questions for discussion (0:30)

**On slide**

- Which target is primary: trajectory simulation or a behavioural event?
- What counts as "branching" for your purpose?
- Is a model-dependent virtual perturbation useful even when not causally valid?
- Is there an independent cohort or matched stimulation dataset?
- What would make you say stop?

**Speaker notes**

These are genuine questions, not reconstructed requirements. The answers change
the scope, the data request and the claim boundary.

## Timing summary

| Section | Time |
| --- | ---: |
| Title, question, scope | 1:25 |
| What was tested and the evidence ladder | 1:55 |
| Central result and decomposition | 2:25 |
| Failures and perturbation boundary | 2:10 |
| Decision, next studies, questions | 1:40 |
| **Total** | **9:35** |

For a strict 8-minute slot, drop slide 6 and fold its one-sentence conclusion
into slide 5.

## Demo cues

- After slide 5: show "Central result" and "Real trajectory" in the focused demo.
- After slide 7: show "Calibration" and "Perturbation boundary".
- Do not demo a perturbation trajectory as if it were a biological prediction.
- If the demo fails, the slides and `HANDOFF_DEMO_GUIDE.md` contain every number.
