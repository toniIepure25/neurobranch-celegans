# Original research objective: neural dynamics, branching and named-neuron perturbations

**Status:** research-direction reset, written 2026-10-31.
**Branch:** `research/core-dynamics-simulator`
**Starting commit:** `e18139b15048b31bc384cab1cb4591b09b80ab34`

This document separates three things that earlier phases progressively mixed:

1. the research interest reported by the student after a meeting with Denis
   Kovačević (Neuroinformatics Research Group, University of Vienna);
2. the scientific questions that follow from that interest;
3. the modelling proposals made by this repository.

Nothing in section 1 is an official specification from Denis Kovačević, Prof.
Moritz Grosse-Wentrup, or the University of Vienna. The conversation was not
recorded as a protocol. Items that require confirmation are in
[`QUESTIONS_FOR_DENIS.md`](QUESTIONS_FOR_DENIS.md).

---

## 1. What was reported, and what is provisional

The student's reconstruction is:

- *C. elegans* has approximately 302 neurons in the adult hermaphrodite.
- The Vienna group works on BunDLe-Net and low-dimensional neural population
  representations.
- Neural trajectories visualised in three dimensions can appear to branch:
  activity associated with one behavioural state may evolve toward different
  future states.
- The motivating example is a sustained reversal followed by different motor
  outcomes, such as dorsal versus ventral turning.
- The student wanted a machine-learning neural simulator that could ask what
  happens to the future neural trajectory if a particular neuron is modified,
  and whether an intervention changes the probability of one future trajectory
  relative to another.

**Provisional.** The exact biological model, intervention modality, target
circuit, behavioural endpoint, data access and expected deliverables were not
specified in writing. They must not be invented.

---

## 2. The scientific problem

The object of study is a stochastic population process observed through calcium
imaging. For worm \(i\), let \(x_t \in \mathbb{R}^{N_i}\) be the measured
population state at frame \(t\), \(b_t\) a behavioural label, and \(u_t\) an
explicit model input. The simulator is a conditional generative model

\[
\hat{x}_{t+1} = F_\theta(\hat{x}_{t-h+1:t}, u_t) + \varepsilon_t,
\qquad \varepsilon_t \sim Q_\theta(\cdot \mid \hat{x}_{t-h+1:t}),
\]

with a behaviour readout \(g_\psi(\hat{x}_t)\). The observation model is
identity on standardised \(\Delta F/F\); this is not a model of membrane
potential, spiking, muscles, or the full nervous system.

The questions that matter are:

1. Can the model be initialised from a real measured state and generate a
   complete future trajectory without reading future observations?
2. Does the generated trajectory reproduce time-series properties of real
   populations, beyond one-step teacher-forced prediction?
3. Does the model reproduce the observed distribution of future behavioural
   states after sustained reversal, including alternative outcomes?
4. Can a named neuron be perturbed in a reproducible, explicitly defined way?
5. Does the perturbation change the simulated future trajectory and the
   simulated distribution of outcomes?
6. Are those changes stable across initial states, model seeds, model classes
   and matched control neurons?
7. What evidence would be required before calling an effect biological rather
   than model-dependent?

Question 3 is not the same as “classify the turn direction.” Question 5 is not
the same as “increase AUROC.” The primary object is the **trajectory-generating
model and its response to intervention**, not a decoder leaderboard.

---

## 3. Why the project drifted

Earlier phases produced useful results:

- animal-disjoint decoding benchmarks;
- BunDLe-Net and NC-MCM reproduction audits;
- behavioural prediction and turn-direction analyses;
- branching null tests;
- virtual-perturbation scans;
- reliability and measurement audits.

They also shifted the implicit success criterion toward reversal onset,
dorsal/ventral classification, AUROC, and predictive-neuron ranking. Those are
secondary tasks. The current reset restores the original target:

> learn and simulate neural population dynamics under hypothetical
> perturbations, and state clearly which parts are observation, simulation and
> experimental validation.

---

## 4. What is already known from this repository

These are constraints, not obstacles to hide:

- Five wild-type, immobilised Kato recordings are the primary dynamics data.
- Each recording has 109–135 segmented traces; only 31–47 are named. The
  behaviour-label neurons are excluded from predictors.
- Behaviour labels were inferred from AVA/SMD/RIB calcium, not independently
  measured movement. Behavioural validation is therefore label-conditional.
- Phase 2 found no specific RIVL perturbation effect: random control neurons
  produced effects of similar or larger magnitude, and the result was
  model-dependent.
- The corrected branching audit found local outcome entropy *lower* than a
  geometry-preserving null in most worms, not excess branching. No worm showed
  significant excess entropy. A visual split in a 3-D projection is not a
  bifurcation.
- Phase 4 and Phase 5 found no robust incremental neural signal for turn
  direction or the SAAV mechanism.
- Phase 15 corrected the reliability headline: `0.560` is an extrapolated
  historical point estimate, not a bound. The observed-count estimate is
  `R_obs = 0.2606`.

The simulator must be evaluated against these facts, not tuned until a
prettier story appears.

---

## 5. What this project will not claim

- That a simulation is a living-worm intervention.
- That a learned model identifies a causal effect from observational data.
- That a virtual perturbation is optogenetic stimulation, a lesion, or a
  pharmacological manipulation.
- That a 3-D visual branch is a mathematical bifurcation.
- That the observed subset is the whole brain.
- That a model fitted to Kato transfers to another preparation without a
  transport test.
- That a high AUROC is evidence of a working simulator.

The standing label for every unvalidated perturbation is:

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

---

## 6. Primary deliverable

A simulator-centred research package that can:

1. load a real recording and select a held-out initial state;
2. generate free-running, stochastic future population trajectories;
3. compare those trajectories with observed dynamics on pre-specified metrics;
4. project real and simulated activity into a common latent representation;
5. estimate future behavioural-state distributions and compare them with
   observed local-outcome distributions and nulls;
6. apply an explicitly defined perturbation to a named neuron;
7. compare paired baseline and perturbed rollouts, including matched controls;
8. report uncertainty, support/OOD status and model disagreement;
9. export the trajectories and experiment manifest for reproduction;
10. state which conclusions are observational, simulated, or experimentally
    validated.

The application is a window into this model, not a dashboard of decoder
accuracy.
