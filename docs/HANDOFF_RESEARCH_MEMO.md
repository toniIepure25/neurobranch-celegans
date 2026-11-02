# NeuroBranch: Evaluating Generative Neural Population Dynamics and Model-Dependent Perturbations in *C. elegans*

**Two-page research memo for an MSc discussion**  
Prepared for Denis Kovačević, Neuroinformatics Research Group, University of Vienna  
Internal scientific handoff · November 2026 · exploratory reanalysis, not a systematic review

## 1. The question, and what is provisional

The motivating interest was whether a generative model of *C. elegans* whole-brain
calcium activity can start from a measured population state, roll forward without
future observations, and show how alternative futures depend on a modelled
perturbation of an identified neuron. A second interest was whether such a model
could support experimentally testable stimulation hypotheses.

That framing is a **provisional reconstruction**, not a confirmed brief. The
preparation, target circuit, intervention modality, behavioural endpoint and
deliverables were never specified in writing. This memo separates what the
repository tested from what a next study would need.

The data are observational: five repeated Kato et al. (2015) `WT_NoStim`
sessions, each about 18 minutes at ~2.9 Hz, with 103–135 calcium traces and eight
inferred behavioural states. The labels are decoded from a subset of the same
neurons. Five sessions are not five documented animals. No matched neuronal
intervention exists in this dataset.

## 2. What was built and tested

1. **Representation audit.** The distributed BunDLe-Net embeddings are
   transductive: they were fitted on all frames, including later test frames.
   A causal-filter reimplementation on one session reached kNN accuracy 0.407 and
   silhouette −0.064, versus 0.949–0.984 and 0.447–0.565 for the leaked
   artifacts. This is not a BunDLe-Net reproduction. The NC-MCM Markov test was
   ported and agreed with distributed p-values within 0.023; NC-MCM training was
   not reproduced.
2. **Baseline dynamics.** Persistence is strong. The validation-selected
   `mlp_residual` had test free-running R² −0.51; VAR reached 0.53 but was not
   selected. No learned family beat persistence by median validation skill.
3. **Intervention evidence.** The Randi et al. (2023) atlas yielded 3,371 valid
   events from 110 animals and 259 targets. Response reliability was low, and a
   held-out-target operator reached pair Spearman 0.098 versus 0.054 for a simple
   baseline. The +0.05 effect gate was missed; connectivity AUROC was 0.509
   versus 0.762 for a connectome ridge.
4. **Branching and turn direction.** Branching showed an entropy deficit, not
   excess: stochastic VAR TV 0.513, Markov 0.264, marginal 0.245. Turn-direction
   decoding found no uncertainty-bounded neural increment beyond behaviour.
5. **Free-running simulation.** A software-correct simulator was built, but the
   selected model failed free-running advantage, branching and perturbation
   specificity. Only the rollout contract and the perturbation interface passed;
   biological validation was not evaluable.
6. **Calibration and joint dynamics.** The historical stochastic model pooled
   multi-step residuals as one-step innovations. Correcting the pool reduced H20
   energy from 4.609 to 2.700 and moved 95% coverage from 0.999 to 0.947, but 50%
   coverage is 0.588 and covariance fidelity fails.

## 3. The central result

The final experiment fixed a 10-D PCA on training frames only and compared six
models on 175 overlapping test landmarks at H20: J0 persistence, J1 independent
AR(1), J2 population VAR with diagonal innovations, J3 population full
covariance, J4 independent full covariance, and J5 population VAR with corrected
lead-1 block-bootstrap innovations. J1/J4 and J2/J3 form a 2 × 2 factorial
separating the population mean from innovation covariance. All models use the
same initial state and 100 ensemble members.

| Contrast | H20 improvement | 95% interval |
| --- | ---: | --- |
| J5 vs J1, state energy | **+0.446** | **[0.183, 0.660]** |
| J5 vs J1, trajectory energy | **+0.375** | **[0.218, 0.522]** |
| J5 vs J0, state energy | +0.250 | [0.020, 0.427] |
| J5 vs J0, trajectory energy | +0.226 | [0.089, 0.337] |
| J5 vs J1, full-neuron CRPS | +0.005 | [−0.007, 0.016] |
| J5 vs J1, all-neuron energy | +0.203 | [−0.010, 0.442] |

Positive means lower score for the first model. The population advantage is
positive in 4/5 sessions (session 1 negative, session 2 ambiguous), survives
bootstrap block lengths 3, 5 and 10 against independent AR, and is not an
artefact of one sampler: J2 is effectively identical (energy 2.700, trajectory
2.058). The benefit is a **conditional-mean** benefit. The mean main effect is
+0.221; full covariance is −0.086 and has the worst covariance error (0.762).
The marginal CRPS gain is unreliable (its interval includes zero), and all-neuron
energy is unresolved. The 10-D result is the strongest and should not be
generalized to all neurons.

## 4. What does not hold

The model is **not** a validated generative simulator. The final label is
`PROBABILISTIC CALIBRATION NOT ESTABLISHED`. Full-neuron 50% coverage is 0.588,
10-D PCA 50% coverage is 0.546, and neural covariance error is 0.604 against a
0.50 gate. A validation-fitted Student-t correction is worse (energy 2.829), as
is a parameter-uncertainty ensemble (2.807).

The result is also **exploratory**. The test split was reused, the 175 landmarks
overlap, and the five sessions are repeated recordings, not independent animals.
The interval resamples landmarks, not animals, and is likely optimistic. Against
persistence, the block-10 energy interval includes zero and only sessions 0, 3
and 4 are positive.

No biological intervention claim is supported. There is no matched stimulation
dataset for the fitted model, no randomized perturbation and no demonstrated
model-to-worm mapping. Every model edit is a mathematical state perturbation:

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

Biological validation: `NOT_EVALUABLE`.

## 5. Interpretation

The contribution is not a new architecture. It is a bounded demonstration that
**population coupling carries joint information independent per-neuron models
miss**, plus negative results that rule out several attractive but unsupported
stories: BunDLe-Net as leak-free representation, branching as excess entropy,
broad free-running superiority, and virtual perturbations as causal predictions.

Evaluation geometry matters. In full-neuron MSE, the earlier frozen VAR was worse
than persistence. In the fixed 10-D PCA with an adaptive population mean, J5 and
J2 beat independent AR on joint energy. These statements are compatible because
they use different models, metrics and sampling schemes. Neither licenses a
general claim that the simulator “predicts the worm”.

## 6. Two bounded next studies

**A. Replicate the population mean.** Freeze the representation and the H20
joint-energy endpoint; obtain a cohort with documented independence; compare
J5/J2 against J1 and persistence with animal-level resampling. A 6-ECTS scope is
150–180 hours: data audit/adapter 30, frozen reanalysis 35, prespecified
evaluation and uncertainty 35, calibration 20, reproduction 25, writing 25–35.
Success is a prespecified positive interval; a negative result is still a result.

**B. Make intervention-aware simulation a design problem.** Specify target,
intervention, timing, sham and outcome before modelling; audit whether matched
data exist; run a minimal held-out-animal, held-out-target falsification test.
The deliverable is a protocol plus a bounded analysis, not a causal simulator.
If no matched data exist, the correct output is a data request and a power
analysis.

Both are detailed in
[HANDOFF_RESEARCH_PROPOSALS.md](HANDOFF_RESEARCH_PROPOSALS.md); the decisions
that change the choice are in
[HANDOFF_QUESTIONS_FOR_DENIS.md](HANDOFF_QUESTIONS_FOR_DENIS.md).

## References

Kato, S. et al. (2015). *Cell* 163(3), 656–669.
<https://doi.org/10.1016/j.cell.2015.09.034>

Kumar, A., Gilra, A., Meunier, A. & Grosse-Wentrup, M. (2023). BunDLe-Net.
*bioRxiv* preprint. <https://doi.org/10.1101/2023.08.08.551978>

Grosse-Wentrup, M., Kumar, A., Meunier, A. & Zimmer, M. (2024). *PLOS
Computational Biology* 20(12), e1012674.
<https://doi.org/10.1371/journal.pcbi.1012674>

Randi, F., Sharma, A. K., Dvali, S. & Leifer, A. M. (2023). *Nature* 623,
406–414. <https://doi.org/10.1038/s41586-023-06683-4>
