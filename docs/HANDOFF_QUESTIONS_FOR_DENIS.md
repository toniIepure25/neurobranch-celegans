# Questions for Denis Kovačević

These are questions, not reconstructed requirements. The earlier conversation is
recorded provisionally in `docs/ORIGINAL_RESEARCH_OBJECTIVE.md`; please correct
any premise that is wrong. The order is the order in which the answers change the
project scope.

## A. The objective

**1. What is the primary scientific object?**  
Is it (a) a free-running generative model of population calcium trajectories,
(b) prediction of a specific behavioural event such as reversal onset, or
(c) a model that predicts the response to an intervention?  
*Why it matters:* the three require different data, metrics and validation. The
current repository treats (a) as primary and uses (b)/(c) as supporting
read-outs.

**2. What would "success" look like at the end of a 6-ECTS project?**  
A positive predictive result, a rigorous negative result, a reusable benchmark,
or a design for a wet-lab experiment?  
*Why it matters:* a negative replication is publishable as an MSc chapter only if
that is an accepted outcome. The current evidence is mostly negative.

**3. Is the exact assignment already fixed, or is this discussion meant to choose
it?**  
If it is fixed, which document or sentence should be treated as the
specification?  
*Why it matters:* the repository currently says the brief is provisional. I do
not want to invent requirements.

## B. The representation

**4. Is BunDLe-Net a required baseline, a conceptual reference, or a model to
extend?**  
The distributed embeddings are transductive, and the causal reimplementation did
not reproduce their geometry.  
*Why it matters:* treating BunDLe-Net as a required baseline means reproducing its
training protocol; treating it as a reference means a fixed PCA or another
leak-free representation is acceptable.

**5. What makes a low-dimensional representation "meaningful" for your purpose?**  
Behavioural decodability, dynamical sufficiency, cross-animal alignment,
interpretable axes, or stability under perturbation?  
*Why it matters:* the current 10-D PCA is an evaluation coordinate system, not a
biological latent space. A different criterion changes the representation and the
claims.

**6. Is a latent trajectory the right object, or do you need neuron-level
predictions and intervention responses?**  
*Why it matters:* the joint-energy result lives in a 10-D PCA; the all-neuron
energy result is unresolved. A neuron-level requirement would make the current
positive result insufficient.

## C. Branching and behaviour

**7. What counts as "branching" for you?**  
A conditional outcome distribution, a visible split in a 3-D projection, a
parameter-dependent change in a vector field, or a probabilistic decision point
in a fitted model?  
*Why it matters:* the current analysis finds an entropy deficit, not excess
branching. The term "bifurcation" should not be used without a demonstrated
qualitative change.

**8. Which behavioural transition is the priority?**  
Reversal onset, dorsal versus ventral turn, sustained reversal, or a
continuous behavioural variable?  
*Why it matters:* the labels are decoded from a subset of the same neurons.
Different endpoints have different circularity and different power.

**9. Are independent behavioural measurements available?**  
Video tracking, manual scoring, or a behavioural readout not derived from the
recorded neurons?  
*Why it matters:* without one, a behavioural claim is observational consistency,
not prediction of an independent outcome.

## D. Data and independence

**10. Is there an independent cohort with documented animal IDs?**  
Same-lab sessions, a second preparation, or a public release with a cohort
manifest?  
*Why it matters:* the five Kato sessions are not five documented animals. This is
the single biggest limitation on the current result.

**11. Is Kato `WT_Stim` acceptable as a limited replication?**  
It has seven sessions, but independence is undocumented and the raw file is not
in the current checkout.  
*Why it matters:* it could support a bounded replication of autonomous dynamics,
but not an animal-independent or oxygen-response claim.

**12. Is there a matched stimulation dataset with neuron identities, timing,
amplitude and a sham/off-target control?**  
*Why it matters:* no observational model can validate an intervention effect
without matched ground truth. The Randi atlas is the strongest available
response dataset, but it does not match the Kato model and has no behaviour
labels.

**13. Which neuron-identity standard should be used across animals?**  
A named-neuron intersection, a NeuroPAL-style registration, or a latent alignment
model?  
*Why it matters:* cross-animal pooling without a validated mapping is not
defensible. The current work uses named traces and per-animal models.

## E. Intervention

**14. Is a model-dependent virtual perturbation useful even when it is not
causally validated?**  
If yes, what decision would it inform: target prioritization, experiment design,
hypothesis generation, or something else?  
*Why it matters:* the current engine can produce mathematical counterfactuals.
Whether that is useful is a scientific-user question, not a software question.

**15. Which named neurons are the priority, and why?**  
The current work used AVBL/AVBR/AVEL/AVER/RIVL/RIVR as a literature-motivated
hypothesis set, not because you specified them.  
*Why it matters:* a target list changes the data request and the multiple-testing
problem.

**16. What would make a perturbation result "specific" in your view?**  
A random-neuron control, a matched off-target control, a dose-response, a
replication in an independent animal, or all of these?  
*Why it matters:* the corrected random-neuron controls already make the current
example candidate non-specific (p ≈ 0.21).

**17. Would you accept a negative intervention result as the thesis outcome?**  
For example: the model's predicted response ordering does not agree with measured
responses, and the correct next step is a prospective experiment.  
*Why it matters:* this is the most likely outcome of a bounded 6-ECTS project.

## F. Scope and deliverables

**18. Which proposal is closer to your expectation: A (replicate the population
mean) or B (intervention-aware design and falsification)?**  
*Why it matters:* they require different data, different supervision and
different success criteria. See `docs/HANDOFF_RESEARCH_PROPOSALS.md`.

**19. What would you consider a sufficient 6-ECTS contribution?**  
A replication with a negative or positive result, a benchmark and audit, a
protocol and power analysis, or a working prototype?  
*Why it matters:* 150–180 hours cannot deliver a validated causal simulator. The
scope must be chosen before the work starts.

**20. What would make you say "stop" and change direction?**  
*Why it matters:* the repository has already accumulated several negative
results. A clear stop rule prevents another architecture search on reused data.

## Questions I am not asking

I am not asking you to approve a causal claim, a digital-twin claim, or a
neuron-specific stimulation recommendation. The current evidence does not support
any of them. The purpose of the discussion is to choose a defensible next
question and the data needed to answer it.

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**
>
> These questions do not ask for approval of a causal claim. Biological
> validation remains `NOT_EVALUABLE`.
