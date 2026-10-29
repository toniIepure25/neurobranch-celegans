# Potential Paper

## Working title

**"Persistence Wins: A Leakage-Audited Benchmark for Whole-Brain Dynamics,
Branching, Virtual Perturbation, and Optogenetic Prediction in *C. elegans*"**

Alternative: *"What a 3-D Branch Does Not Prove: Negative Benchmarks for Latent
Dynamics, Virtual Perturbation and Prospective Behaviour in Whole-Brain Calcium
Imaging."*

---

## 1. One-sentence contribution

We provide a reproducible, leakage-audited benchmark showing that on the standard
Kato et al. (2015) five-worm dataset, neural and latent dynamics models do not
beat persistence, local outcome diversity is *below* a geometry-preserving null
(predictability, not branching), and model-ranked virtual perturbations are not
distinguishable from arbitrary-cell perturbations. On the Randi et al. (2023)
optogenetic atlas, pre-stimulus state and connectome features do not beat a
constant mean on held-out animals. On an independent 27-animal freely-moving
dataset, the primary neural increment in prospective reversal prediction has the
right sign but a 95% animal CI that includes zero, and it does not survive a
quadratic behavioural control or identity-aligned neurons. On the independent
Kramer et al. (2026) directed-turning archive, the Phase 3 6 s result does not
replicate (22/32 animals, CI includes zero), and an animal-disjoint
dorsal/ventral benchmark finds no bounded neural increment beyond behaviour
(Δ AUROC +0.012, CI [−0.017, +0.042]). We argue that these negatives are the
correct baseline for causal claims in computational neuroscience.

---

## 2. Why this is a paper (and what kind)

This is **not** a discovery paper and **not** a causal-identification paper. It is
a methods + benchmark + negative-results paper with an artifact-verification
component. Its value is that it removes three attractive but unsupported claims
from the literature's implicit evidence base and replaces them with falsifiable
tests:

1. **"Latent dynamics capture whole-brain dynamics."** On an honest blocked split,
   no learned model beats persistence; under a strict forward split, most collapse.
2. **"Trajectories branch."** In a 3-D projection, local outcome entropy is
   *lower* than a geometry-preserving null in 4/5 worms. That is local
   predictability, not a bifurcation; the Phase 1 one-sided p=1.0 was read in the
   wrong direction.
3. **"Virtual perturbations identify control points."** The top model-ranked cell
   is exceeded by 4/19 arbitrary non-candidate cells; the effect is nonspecific.
4. **"Optogenetic responses are predictable from pre-stimulus state."** On 3,371
   Randi events, every learned model is worse than the constant mean on held-out
   targets and animals; the event-level connectome tie is a coin flip.

The paper also documents a concrete methodological failure — a "random" control
that sampled the candidate cells — and shows how the conclusion changes when it is
fixed. That is a useful cautionary contribution.

---

## 3. Contributions

| # | Contribution | Evidence |
|---|---|---|
| C1 | Leakage-audited dual benchmark (blocked interpolation + strict forward) on all five Kato worms | 105 + 35 runs, manifests, tests |
| C2 | Persistence and label-persistence are strong baselines that most latent/neural models fail to beat | Paired CIs, 3 seeds |
| C3 | Corrected two-sided branching nulls, block/shift nulls, synthetic power checks, all five worms | 4/5 entropy deficit; no excess |
| C4 | A paired virtual-perturbation engine with OOD flags and matched non-candidate controls | 26 candidates + 19 controls/model × 6 models; none significant |
| C5 | Artifact-level verification of BunDLe-Net and NC-MCM, separating leaky upstream artifacts from held-out claims | kNN/silhouette, Markov p port, deviations |
| C6 | Open, reproducible platform (CLI, app, tests, manifests) | Automated tests, 14-tab app |
| C7 | External Randi 2023 optogenetic benchmark with event/target/animal splits and reliability ceiling | 110 animals, 3,371 events; negative |
| C8 | Incremental behaviour benchmark on transition frames and proper scoring rules | No calibrated increment |
| C9 | Independent 27-animal prospective benchmark with causal features, animal-disjoint splits and explicit nulls | Primary neural increment inconclusive; PCA secondary only |
| C10 | Randi target/shape/repeated-average/context audit | Reliability r≈0.09; averaging worse than global; context scale-only |
| C11 | Independent Kramer 2026 replication of the Phase 3 6 s protocol | 32 animals; criterion not met; 22/32 improved |
| C12 | Animal-disjoint turn-direction benchmark with neural population/SAAV/circuit/identified controls | 56 animals, 945 events; primary Δ +0.0123, CI includes 0; no representation or nonlinear rescue |
| C13 | Turn-direction dynamics, PCA trajectories, peak timing and model-dependent controls | Immediate pre-offset behaviour dominates; no identity-specific neural effect |
| C14 | Source-audited SAAV reconstruction with four offline regimes and a causal-onset online benchmark | Original-style SAAV AUROC 0.540; no properly scored animal-disjoint increment; exact reproduction blocked |
| C11 | Cross-animal one-step dynamics probe | Behaviour beats persistence; neural does not |
| C15 | Metric-aligned reliability ceilings and design budget; explicit correction of a facet mismatch | Single animal 0.162/0.403; pooled 22 animals 0.693/0.833; target rankings unstable; jackknife not calibrated |

---

## 4. Figures and what each shows

| Figure | Content | Claim supported |
|---|---|---|
| Fig 1 | Dataset coverage, behaviour composition, partial neuron counts | The data are five worms and a partial population; "whole-brain" needs qualification |
| Fig 2 | One-step R², behaviour accuracy vs label persistence, transition TV | No model clearly beats persistence; decoding is below the trivial baseline |
| Fig 3 | Strict-forward R² heatmap and behaviour accuracy | Non-stationarity breaks nonlinear forecasting; interpolation is not forecasting |
| Fig 4 | Official BunDLe 3-D embedding, two views | The geometry is behaviour-separated **in the upstream-fitted, leaky artifact**; not a generalization result |
| Fig 5 | Branching entropy, outcome mixtures, shuffled null | Observed diversity is below the null; no excess branching |
| Fig 6 | Candidate vs random-neuron intervention forest | Named effects are within the arbitrary-cell range |
| Fig 7 | NC-MCM p-value curves and port agreement | The port is faithful; Markovianity is not uniform across worms |
| Fig 8 | Observed trajectory vs model rollout | The model is a predictor, not a data generator; rollouts drift |

Supplementary: per-worm tables, all 105 runs, deviation list for BunDLeLite,
full intervention scan, projection-dimension stability.

---

## 5. Claims ladder

| Level | Claim | Status in this work |
|---|---|---|
| 1 | Observational prediction | Weak on Kato; on independent freely-moving data behavioural history predicts, but the neural increment is small and CI-inclusive |
| 2 | Simulator consistency | Partial; transition marginals can match while rollouts fail; no cross-animal generative simulator claimed |
| 3 | Biological plausibility | Not assessed |
| 4 | Intervention generalization | Assessed and negative: internal nonspecific; external optogenetic benchmark below constant; observed-response prediction only |
| 5 | Causal identification | Not attempted; explicitly disclaimed |

The paper should stay at levels 1–2 and use the failures at levels 4–5 as the
motivation for the next experiment. It must not slide from "the model predicts"
to "the brain does".

---

## 6. What the paper can and cannot claim

### Can claim

- "Under a fixed training budget and a blocked, purged split on five worms, no
  model beats persistence with a paired CI excluding zero."
- "Under a strict chronological split, the tested nonlinear models fail, which is
  consistent with non-stationarity."
- "In a 3-D PCA projection, local outcome entropy at held-out reversal frames is
  lower than a geometry-preserving null; this is evidence for local
  predictability, not branching."
- "The top model-ranked perturbation is not distinguishable from arbitrary
  non-candidate cells in the fitted model, across six architectures and seeds."
- "On the Randi atlas, pre-stimulus state and connectome features do not beat a
  constant mean on held-out targets or animals; the event-level tie is within
  ±0.018 MAE."
- "Neural activity improves transition detection but degrades calibrated
  next-state prediction relative to a state-only model."
- "The official BunDLe-Net embedding is behaviour-separated in the upstream-fitted
  artifact, with embedding-level leakage; our blocked-holdout reimplementation is
  substantially worse."
- "The Python NC-MCM Markov test agrees with distributed p-values within 0.023 at
  K=3."
- "On 27 independent freely-moving animals, adding population neural activity to
  behavioural history changes 6 s log loss by −0.0070 with a 95% animal bootstrap
  CI of [−0.0148, +0.0005]; the prespecified criterion is not met."
- "The secondary train-only PCA representation reaches a CI excluding zero, but
  the increment shrinks against a quadratic behavioural baseline, is absent for
  52 identity-aligned neurons, and reverses within-animal."
- "Randi repeated-response reliability is r≈0.09 for peak, mean and integral;
  repeated-event averaging is worse than a global mean; temporal shape is not
  reliable."
- "A cross-animal one-step behaviour ridge beats persistence on next-velocity MAE,
  but adding neural activity does not."
- "The Phase 3 6 s result does not replicate on 32 independent WT octanol animals:
  Δ −0.0057, 95% CI [−0.0113, +0.0005], 22/32 improved; criterion not met."
- "On an animal-disjoint turn-direction benchmark (56 animals, 945 events), the
  population increment is +0.0123 AUROC with a 95% CI of [−0.0166, +0.0425];
  SAAV, circuit and identified representations are null or worse."
- "The apparent turn-direction predictability is concentrated in behaviour at
  frames −1/0; the primary window excludes those frames."

### Cannot claim

- "We reproduced BunDLe-Net" (partial, with deviations).
- "We reproduced NC-MCM end-to-end" (artifact verification only).
- "We discovered a causal role for RIVL/RIVR."
- "The worm's dynamics are not multi-stable" (only: this projection/null shows no
  excess diversity).
- "Whole-brain dynamics" without qualifying the 109–135 trace coverage.
- "The model predicts a real experiment's effect size."
- "Neural activity is irrelevant to reversal or turn direction" (the primary
  results are bounded nulls, not proof of absence).
- "We reproduced Kramer et al.'s RNN or SAAV decoding" (the derived CSVs,
  preprocessing script and trial-level optogenetic data are unavailable; exact
  reproduction is `BLOCKED_MISSING_INPUTS`).
- "A diacetyl population effect is a general turn-direction signal" (secondary,
  exploratory, CI includes zero, near-chance behavioural baseline).
- "A frozen identity representation adds robust predictive value" (PCA is
  secondary; identified neurons and quadratic controls do not support it).
- "The cross-animal probe is a generative simulator" (it is one-step, linear, and
  has no rollout validation).

---

## 7. Reviewer objections and planned responses

| Objection | Response | Additional analysis needed |
|---|---|---|
| "You under-trained the models." | Fixed budget; residual MLP matches persistence, so optimization is not broken. | Equal-compute validation sweep; report best-tuned results |
| "Persistence is trivial; of course it wins at h=1." | It also wins h=3 rollout, and behaviour persistence wins by 14–17 points; that is the point. | h=10 and longer horizons; multi-step training |
| "Your split is not forecasting." | Correct; we label it interpolation and include a strict-forward negative control. | Time-varying models to test non-stationarity |
| "Your branching null is too strong." | Possible; excess is −0.38 bits and robust across dimensions. | Block-preserving null, all five worms, generative null |
| "Random controls are not biological controls." | True; they are a computational specificity null. We say so. | Cell-type-matched controls in the biological experiment |
| "The intervention is not causal." | We never claim it is; the title and every output say so. | The RIVL/RIVR experiment |
| "Five worms is too few." | Yes; we report per-worm ranges and do not claim population generalization. | External cohort/lab replication |
| "BunDLe comparison is unfair." | We separate the leaky official artifact from our held-out reimplementation and list deviations. | Full 7-fold reproduction with upstream code |

---

## 8. Target venues

In descending fit:

1. **PLOS Computational Biology** — methods/negative results with biological
   relevance; tolerates careful destructive work.
2. **NeurIPS Datasets and Benchmarks** — if framed as a benchmark + audit; the
   negative result and manifest/test infrastructure fit the track.
3. **GigaScience** — reproducibility and open data/code emphasis.
4. **eLife Tools and Resources** — if the biological experiment is added and the
   platform is the main deliverable.
5. **Journal of Open Source Software** — software note only; not enough for the
   scientific claims.

A causal-discovery venue (Nature Neuroscience, Neuron) is **not** appropriate
without the biological experiment.

---

## 9. Minimum work before submission

1. **Clean rerun** of E2/E3 from a tagged commit with manifests written during
   execution; report exact environment.
2. **Equal-compute tuning** on validation only for all models, to answer the
   under-training objection.
3. **All five worms** for the branching analysis, with a block-preserving null.
4. **Model-agnostic intervention scan**: VAR, residual MLP, latent kNN; full
   dose–response; control distribution per model.
5. **Time-varying model** under strict forward split, to test the
   non-stationarity explanation.
6. **External replication** on `WT_Stim` or an independent dataset, with the
   pipeline frozen.
7. **Biological experiment** (RIVL/RIVR) if the paper is to make any causal or
   translational claim; otherwise keep it as "proposed validation".

Without items 1–5 the paper is a strong internal report, not a submission-ready
benchmark. With them, it is a credible negative-results/methods paper even if the
biological experiment is not done.

---

## 10. Suggested abstract (draft, no causal language)

> Whole-brain calcium imaging of *C. elegans* is widely used to infer latent
> population dynamics, behavioural branching, and candidate control neurons. We
> built a leakage-audited benchmark on the five public Kato et al. (2015)
> recordings and evaluated persistence, VAR, MLP, residual MLP, GRU, PCA+MLP and
> PCA+kNN models under blocked and strict-forward splits. Under a blocked split,
> no model beat persistence with a paired confidence interval excluding zero; the
> best residual model improved one-step R² by 0.011 (95% CI −0.008 to 0.030) and
> behaviour decoding remained far below label persistence (0.80–0.83 vs 0.971).
> Under a strict forward split, direct nonlinear and latent models collapsed on
> two worms. A 3-D branching analysis at 328 held-out reversal frames found lower
> local outcome entropy than a shuffled null (0.578 vs 0.960 bits), and a paired
> virtual-perturbation scan found that the top model-ranked cell (RIVL) was
> exceeded by 4/19 arbitrary non-candidate cells.
>
> We then tested whether these negatives generalise. On 27 independent
> freely-moving animals (Baskoylu & Flavell 2026; 1,162 reversals), adding
> population neural activity to behavioural history changed prospective 6 s log
> loss by −0.0070, with a 95% animal bootstrap CI of [−0.0148, +0.0005]; the
> prespecified criterion was not met. A train-only PCA representation reached a CI
> excluding zero, but the increment shrank against a quadratic behavioural
> baseline, was absent for 52 identity-aligned neurons, and reversed within-animal.
> On the Randi et al. (2023) optogenetic atlas, peak, mean and integral responses
> all had repeated-stimulation reliability r≈0.09; repeated-event averaging was
> worse than a global mean, and temporal shape was unreliable. A cross-animal
> one-step behaviour model beat persistence, but adding neural activity did not.
> We then repeated the prospective protocol on an independent directed-turning
> archive (Kramer et al. 2026; 73 recordings, 2,670 reversals). The 6 s neural
> increment did not replicate (Δ −0.0057, 95% CI [−0.0113, +0.0005], 22/32
> animals). An animal-disjoint dorsal/ventral benchmark with 56 animals and 945
> events found a population increment of +0.012 AUROC (95% CI [−0.017, +0.042]);
> SAAV, circuit and identified-neuron representations were null or worse, and a
> nonlinear model did not rescue the effect. The largest cohort-specific point
> estimate was exploratory and its CI included zero. No trial-level optogenetic
> recordings were available, so no causal intervention claim is made.
> Official BunDLe-Net embeddings were behaviour-separated in the upstream-fitted
> artifact, which leaks test frames into the encoder, while a blocked-holdout
> reimplementation scored substantially lower. We argue that persistence, shuffled
> branching nulls, animal-disjoint prospective tests, and random-neuron controls
> should be mandatory baselines before latent trajectories or virtual
> perturbations are described as causal, and we specify the cell-specific,
> controlled experiments that would be required to support a control claim.

---

## 11. Authorship and collaboration

The natural collaboration is with the Vienna Neuroinformatics group (BunDLe-Net,
NC-MCM). Contributions would need to be negotiated: this project built the
independent benchmark and audit; the group provides the reference methods, data
context, and possibly the biological collaboration. Do not list anyone as an
author without consent. The supervisor brief lists the specific questions to
resolve before framing.
