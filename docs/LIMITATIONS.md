# Limitations

Limitations are grouped by whether they are structural (cannot be fixed with this
dataset), statistical (fixable with more data/compute), or interpretive (fixable
only with a different experiment). None is hidden in an appendix.

---

## 1. Structural limitations of the dataset

### 1.1 Five worms, one laboratory, one preparation

All results come from five wild-type, unstimulated, immobilised worms from a
single study. There is no independent replicate cohort, no second lab, and no
freely moving preparation. The unit of generalization is the worm, and n=5 is
small. Per-worm heterogeneity is large (persistence one-step R² ranges from 0.135
to 0.593), so a mean over five worms is not a population estimate.

### 1.2 Partial coverage of the nervous system

Each worm has 109–135 segmented traces, of which only 31–47 are named. The
nervous system has 302 neurons. The model therefore sees a biased, incomplete
subset, and 60–70% of the traces cannot be assigned to a cell. Any claim about
"whole-brain dynamics" is false for this pipeline; it is a partial population.

### 1.3 Immobilised preparation

The worms are immobilised. Proprioceptive feedback, muscle activity, and
locomotion-related signals are altered or absent. Behaviour is inferred from
neural activity rather than observed as free movement. This limits transport to
natural behaviour and makes "behaviour decoding" partly a readout of the
annotation model.

### 1.4 Behaviour labels are not independent ground truth

The eight states are derived from calcium in command/interneuron populations.
Excluding eight named cells reduces but does not eliminate circularity. The labels
are coarse, imbalanced, and annotated with an unknown error rate. No independent
behavioural validation (video tracking, manual scoring) is available here.

### 1.5 No interventions

There is no optogenetic, chemogenetic, mechanical, or thermal perturbation in the
primary data. The intervention engine is therefore observational-only. This is the
single most important limitation and the reason for the standing warning.

### 1.6 No stimulus variation in the primary file

`WT_NoStim.mat` has no controlled stimulus. The system is spontaneous. This limits
the ability to identify input-driven dynamics, and it means "non-stationarity" may
reflect slow internal state rather than a controlled experimental variable.

---

## 2. Measurement limitations

- ΔF/F is a slow, nonlinear, noisy proxy for calcium, not spike rate or membrane
  potential. Deconvolution was not performed. The 2.9 Hz sampling rate aliases
  fast dynamics.
- Bleaching correction was performed upstream; residual drift remains.
- Neuropil contamination, motion artifacts, and z-drift are not modelled.
- Unnamed traces may include non-neuronal sources or duplicate segmentations.
- The eight behaviour states are not a linear scale; macro-collapsing to four
  states loses information and mixes distinct motor programs.

---

## 3. Modelling limitations

### 3.1 Fixed compute budget

All models were trained with one configuration, 60 epochs, and a small
hyperparameter budget. The GRU in particular may be under-trained. The correct
claim is "under this budget", not "these architectures cannot model the system".

### 3.2 Model classes are simple

The dynamics suite contains persistence, VAR, MLPs, a GRU, PCA+MLP, and PCA+kNN.
It does not contain switching linear dynamical systems, state-space models with
input-dependent transitions, transformers, or neural ODEs. Phase 12 adds a
hierarchical Bayesian *measurement* model for reliability, not a new dynamics or
perturbation model. A model class not tested could perform better.

### 3.3 One-step training, multi-step evaluation

Models are trained on one-step prediction and evaluated on open-loop rollouts.
This is a known mismatch; scheduled sampling, multi-step losses, or trajectory
training could improve rollouts. They were not used.

### 3.4 Latent models use PCA

The latent models use a linear PCA projection fitted on training frames. This is
interpretable and leak-free but limits the latent representation. BunDLe-Net is a
nonlinear alternative; its reimplementation is partial.

### 3.5 No explicit uncertainty model for neural transitions

The MLP/VAR models are deterministic apart from injected rollout noise. They do
not model heteroscedastic or state-dependent uncertainty. The kNN model is
stochastic but non-parametric and OOD-sensitive. Predictive intervals are
therefore not calibrated.

### 3.6 Behaviour probe is auxiliary

The behaviour head is trained with a small auxiliary loss, not as a dedicated
decoder. A dedicated classifier could improve accuracy. That would not by itself
fix the dynamics result, but it means the decoding comparison is not a
best-effort readout.

### 3.7 Causal graph is assumed

The intervention engine edits model state; it does not identify a causal graph.
No invariance, instrumental-variable, or interventional-data constraint is used.

---

## 4. Evaluation limitations

### 4.1 Blocked split is interpolation

The primary benchmark holds out a middle block and trains on later data. It is not
a forecast. It answers "can the model interpolate within this recording?", not
"can it predict the future?".

### 4.2 Strict-forward split has degenerate late blocks

In several worms the final block is nearly a single behaviour. A model can appear
to fail because the test distribution is degenerate, not because its dynamics are
wrong. This is a property of the data, and it is why the forward result is framed
as a non-stationarity stress test rather than a clean forecast benchmark.

### 4.3 Metrics are pooled and can hide per-worm failure

Pooled R² and mean accuracy can be dominated by high-variance worms. Per-worm
tables and ranges are reported, but the headline means are still aggregates.

### 4.4 Bootstrap intervals are optimistic

The cluster bootstrap resamples frames, not contiguous time blocks. Adjacent
frames are correlated, so intervals are narrower than a block bootstrap would
give. No multiple-comparison correction is applied to the 26 candidate tests.

### 4.5 Transition TV is a marginal statistic

Matching transition marginals does not mean matching trajectories. A model can
score well on TV while failing rollouts (the plain MLP does). TV is reported as a
secondary, not primary, metric.

### 4.6 No external test set

There is no held-out worm, lab, or dataset. The five worms are used for both model
comparison and interpretation. Leave-one-worm-out is not reported as a
generalization claim.

---

## 5. Branching limitations

- "Branching" is defined as local outcome entropy in a 3-D projection at a fixed
  horizon. It is projection-dependent (Spearman 0.71–0.77 across dimensions) and
  horizon-dependent.
- The shuffled null destroys the state→future association but may also destroy
  temporal structure, making it conservative or anti-conservative in unknown
  ways. A block-preserving null was not run.
- Only worm 0 was analysed in the saved artifact. The result should not be
  generalized to all worms without running the others.
- High entropy is not a bifurcation, an unobserved state, or a causal branch
  point. The analysis can fail to find excess diversity; it cannot establish its
  absence in the underlying system.
- The model-based branching comparison is descriptive; it is not a likelihood
  ratio test between a branching and non-branching model.

---

## 6. Intervention limitations

- **No causal validation.** The engine edits model state; it says nothing about a
  living animal.
- **Nonspecific effects.** The top candidate is exceeded by 4/19 non-candidate
  controls; empirical p≈0.21. The candidate ranking should not be interpreted as
  a biological priority list.
- **OOD extrapolation.** 8–38% of simulated states are outside training support.
  The largest control effects occur at the highest OOD.
- **Unit mapping.** A +2 SD model edit has no known biological dose. It is not
  "2× activity" or a given light power.
- **Single model, single worm, single seed set.** The scan uses one MLP checkpoint
  (worm 0, seed 17) and three simulation seeds. It is not a model-ensemble or
  cross-worm result.
- **Label feedback.** Behaviour is decoded by a probe trained on the same
  observational data; the outcome is not independent of the model's training
  distribution.
- **Control definition.** "Random" controls are randomly sampled non-candidate
  named cells, not a biologically matched cell class. A better control would match
  cell type, anatomy, and baseline activity.
- **No dose–response.** Only ±2 SD was scanned for most cells; a real
  intervention's effect may be non-monotonic.

---

## 7. Reproduction limitations

- **BunDLe-Net.** The official artifacts were fitted on all windows; kNN on them
  has embedding-level leakage (the encoder saw the test frames), and the
  silhouette is fully in-sample. Our reimplementation is
  partial: no commutativity loss, causal filter instead of zero-phase, single
  blocked holdout instead of 7-fold CV, and a different normalisation treatment.
  It is not a faithful reproduction and its causal-filter kNN accuracy (0.407)
  should not be
  compared directly with the paper's number.
- **NC-MCM.** We verified distributed artifacts and ported the Markov test; we did
  not retrain the 28-regression + k-means pipeline from raw calcium. The p-value
  agreement is a port check, not an end-to-end reproduction.
- **Upstream code licences.** BunDLe-Net and NC-MCM have no licence file in the
  cloned checkouts; their code is not copied into `src/`. Verify licences before
  redistribution.
- **Clean-environment rerun.** Not performed. Manifests for the main suites were
  written after the runs. Bitwise determinism is not claimed.
- **Version drift.** The environment is pinned in `pyproject.toml`/`uv.lock` where
  possible, but the runs used a user-level Python environment; a fresh `uv sync`
  may resolve slightly different transitive versions.

---

## 8. Interpretive limitations

- A failure to beat persistence does **not** prove the brain has no dynamics. It
  shows that this model class, this preprocessing, this partial population, and
  this evaluation do not extract predictive dynamics from these five recordings.
- A negative branching result does **not** prove the system is not multi-stable.
  It shows that this projection, horizon, and null do not reveal excess local
  diversity.
- A nonspecific virtual effect does **not** prove RIVL is biologically irrelevant.
  It shows that this fitted model's response is not distinguishable from arbitrary
  cells, which is a statement about the model and the analysis.
- Observational prediction, simulator consistency, biological plausibility,
  intervention generalization, and causal identification are five different
  things. This project establishes, at best, weak observational consistency; it
  establishes none of the last three.

---

## 9. What is nevertheless usable

- A reproducible, leakage-audited pipeline for this dataset.
- A strong negative benchmark that future methods must beat (persistence, label
  persistence, shuffled branching null, random-neuron control).
- A verified Python port of the NC-MCM Markov test.
- An honest separation of leaky official artifacts from the held-out
  reimplementation.
- A concrete, falsifiable next experiment.


---

## 10. Measurement-limit limitations (Phase 10)

- The reliability model is a moment approximation, not a likelihood or REML
  fit. It assumes additive, exchangeable noise.
- The pooled ceiling curve is model extrapolation beyond each target's observed
  animal count; those points are flagged `extrapolated`.
- Target-level reliability rankings do not replicate across animal halves. They
  cannot be used to prioritise targets.
- The animal jackknife interval does not reach nominal coverage in simulation.
  It is a diagnostic, not a calibrated confidence interval.
- The animal-budget calculator is a plug-in planning approximation. It has not
  been validated prospectively on a new cohort.
- The atlas has 110 animals; pooled estimates above that count are arithmetic,
  not observations.
- None of these results identifies a mechanism, a causal effect, or a digital
  twin.


## 11. Coverage and uncertainty limitations (Phase 11)

- The pooled ceiling is sensitive to responder coverage: 0.833 with all
  responders, 0.560 when each responder must be observed in at least 80% of a
  target's animals. The filter changes the responder population as well as the
  sampling correction, so 0.560 is not a bias-corrected 0.833. It is a
  descriptive bound for broadly observed responders, not a population parameter.
- The extrapolation gate failed (median absolute error 0.210, bias +0.016), so
  the curve beyond the observed animal count is not validated.
- The two-way percentile bootstrap is biased upward on the real atlas: the
  point estimate 0.314 falls outside [0.497, 0.583]. It is not a calibrated
  interval and is not used for claims.
- The synthetic coverage study (coverage 1.000, width 0.185) is conservative
  under an additive model with balanced coverage; it does not validate the
  real-data interval.
- Empirical-Bayes shrinkage does not recover target rankings at coverage 0.8;
  no ranking is published.


## 12. Hierarchical measurement and calibration limitations (Phase 12)

- The hierarchical model is additive and Gaussian. The response table has a
  standard deviation of 23.8 and heavy outliers; the likelihood is misspecified
  for that distribution. The model is a sensitivity analysis, not a validated
  measurement model.
- SBC rank uniformity failed (KS p = 1.4e-06, mean rank 0.568), SBC coverage was
  0.923 against a [0.93, 0.97] gate, held-out coverage was 0.896 with median
  absolute error 0.191, and the animal-wise permutation control gave median
  R(22) = 0.176 (q95 = 0.525). Any one of these blocks a calibrated interval;
  together they establish positive null bias and model misspecification.
- The point estimate (0.304; ceiling 0.551) is within 0.05 of the Phase 11
  moment estimate (0.314), but agreement of point estimates does not repair a
  failed interval. The 0.560 descriptive ceiling remains the headline.
- The prior sensitivity gate passed (range 0.003 across HalfNormal scales 0.5,
  1.0, 2.0), but this only shows the result is not driven by the prior scale; it
  does not show the likelihood is correct.
- The deterministic grid posterior is discretised. Boundary mass is low
  (maximum 0.0069 at coverage 0.8), but low boundary mass is not a convergence
  or calibration guarantee.
- Animal centering removes stable animal-wide offsets. The uncentered fit gave
  0.275; the centered fit gave 0.304. This choice is documented, not selected
  post hoc to maximise the estimate, but it changes the estimand.
- The coverage filter changes the responder population: coverage 1.0 gives 79
  targets and a higher model estimate (0.359), which is not comparable to the
  126-target 0.8 estimate without acknowledging the population change.
- No target ranking is published. `design_posterior.csv` is a conditional
  planning table, not a prospective sample-size guarantee.
- Benchmark comparisons use 88--96 animal clusters. The largest paired delta
  (global mean vs static CRO, +0.189) had MDE 0.238 against a 0.05 SESOI and a
  Holm-adjusted p-value of 0.185. Every configuration is underpowered; none is
  supported as superior.
- The cluster bootstrap resamples whole animals, which respects animal sharing
  but has low resolution with this cohort; it does not fix confounding by
  target, session, or responder coverage.
- None of these results identifies a mechanism, a causal effect, a digital twin,
  or a foundation model.
