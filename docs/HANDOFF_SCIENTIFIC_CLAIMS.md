# Scientific claims and evidence status

**Scope.** This ledger records what the repository can defend at handoff. It is
not a thesis and not a systematic review. A status is attached to a specific
statement, dataset, metric, estimand and design; it does not transfer to another
dataset or preparation.

**Status vocabulary**

| Status | Meaning here |
| --- | --- |
| `supported` | Reproduced within this design, with an interval or exact audit that supports the stated scope. Not proof and not confirmatory outside the design. |
| `exploratory` | Directionally supported but exposed to test reuse, small session count, post-hoc model choice, or undocumented independence. |
| `negative` | The prespecified or natural comparison did not support the claim. Absence of evidence is not proof of absence. |
| `withdrawn` | A previously reported number or interpretation is superseded by a documented correction. |
| `not_evaluable` | The data or design cannot answer the question. |

## Ledger

### C1. BunDLe-Net distributed embeddings are transductive

**Status:** `supported` as a technical artifact audit.  
**Statement:** The distributed 3-D embeddings were fitted on all frames,
including later test frames, so high kNN accuracy and silhouette are in-sample
geometry, not evidence of causal representation learning or generalization.  
**Evidence:** kNN 0.949–0.984 and silhouette 0.447–0.565 on the distributed
artifacts; the audit documents the all-frame fit.  
**Source:** `docs/BUNDLENET_REPRODUCTION_AUDIT.md`;
`artifacts/reproduction_bundlenet_official.json`.  
**Caveat:** This does not show that the published method is invalid. It shows
that this distributed artifact cannot be used as a leak-free baseline.

### C2. The causal BunDLe-Lite reimplementation does not reproduce the geometry

**Status:** `exploratory` (one session, reduced training, architecture-compatible
only).  
**Statement:** With a causal filter and blocked split, the reimplementation
reached kNN 0.407 and silhouette −0.064, far below the leaked artifact.  
**Source:** `docs/BUNDLENET_REPRODUCTION_AUDIT.md`;
`artifacts/reproduction_bundlenet_lite.json`.  
**Caveat:** This is not a full BunDLe-Net reproduction. It does not establish that
the published model fails under its intended protocol.

### C3. NC-MCM's Markov test was ported, but training was not reproduced

**Status:** `supported` as a technical port; `negative` for full reproduction.  
**Statement:** The Python lag-2 Markov test agrees with the distributed p-values
within 0.023 using 1,000 surrogates, but the NC-MCM training pipeline was not
reproduced.  
**Source:** `docs/LITERATURE_REVIEW.md`; E4 in `docs/EXPERIMENT_REGISTRY.md`.  
**Caveat:** Agreement of one test statistic is not a reproduction of the full
abstraction or its causal interpretation.

### C4. Persistence is a strong baseline on Kato NoStim

**Status:** `supported` as a descriptive, within-session result.  
**Statement:** Across blocked and strict-forward evaluations, nonlinear and
latent models do not robustly beat persistence; several are substantially worse
in forward rollout.  
**Evidence:** In the core simulator, VAR was the descriptive free-running winner
(R² 0.5325) but the validation-selected `mlp_residual` had test R² −0.5098 and
OOD 0.7903. The frozen Phase 6 VAR did not beat the earlier frozen VAR.  
**Source:** `docs/SIMULATOR_RESULTS.md`, `docs/PHASE6_FINAL_SCIENTIFIC_AUDIT.md`.  
**Caveat:** The result depends on horizon, split, preprocessing and metric. A
later adaptive population model can beat persistence on some metrics; see C11.

### C5. Observed trajectories do not show excess branching

**Status:** `negative` / `withdrawn` as a positive branching claim.  
**Statement:** The observed branching statistics show an entropy deficit, not
excess; the stochastic simulator's conditional outcome distribution is worse than
a first-order Markov chain and the marginal baseline.  
**Evidence:** TV 0.513 for stochastic VAR, 0.531 for persistence, versus 0.264
for Markov and 0.245 for the marginal baseline.  
**Source:** `docs/BRANCHING_STATISTICAL_AUDIT.md`,
`docs/PHASE6_STOCHASTIC_BRANCHING.md`.  
**Caveat:** This is not evidence that no biologically meaningful decision points
exist. It is evidence that this analysis does not establish them.

### C6. Randi optogenetic responses have low repeat reliability

**Status:** `supported` as a descriptive measurement result.  
**Statement:** Split-half and repeat-stimulation reliability are low; pooled
profiles are more reliable than single-animal or single-repeat profiles.  
**Evidence:** 3,371 valid events, 110 animals, 259 targets, 305 recorded cells;
single-animal reliability 0.162 [0.134, 0.191], repeat 0.243 [0.206, 0.279].  
**Source:** `docs/CLAIMS.md`, `docs/PHASE10_RESULTS.md`.  
**Caveat:** These are target-level summaries with shared animals; they are not
independent animal estimates.

### C7. A held-out-target response operator does not meet its effect gate

**Status:** `negative` / `not supported` under the frozen rule.  
**Statement:** The CRO reaches pair Spearman 0.098 versus 0.054 for the simple
baseline, a +0.044 interval that excludes zero but misses the prespecified +0.05
effect-size gate; calibration is also outside target.  
**Source:** `docs/PHASE6_SUPERVISOR_BRIEF.md`;
`reports/phase6/selection_frozen.json`.  
**Caveat:** The direction is positive and may be worth follow-up. It is not
evidence of causal connectivity: CRO AUROC 0.509 versus 0.762 for a connectome
ridge.

### C8. Named-neuron virtual perturbations are not specific

**Status:** `withdrawn` as a specific-effect claim; `not_evaluable` for biology.  
**Statement:** After correcting the random-control sampling, the RIVL +2 SD
example gives ΔP(TURN)=+0.333, but 4/19 non-candidate cells give |Δ| ≥ 0.333 and
the empirical one-sided p is about 0.21. The effect is not specific.  
**Source:** `docs/INTERVENTION_ASSUMPTIONS.md`,
`docs/GENFAIL_PERTURBATION_IDENTIFIABILITY.md`.  
**Caveat:** The corrected controls are the only ones that should be cited. The
earlier candidate-overlapping controls are invalid.

### C9. The free-running simulator has no demonstrated dynamical advantage

**Status:** `negative` on the frozen gates.  
**Statement:** The selected model fails G2 (free-running advantage), G3
(branching), and G5 (specific perturbation); G6 biological validation is
`not_evaluable`. Only G1 (valid rollout) and G4 (interface) pass.  
**Source:** `docs/SIMULATOR_RESULTS.md`; `reports/simulator/gates.json`.  
**Caveat:** G1/G4 are software-contract gates, not scientific validity.

### C10. The Kato Stim external pilot is negative

**Status:** `exploratory` external pilot.  
**Statement:** Per-animal refits on the constant-oxygen first 360 s of seven Kato
Stim sessions produced 0/7 positive skills; median H20 skill −0.602, mean −0.593,
bootstrap 95% CI [−1.024, −0.241].  
**Source:** `reports/external_validation/kato_stim_pilot_summary.json`.  
**Caveat:** Animal independence is undocumented; the cohort is seven sessions;
the raw `WT_Stim.mat` is not in this checkout. This is not a sensory-response or
oxygen-response test.

### C11. Adaptive rolling VAR improves deterministic H20 MSE over frozen VAR

**Status:** `supported` within the adaptive repeated-session design; `exploratory`
for animal generalization.  
**Statement:** The prespecified rolling 500-frame VAR reduces paired H20 MSE by
0.657 (33.9%; 95% block-bootstrap interval [0.326, 0.986]) and improves 86.3% of
landmarks; 4/5 sessions improve.  
**Source:** `docs/ADAPTIVE_DYNAMICS_RESULTS.md`.  
**Caveat:** The gain is not a distributional reconstruction. Session 2 is
slightly worse; the largest covariance/subspace shifts are where adaptation helps
least. This is a deterministic MSE result, not the J5 energy result.

### C12. Historical stochastic overdispersion was caused mainly by the residual pool

**Status:** `supported` as a technical reproduction and variance attribution.  
**Statement:** Pooling multi-step residuals across leads 1–50 and resampling them
as one-step innovations inflated the ensemble. Correcting to lead-1 residuals
reduced H20 energy from 4.609 to 2.700 and 95% coverage from 0.999 to 0.948. The
historical model reproduces to a maximum relative difference of 5.5e-9.  
**Source:** `docs/UNCAL_RESULTS.md`; `docs/JOINTDYN_REPRODUCTION.md`.  
**Caveat:** The operator is also near unit root (spectral radius 0.991), so the
pool is dominant but not the only amplifier.

### C13. Population conditional mean improves joint H20 prediction over independent AR

**Status:** `exploratory`; internally supported but not confirmatory.  
**Statement:** In a fixed 10-D PCA fitted on training frames only, J5 improves H20
energy by +0.446 [0.183, 0.660] and trajectory energy by +0.375 [0.218, 0.522]
over J1 independent AR. J2 reproduces the result with parametric diagonal
innovations. Direction is positive in 4/5 sessions.  
**Source:** `reports/simulator/joint_dynamics/paired_scores.csv`;
`reports/simulator/joint_dynamics/gates.json`.  
**Caveat:** Test split reused; 175 overlapping landmarks; five repeated sessions,
not five animals; block bootstrap over landmarks, not animals. This is the
strongest allowed positive claim, and it is not a validation of a generative
simulator.

### C14. Full innovation covariance does not help and harms joint diagnostics

**Status:** `negative` / rejected.  
**Statement:** The full-covariance main effect is −0.086 energy and −0.071
trajectory; J3 over-covers at 50/80/95 (0.767/0.957/0.992) and has covariance
error 0.762.  
**Source:** `reports/simulator/joint_dynamics/factorial_decomposition.csv`.  
**Caveat:** This is a statement about this sample size and estimator, not a
general claim that covariance is uninformative.

### C15. Marginal per-neuron CRPS does not improve reliably

**Status:** `negative` / unresolved.  
**Statement:** J5 versus J1 full-neuron CRPS improvement is +0.0048
[−0.0073, 0.0156]. The independent AR model is already strong on marginal CRPS,
so this interval should be treated as unreliable, not as a gain.  
**Source:** `reports/simulator/joint_dynamics/secondary_paired.csv`.  
**Caveat:** The interval includes zero; this does not prove equivalence.

### C16. All-neuron energy advantage is unresolved

**Status:** `exploratory` / unresolved.  
**Statement:** J5 versus J1 all-neuron energy improvement is +0.203
[−0.010, 0.442]; excluding the top-5 variance neurons gives +0.202
[−0.017, 0.437]. The 10-D PCA result is stronger.  
**Source:** `reports/simulator/joint_dynamics/secondary_paired.csv`.  
**Caveat:** Do not generalize the PCA result to all 103 neurons.

### C17. Probabilistic calibration is not established

**Status:** `negative` / stop label.  
**Statement:** J5 full-neuron 50/80/95 coverage is 0.588/0.849/0.947; 10-D PCA
coverage is 0.546/0.830/0.938. Neural covariance error is 0.604 against a 0.50
gate. Student-t and parameter-uncertainty corrections are worse.  
**Source:** `reports/simulator/joint_dynamics/calibration_h20.csv`;
`reports/simulator/joint_dynamics/gates.json`.  
**Exact label:** `PROBABILISTIC CALIBRATION NOT ESTABLISHED`.

### C18. The 0.560 measurement ceiling is not a bound

**Status:** `withdrawn` / superseded.  
**Statement:** Phase 11's `sqrt(R(22))=0.560` was an extrapolated point estimate,
not a calibrated bound. Phase 15 reports `R_obs=0.2606` [0.1782, 0.3430],
`R_22=0.3006` [0.2087, 0.3924] (counterfactual), and `R_id=0.3501`
[0.2020, 0.4982] (39 targets with n≥22).  
**Source:** `docs/PHASE15_RESULTS.md`; `reports/phase15/summary.json`.  
**Caveat:** None of these is a target ranking, an animal-budget guarantee, or a
causal effect.

## What must not be claimed

- A validated generative digital twin or whole-brain simulator.
- A validated causal effect of any named neuron.
- Reliable prediction of a real optogenetic or other stimulation outcome.
- Animal-independent replication from five Kato sessions.
- Confirmatory significance from the reused test split.
- That the 0.560 value is a measurement ceiling or bound.
- That observed trajectories show bifurcations or excess branching.
- That BunDLe-Net was reproduced or that its distributed embedding is leak-free.

## Strongest allowed sentence

> In a fixed 10-D PCA fitted on training frames only, the corrected population-mean
> model provides evidence of improved joint neural-trajectory prediction over an
> independent-neuron AR baseline at H20, but generative calibration and
> intervention validity remain unresolved.

Every perturbation slide, table or demo must also carry:

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

Biological validation: `NOT_EVALUABLE`.
