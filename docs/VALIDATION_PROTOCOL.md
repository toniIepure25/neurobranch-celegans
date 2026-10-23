# Validation Protocol and Red-Team Audit

This protocol was written to make failure detectable. It is **not** a
pre-registration: the analyses were exploratory, the dataset was already public,
and several gates were sharpened after a failure was observed (noted below). No
external held-out dataset was used. Treat the outcome as an internal audit, not a
confirmatory trial.

The adversarial questions are: what would falsify each claim, what is the
strongest trivial baseline, where could leakage enter, and which alternative
explanation survives the controls?

---

## 1. Gates and outcomes

| Gate | Question | Decision rule | Outcome |
|---|---|---|---|
| G1 integrity | Is the raw file the expected bytes? | SHA256 match; no NaN explosion | **Pass** |
| G2 leakage | Can any window cross a split or use held-out frames for fitting? | Masks disjoint; scaler/PCA/probe train-only; purge respected | **Pass** (tests) |
| G3 dynamics | Does any model beat persistence on held-out one-step R²? | Paired ΔR² > 0 with 95% CI excluding 0, and wins > 7/15 | **Fail**: best is residual +0.011, CI [−0.008,+0.030], 7/15 |
| G4 behaviour | Does any model beat label persistence? | Accuracy > label persistence with CI | **Fail**: 0.80–0.83 vs 0.971 |
| G5 rollout | Does any model beat persistence at h=3 and h=10? | Lower MSE at both | **Fail**: persistence best at h=3, VAR slightly better at h=10 |
| G6 forward | Do conclusions survive strict chronology? | Direction consistent with blocked | **Fail for nonlinear models**; residual tracks persistence only |
| G7 branching | Is local outcome diversity greater than a shuffled null? | Excess entropy > 0, p < 0.05 | **Fail**: excess −0.382 bits, p=1.0 |
| G8 intervention specificity | Is the top candidate larger than matched random-neuron controls? | Candidate |Δ| exceeds control distribution, p < 0.05 | **Fail**: p≈0.21, 4/19 controls larger |
| G9 OOD | Are effects concentrated in support? | OOD fraction low and effects not explained by OOD | **Fail/weak**: OOD 8–38%; largest control effect has highest OOD |
| G10 NC-MCM | Does the Python test agree with distributed p? | Absolute difference < 0.12 at K=3 | **Pass** for all five (max 0.023) |
| G11 BunDLe honesty | Are leaky and held-out metrics separated? | No claim of generalization from official artifacts | **Pass** (documented; causal-filter reimplementation 0.407 held-out) |
| G12 causal | Is there a real matched intervention? | Evidence gate validates | **Not attempted**; gate rejects |

A gate "fails" when the evidence does not support the positive claim. A failed
gate is a result, not a reason to change the metric.

---

## 2. Leakage audit

| Path | Risk | Control | Status |
|---|---|---|---|
| Temporal adjacency | Adjacent frames nearly identical; random splits leak | Contiguous blocked/temporal splits + 12-frame purge | Controlled |
| Window boundary | History/target can cross a purge gap | `split_starts` requires all frames in one mask; tested | Controlled |
| Scaling | Mean/SD fitted on all frames leaks test distribution | Fit on train mask only; tested | Controlled |
| PCA | Latent projection fitted on all frames | Fit on train frames only | Controlled |
| Behaviour probe | Trained on test windows | Trained on train windows only | Controlled |
| Early stopping | Test used for model selection | Validation block only | Controlled |
| Hyperparameters | Test-tuned | Fixed before suite; no per-worm tuning | Controlled, but see §5 |
| Labels | Labels derived from neurons also used as predictors | 8 label-defining neurons excluded | Partially controlled; residual circularity remains |
| Official BunDLe | Encoder fitted on all windows, including test frames | Reported as embedding-level leakage, never as generalization | Controlled by disclosure |
| Intervention initial states | Sampled from test frames used to fit? | Test frames are held out from training; initial states are test REV frames | Controlled |
| Random controls | Controls sampled from candidate pool | Exclude candidates; re-ran | **Bug found and fixed** |

### 2.1 The control bug

The first random-control run sampled 10 cells from `data.predictor_names`
including the candidate cells (RIVL, RIML, AIBL, …). The resulting "null" was a
subset of the candidates, so the comparison "candidate vs random" was circular.
The corrected code excludes all `available_candidates(data)` and samples 19
non-candidate named cells. The corrected null contains larger effects than the
top candidate. This is a concrete example of why the audit is included.

### 2.2 Residual circularity

Excluding AVA/SMD/RIB removes direct label leakage, but the labels were assigned
using the worm's behaviour and the remaining neurons are recorded simultaneously.
No analysis in this repository can make the labels independent. All decoding and
branching results are therefore **observational consistency** statements.

---

## 3. Baseline audit

| Baseline | Why it is the right null | Result |
|---|---|---|
| Persistence (`x̂_{t+1}=x_t`) | Neural and behavioural states are highly autocorrelated | Beats all learned models on one-step R² and h=3 rollout |
| Label persistence (`b̂_{t+1}=b_t`) | Behaviour is strongly autocorrelated | 0.971 vs 0.80–0.83 for models |
| VAR | Linear dynamics is the standard strong baseline | Statistically indistinguishable from persistence (+0.004) |
| Test-set mean | Checks whether R² is being computed against a trivial predictor | Models with negative R² are worse than this |
| Shuffled outcomes | Preserves candidate geometry and outcome marginal, destroys state→future link | Observed branching entropy is *below* this null |
| Non-candidate neurons | Matched intervention dose and timing, arbitrary target | Candidate effect is within the null range |
| PCA-3 | Is the learned embedding more than a linear projection? | Official BunDLe silhouette is higher; but the artifact is upstream-fitted and leaky |

### 3.1 Multiple comparisons

The candidate scan tests 26 perturbations. No correction is applied to the
reported bootstrap CIs; they are descriptive. The specificity test uses the
control distribution and is explicitly one-sided and uncorrected. With 26 tests,
the largest effect is expected to look impressive by chance; the control
comparison is the relevant guard, and it fails to support specificity.

---

## 4. Alternative explanations

### A. The model is under-trained, not the hypothesis false

**Observation.** MLP/GRU/latent models are worse than persistence; residual MLP is
marginally better.

**Alternative.** With more capacity, longer training, or tuned learning rates,
nonlinear models could win.

**Discriminating control.** The residual MLP has the same backbone and *does*
match persistence, so optimization is not catastrophically broken. The direct MLP
is significantly worse than persistence, which is a model-class/regularization
problem as much as a data problem. A fair test would tune all models on a
validation split with equal compute; that was not done (fixed budget).

**Status.** Unresolved. The claim must be "under this fixed budget", not
"nonlinear dynamics are absent".

### B. Non-stationarity, not absence of dynamics

**Observation.** Strict-forward performance collapses on worms 0 and 1.

**Alternative.** The underlying dynamics are real but change over the recording
(adaptation, oxygen, locomotion state), so a single stationary model cannot
forecast.

**Discriminating control.** Fit per-block models or time-varying parameters and
test whether forward performance recovers. Not run. The blocked benchmark is
consistent with a time-varying system, but cannot identify it.

**Status.** Unresolved and biologically plausible. This is the most important
alternative explanation for the negative dynamics result.

### C. The branching null is too strong

**Observation.** Observed entropy 0.578 < null 0.960.

**Alternative.** The permutation destroys not only the state→future link but also
spatial/temporal structure that makes neighbours informative; the null is
conservative.

**Discriminating control.** A block-preserving permutation (shuffle outcomes
within behaviour and time blocks) or a generative null that matches the
autocorrelation. Not run. The negative excess is large (−0.38 bits), so it is
unlikely to be reversed by a slightly weaker null, but the effect size is
uncertain.

**Status.** Direction robust to projection dimension; null calibration unresolved.

### D. The intervention effect is an OOD artifact

**Observation.** Largest effects occur at OOD 10–38%.

**Alternative.** The model extrapolates; the apparent effect is not meaningful
even inside the model's intended domain.

**Discriminating control.** Restrict initial states and rollouts to low-OOD
regions, or use a model with explicit support constraints. Not run at scale. The
OOD flag makes the concern visible; it does not resolve it.

**Status.** Unresolved; this is why the output is labelled non-causal.

### E. Behaviour labels are the bottleneck

**Observation.** Decoding is below label persistence; branching is negative.

**Alternative.** The labels are coarse/noisy and derived from the same neurons, so
both the outcome and the branching analysis are limited by annotation, not by
neural dynamics.

**Discriminating control.** Independent behavioural tracking (video) or a
different label set. Not available in this dataset.

**Status.** Unresolved; structural limitation.

---

## 5. Stability and selection

- **Seeds.** The blocked suite uses 3 seeds. Seed-to-seed SD of one-step R² is
  0.05–0.17, comparable to the between-model differences. The residual advantage
  (+0.011) is far below seed noise.
- **Worms.** Five worms, treated as the unit of generalization in the paired
  analysis. The per-worm spread is large (persistence R² 0.135–0.593). No
  leave-one-worm-out claim is made.
- **Hyperparameters.** Fixed, not tuned per worm. This is conservative for the
  negative claim and unfair to the learned models; both facts are stated.
- **Selection.** The candidate list is literature-motivated, not learned from the
  test set. The control list is random with a recorded seed. The top candidate is
  reported without post-hoc swapping.
- **Stopping.** Training uses early stopping on validation; the best epoch is
  recorded. No test-based stopping.

---

## 6. Reproducibility audit

| Requirement | Status |
|---|---|
| Raw data hash | Recorded and checked |
| Config hash | Recorded in run manifest |
| Git commit + dirty flag | Recorded per suite |
| Split manifest | Per-worm frame counts and bounds in run manifest |
| Seeds | Recorded |
| Environment | Python/NumPy/SciPy/sklearn/torch/platform recorded |
| Exact command | Recorded in run manifest |
| Per-run predictions | `test_predictions.npz` saved |
| Metric code version | Tagged `neurobranch.evaluation.metrics.v1` in manifest |
| Clean-environment rerun | **Not performed**; this is a limitation, not a claim |
| Bitwise determinism | **Not claimed** (CPU threading/PyTorch kernels) |

The manifests for E2/E3 were generated after the runs from the saved configs and
data, because the manifest writer was added later. They record the current Git
state, which may be dirty. They do **not** prove the exact historical commit.
This is stated rather than hidden. A future rerun should be done from a clean
commit with the manifest writer active.

---

## 7. Claim grading

| Claim | Grade | Wording allowed |
|---|---|---|
| "No model clearly beats persistence under blocked interpolation" | **Survives** | "Under a fixed budget and five worms, ..." |
| "Nonlinear forecasting fails under strict chronology" | **Survives (descriptive)** | "The tested models fail; this is consistent with non-stationarity, not proof of it" |
| "Behaviour decoding is below label persistence" | **Survives** | "Observational consistency only" |
| "3-D branching reflects multi-stability" | **Falsified / unsupported** | Do not claim |
| "RIVL is a specific virtual control point" | **Weakened / unsupported** | "A model-dependent effect within the random-neuron range" |
| "NC-MCM states are Markovian" | **Mixed** | "Two of five worms reject at p<0.16; not uniform" |
| "BunDLe-Net reproduces at 0.95+ kNN" | **Unsupported** | "Official artifacts have embedding-level leakage; our causal-filter reimplementation is 0.407 held-out" |
| "We causally identified a perturbation" | **Falsified** | Never claim |

---

## 8. Remaining tests before a paper claim

1. Clean-environment rerun of E2/E3 from a tagged commit, with manifests written
   during execution.
2. Equal-compute hyperparameter tuning on validation only, all models.
3. Time-varying or per-block models to test alternative B.
4. Block-preserving branching null and at least two more worms.
5. Low-OOD stratified intervention analysis, and a model-agnostic (VAR/kNN)
   replication of the candidate ranking.
6. A second dataset (e.g. WT_Stim or an independent lab's recordings) as external
   validation, not just a new split of the same five worms.
7. The biological experiment in [`NEXT_EXPERIMENTS.md`](NEXT_EXPERIMENTS.md).
