# Phase 3 results

**Branch:** `research/phase3-neural-generalization`
**Starting commit:** `c69fd5afe1a084827ce0c64f3f2c96fa0679b077`
**Date:** 2026-10-23
**Scope:** observational generalisation and response prediction; no new animal
experiment, no causal claim.

## 1. Bottom line

| Question | Answer | Strength |
|---|---|---|
| RQ1: prospective behavioural prediction | Behavioural history predicts better than constant/state; neural increment is small and **inconclusive** at the prespecified 6 s horizon | primary negative/inconclusive |
| RQ2: Kato negative results reproduce on independent behaviour | Yes, the same pattern: no robust neural increment; transition accuracy alone is misleading | reproduced negative |
| RQ3: frozen neural representation | Train-only PCA clears the zero-CI rule as a secondary analysis, but fails the quadratic-behaviour control and identified-neuron test | not robust; STOP |
| RQ4: Randi temporal shape / averaging / context | Alternative targets do not rescue r≈0.09 reliability; repeated averaging is worse than global mean; context improves scale/outliers but only 1.8% MAE | negative / weak |
| RQ5: cross-animal dynamics without transductive representations | One-step behaviour ridge beats persistence; adding neural activity does not improve MAE or log loss | negative for neural; no generative generalisation |
| RQ6: evidence for neuron-specific control | None. A concrete causal-experiment checklist is provided | no control claim |

**No result in this repository establishes that a named neuron controls a
behavioural transition.**

## 2. Data actually used

Baskoylu & Flavell (2026), Dryad `doi:10.5061/dryad.w9ghx3g4v`, CC0-1.0:

- 27/27 animals structurally usable;
- 43,200 confocal frames, 1,600 per animal;
- 1,162 validated reversal events;
- 132–163 recorded neurons per animal (mean 147.6);
- 2,595 labelled traces, 170 unique NeuroPAL labels, 6 ambiguous duplicate-label
  traces;
- median sampling interval 0.6000189 s; duration 941.3–962.0 s;
- zero non-finite raw trace values; zero non-monotonic timestamp series.

Randi et al. (2023): 3,371 events, 110 animals, 259 targets, 305 recorded
neurons, 237,639 downstream response rows.

The published `gcamp/trace_array` is z-scored over the whole recording and was
**not** used. `trace_array_original` plus causal/train-only normalisation was
used instead. Details: [`PHASE3_DATASET_AUDIT.md`](PHASE3_DATASET_AUDIT.md).

## 3. RQ1 — prospective behavioural prediction

Target: reversal onset in `(t, t+H]`, animal-disjoint leave-one-animal-out,
prespecified H=6 s. Features are causal. The primary model is an L2 logistic
regression with fixed C=0.1; uncertainty is a 10,000-resample animal bootstrap.

| H | Behaviour log loss | + population neural | Δ | 95% CI | Animals improved |
|---:|---:|---:|---:|---|---:|
| 3 s | 0.4087 | 0.4056 | -0.00306 | [-0.00740, +0.00105] | 19/27 |
| **6 s** | **0.5927** | **0.5856** | **-0.00701** | **[-0.01483, +0.00053]** | **20/27** |
| 12 s | 0.6283 | 0.6208 | -0.00742 | [-0.01566, +0.00173] | 20/27 |

The prespecified 6 s criterion (mean Δ<0, CI upper<0, ≥70% animals improved) is
**not met** because the CI includes zero. This is not proof of absence; it is an
inconclusive/negative primary result.

Calibration at 6 s is good for both models (slope 1.005 behaviour, 0.963
behaviour+neural), so the null is not caused by a badly calibrated probability
scale.

The effect is heterogeneous: per-animal Δ ranges -0.0630 to +0.0389. Three
animals drive most of the positive tail and a different three drive most of the
negative tail. See [`PHASE3_BEHAVIORAL_VALIDATION.md`](PHASE3_BEHAVIORAL_VALIDATION.md).

## 4. RQ2 — independent reproduction of the Kato negatives

The Kato results were: no nonlinear model robustly beats persistence; neural
activity adds no calibrated behavioural value; transition accuracy alone is
misleading; branching shows an entropy deficit, not excess.

The independent freely-moving data reproduce the **scientific pattern**:

- Persistence is strong: next-frame velocity MAE 0.02722, R² 0.822.
- Behavioural history beats constant/state baselines, but the neural increment is
  small and its CI crosses zero.
- At 12 s, 60.8% of eligible frames are positive, so a constant classifier is
  60.8% accurate while its log loss is 0.673. Accuracy alone would be misleading.
- No branching-excess claim is made; the Phase 2 corrected deficit result is
  preserved.

This is a reproduction of a *negative pattern*, not a reproduction of the Kato
experiment or a new positive result.

## 5. RQ3 — frozen representations

| Representation | Δ log loss at 6 s | 95% CI | Verdict |
|---|---:|---|---|
| Population summaries (primary) | -0.00701 | [-0.01483, +0.00053] | inconclusive |
| Train-only PCA(10) | -0.00841 | [-0.01475, -0.00220] | secondary positive |
| Identified 52 neurons | -0.00172 | [-0.01054, +0.00699] | no |
| Causal delayed neural features | -0.00345 | [-0.00851, +0.00175] | small, not a clean null |

The PCA result is not robust: adding neural features to a **quadratic behavioural
state** gives Δ=-0.00524 [-0.01334, +0.00301]; the identified-neuron
representation is weaker than anonymous population summaries; the within-animal
forward task is worse with neural activity (+0.00288, 9/27 improved). The causal
delayed-feature control also improves slightly (6 s Δ -0.00345), so delayed
activity still carries slow behavioural/arousal information and the control is
not a clean null.

Decision: **STOP** for a robust frozen-representation claim. The PCA lead is
worth a preregistered replication, not a conclusion. Details:
[`PHASE3_NEURAL_REPRESENTATIONS.md`](PHASE3_NEURAL_REPRESENTATIONS.md).

## 6. RQ4 — Randi response modelling

| Target | Median repeated r | Best repeated-average MAE | Global-mean MAE |
|---|---:|---:|---:|
| Signed peak | 0.0900 | 6.3238 (pair) | **5.9008** |
| Mean post response | 0.0921 | 3.2021 | **2.8510** |
| Integral | 0.0921 | 38.4254 | **34.2115** |

Temporal shape is also unreliable: per-neuron profile r median 0.0479 (n=30,220),
population time-course r 0.1261, amplitude-vector r 0.0852, sign agreement 0.518.
Repeated-event averaging is **worse than a global mean** for every target.

Stimulus context (pre-SD, time since previous stim, previous target) improves
held-out-event MAE from 5.9967 to 5.8900 (1.8%), while RMSE falls from 60.78 to
49.91 and Pearson rises from 0.083 to 0.841. The MAE CI still overlaps and the
large correlation gain is driven by response-scale/outliers. The context result
is not a reliable single-trial response model.

Details: [`PHASE3_RANDI_RESPONSE_AUDIT.md`](PHASE3_RANDI_RESPONSE_AUDIT.md).

## 7. RQ5 — cross-animal dynamics

One-step, animal-disjoint probe on Baskoylu (features causal, scaler/Ridge fit on
training animals):

| Model | Next-velocity MAE | R² | Next-state log loss | Balanced accuracy |
|---|---:|---:|---:|---:|
| Persistence | 0.02722 | 0.8220 | — | — |
| Behaviour ridge | **0.02587** | **0.8450** | — | — |
| Behaviour + neural ridge | 0.02590 | 0.8448 | — | — |
| Behaviour logistic | — | — | **0.11631** | 0.9472 |
| Behaviour + neural logistic | — | — | 0.11655 | 0.9473 |

Neural activity does not improve next-velocity MAE (Δ +0.000031, 9/27 animals
improved) or next-state log loss (Δ +0.000240). The behavioural ridge improves
over persistence, so short-horizon dynamics are learnable across animals, but
there is no evidence that the neural representation adds generalisable dynamics.

No cross-animal generative simulator is claimed. The existing Streamlit
simulator remains the Kato-trained, model-dependent explorer; it is not a
validated cross-animal model. This is the honest RQ5 boundary.

## 8. RQ6 — intervention limitations

No neuron-specific control is supported. The required evidence is a
preregistered, cell-specific, bidirectional perturbation with:
randomisation, mock/light/off-target controls, dose–response, a temporal window,
animal-level replication, blinded scoring, a manipulation check, mediation, and
cross-animal identity verification.

See [`PHASE3_INTERVENTION_LIMITATIONS.md`](PHASE3_INTERVENTION_LIMITATIONS.md).
The 6-ECTS-feasible subset is observational replication, stronger behavioural
baselines, identity reproducibility and Randi window/target sensitivity — not a
new optogenetics rig.

## 9. Engineering evidence

- New Baskoylu HDF5 adapter with strict event-index validation (0-based start,
  exclusive end; 1,162 events; 7 initial partial reversals).
- One-time JLD2→TSV NeuroPAL export via vendored Julia script; Python pipeline
  has no Julia dependency at analysis time.
- Causal features, train-only scalers/PCA, animal-disjoint splits, chronological
  purge, causal delayed-feature control, quadratic-behaviour control, C sensitivity.
- New Phase 3 Randi target/context/temporal-shape analysis.
- New cross-animal dynamics probe.
- New Streamlit **Prospective Behavior** tab; all seven Phase 2 tabs retained.
- New tests: HDF5 parsing, event indices, timestamps, identity ambiguity,
  causal normalisation, horizon labels, purged split, calibration, app tabs,
  Randi target schema.
- All prior Phase 1/2 artifacts, manifests and warnings preserved.

## 10. What is still not known

- Whether the small neural direction is real in a new cohort.
- Whether a nonlinear behavioural/arousal baseline can fully explain it (the
  quadratic control suggests much of it can).
- Whether a better cross-animal identity representation would change RQ3.
- Whether any neuron is necessary or sufficient for a reversal.
- Whether the Randi response window is the wrong measurement; reliability may be
  a window/aggregation artifact, but that requires a prespecified sensitivity
  study.

## 11. Artifact index

| Artifact | Contents |
|---|---|
| `reports/phase3/dataset_manifest.json` | 30 Dryad files + SHA256 + derived label hash |
| `reports/phase3/dataset_coverage.csv` | per-animal QC |
| `reports/phase3/neuron_identity_coverage.csv` | per-trace NeuroPAL labels/ambiguity |
| `reports/phase3/prospective_metrics_*.csv` | per-animal proper scores |
| `reports/phase3/prospective_deltas_*.csv` | paired neural increments |
| `reports/phase3/prospective_summary.json` | aggregate + bootstrap CIs |
| `reports/phase3/prospective_c_sensitivity.csv` | C sensitivity |
| `reports/phase3/prospective_within_animal_metrics.csv` | chronological task |
| `reports/phase3/randi_*` | Randi tables and summary (trial table regenerated, untracked) |
| `reports/phase3/cross_animal_dynamics.*` | RQ5 probe |
| `reports/phase3/figures/*.png` | six figures |

> All virtual interventions remain a **MODEL-DEPENDENT VIRTUAL PERTURBATION —
> NOT CAUSALLY VALIDATED.**

## 12. Deviations from the locked plan

- The separate matched pre-event/transition-state analysis in plan §9 was **not**
  run as an independent matched analysis. The prospective horizon benchmark is the
  implemented transition test; a matched pre-event comparison remains future work.
- The temporal control was implemented as a causal 211-frame delay, not a circular
  shift. The circular version used future rows at the start and was replaced after
  the causality audit; the corrected control is not a clean null.
- Hyperparameter selection was fixed at C=0.1 as preregistered; no inner search was
  run. Sensitivity at C=0.03/0.3 is reported but is not a model-selection result.
- RQ5 is a one-step cross-animal ridge/logistic probe, not a multi-step generative
  simulator generalisation test.
- No new animals, optogenetic stimulation, or causal intervention were performed.
