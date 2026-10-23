# Results

All numbers below are recomputed from saved artifacts in this repository. The
commands that produce them are in [`EXPERIMENT_REGISTRY.md`](EXPERIMENT_REGISTRY.md).
Nothing here is a causal claim. The intervention section is explicitly
**model-dependent and not causally validated**.

> **Headline.** Under an honest blocked split, no neural dynamics model clearly
> beats persistence; under a strict forward split, most nonlinear models collapse.
> Behaviour decoding is well below label persistence. Apparent 3-D branching is
> *less* diverse than a shuffled null. The top named-neuron virtual effect is
> *within the range* of arbitrary non-candidate neurons (4/19 controls match or
> exceed it, empirical p≈0.21), so it is not specific. Phase 3 repeats the
> behavioural test on 27 independent freely-moving animals: the primary neural
> increment has the right sign but a 95% CI that includes zero, and it does not
> survive a quadratic behavioural baseline or identity-aligned neurons. These are
> negative/inconclusive results and they are the main scientific output.

---

## 1. Dataset

| Worm | Frames | Seconds | Segmented traces | Named cells | Predictors after exclusions |
|---:|---:|---:|---:|---:|---:|
| 0 | 3137 | 1080.0 | 109 | 38 | 103 |
| 1 | 3134 | 1080.0 | 135 | 44 | 127 |
| 2 | 3059 | 1080.0 | 131 | 32 | 123 |
| 3 | 3311 | 1080.0 | 125 | 47 | 118 |
| 4 | 3021 | 1080.0 | 129 | 31 | 121 |

- Total: 15,662 frames, 629 segmented traces, 192 named traces.
- Effective sampling rate: 2.90–2.91 Hz.
- `WT_NoStim.mat`, SHA256 `7531811c1718e26a0455f59cb2f8188cf25809278cfea85770dfe03315a0098a`.
- Behaviour labels are not independent: see [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md).

Behaviour composition is strongly worm-specific. Worm 0 is roughly balanced
FWD/REV; worm 2 and worm 4 are dominated by REV and TURN in different proportions;
late blocks in the strict-forward split are close to single-state. Figure
`figures/fig01_dataset_coverage.png`.

---

## 2. Primary benchmark: blocked interpolation

**Protocol.** Five contiguous blocks per worm; test = middle block (block 2),
validation = block 4, train = blocks 0,1,3; 12-frame purge. 5 worms × 3 seeds
(17,23,31) × 7 models = **105 completed runs, 0 failures**. This is interpolation
in time, not forecasting.

### 2.1 One-step neural prediction

| Model | Mean one-step R² | SD across 15 runs | Min–max run |
|---|---:|---:|---:|
| `mlp_residual` | **0.3723** | 0.1523 | 0.178–0.596 |
| `var` | 0.3655 | 0.0465 | 0.307–0.441 |
| `persistence` | 0.3613 | 0.1708 | 0.135–0.593 |
| `latent_mlp` | 0.2509 | 0.0622 | 0.196–0.366 |
| `latent_knn` | 0.2500 | 0.0658 | 0.188–0.369 |
| `gru` | 0.2135 | 0.0682 | 0.111–0.339 |
| `mlp` | 0.2095 | 0.0667 | 0.135–0.338 |

Paired against persistence on the same worm and seed:

| Model | Mean paired ΔR² | 95% CI (t, 14 df) | Wins / 15 |
|---|---:|---:|---:|
| `mlp_residual` | +0.0110 | [−0.0082, +0.0301] | 7 |
| `var` | +0.0042 | [−0.0687, +0.0770] | 6 |
| `latent_mlp` | −0.1104 | [−0.1801, −0.0406] | 3 |
| `latent_knn` | −0.1113 | [−0.1789, −0.0437] | 3 |
| `gru` | −0.1478 | [−0.2325, −0.0632] | 5 |
| `mlp` | −0.1518 | [−0.2368, −0.0669] | 3 |

The residual MLP's +0.011 is not distinguishable from zero and wins only 7/15
pairs. The VAR point estimate is essentially identical to persistence. Every
other learned model is **significantly worse** than persistence. There is no
evidence that nonlinear population dynamics add predictive value at one step.

### 2.2 Behaviour decoding

| Model | Mean behaviour accuracy | Label-persistence accuracy |
|---|---:|---:|
| `mlp` | 0.8324 | 0.9713 |
| `latent_mlp` | 0.8235 | 0.9713 |
| `latent_knn` | 0.8153 | 0.9713 |
| `var` | 0.8053 | 0.9713 |
| `persistence` (probe) | 0.8053 | 0.9713 |
| `mlp_residual` | 0.8000 | 0.9713 |
| `gru` | 0.8130 | 0.9708 |

Every model is **~14–17 percentage points worse** than simply repeating the
current label. This is expected given strong behavioural autocorrelation, but it
means the decoding result cannot be presented as evidence of a useful neural
readout. The behaviour predictor is trained with a small auxiliary loss
(`behavior_weight=0.2`) and is not the main training objective; a dedicated
decoder could close some of the gap, but that would not change the conclusion
about dynamics.

### 2.3 Transition distributions

| Model | Transition TV (lower is better) |
|---|---:|
| `var` | **0.1717** |
| `mlp` | 0.1833 |
| `persistence` | 0.2167 |
| `gru` | 0.2233 |
| `latent_knn` | 0.2836 |
| `mlp_residual` | 0.3242 |
| `latent_mlp` | 0.3267 |

TV is computed at a 20-frame horizon over held-out reversal starts. VAR and the
plain MLP produce transition statistics closer to the observed distribution than
persistence, while the residual model is worse. This is a distributional
statistic, not evidence of correct dynamics: a model can match marginals while
failing trajectories (see rollout MSE below).

### 2.4 Open-loop rollouts

Mean rollout MSE (standardised units):

| Model | h=1 | h=3 | h=5 | h=10 |
|---|---:|---:|---:|---:|
| `persistence` | — | **0.488** | — | **0.664** |
| `var` | — | 0.492 | — | 0.590 |
| `mlp_residual` | — | 0.532 | — | 1.289 |
| `latent_mlp` | — | 0.566 | — | 0.822 |
| `gru` | — | 0.619 | — | 0.784 |
| `mlp` | — | 0.671 | — | 0.935 |
| `latent_knn` | — | 0.672 | — | 0.860 |

Persistence is best at h=3 and second-best at h=10 (VAR is better at h=10). The
residual model degrades sharply beyond one step. This is the classic signature
of a model that has learned a local update, not a stable flow.

**Figure:** `figures/fig02_model_comparison.png`.

---

## 3. Strict-forward negative control

**Protocol.** Train on the first 70%, validate on the next 15%, test on the final
15%, 12-frame purge, seed 17, 5 worms × 7 models = 35 runs. This is the honest
forecast regime; it is included to show what the blocked benchmark hides.

One-step R² by worm:

| Model | worm 0 | worm 1 | worm 2 | worm 3 | worm 4 | mean |
|---|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.396 | 0.319 | 0.721 | 0.500 | 0.055 | 0.398 |
| `mlp_residual` | 0.378 | 0.304 | 0.724 | 0.505 | 0.123 | **0.407** |
| `var` | −0.671 | −2.808 | 0.121 | −0.032 | −0.127 | −0.703 |
| `mlp` | −1.308 | −3.673 | −0.085 | −0.224 | −0.264 | −1.111 |
| `gru` | −1.393 | −4.345 | −0.128 | −0.144 | −0.282 | −1.258 |
| `latent_mlp` | −1.051 | −3.120 | −0.046 | −0.106 | −0.180 | −0.901 |
| `latent_knn` | −1.040 | −3.110 | −0.046 | −0.106 | −0.164 | −0.893 |

The residual MLP again tracks persistence almost exactly (+0.009 mean) and is not
a useful forecast. Direct nonlinear models and latent models fail catastrophically
on worms 0 and 1, where the final block is far outside the training distribution.
Worm 1 behaviour accuracy is **0.123** for the plain MLP (label persistence
0.982); the model predicts the wrong late-recording regime.

**Figure:** `figures/fig03_forward_stress.png`.

This is a **negative result about non-stationarity**, not a tuning failure. It is
the strongest argument against presenting the blocked benchmark as forecasting.

---

## 4. BunDLe-Net

### 4.1 Official 3-D artifacts (embedding-level leakage)

The distributed `bundlenet_consistent_embedding_worm_*.npz` files were fitted by
the upstream code, which trains on all windows. Our kNN classifier is fit on
training-block labels and scored on test-block labels, but the **test embeddings
were seen by the encoder during upstream training**. This is embedding-level
leakage, not held-out generalization. The silhouette is computed on all points
and is therefore fully in-sample.

| Worm | kNN accuracy (3-D, leaky) | Silhouette (8 fine states, in-sample) | PCA-3 silhouette (full-data baseline) |
|---:|---:|---:|---:|
| 0 | 0.9841 | 0.486 | −0.008 |
| 1 | 0.9809 | 0.547 | 0.012 |
| 2 | 0.9739 | 0.536 | 0.104 |
| 3 | 0.9789 | 0.565 | 0.063 |
| 4 | 0.9487 | 0.447 | 0.030 |

The geometry is strongly behaviour-separated *on the data used to fit it*. The
comparison with PCA-3 shows the learned embedding is not merely a linear
projection. It says nothing about held-out predictive value, and the kNN number
cannot be quoted as a generalization result.

**Figure:** `figures/fig04_bundlenet_official_3d.png` and
`figures/bundlenet_latent_3d.html`.

### 4.2 PyTorch reimplementation (`BunDLeLite`)

Trained on worm 0 with a blocked holdout, 15-frame windows, 3-D latent,
Gaussian noise 0.05, `γ=0.9`, early stopping on validation, 120 epochs maximum.

| Metric | Value |
|---|---:|
| kNN accuracy (held-out windows) | 0.407 |
| Macro F1 | 0.304 |
| Silhouette (8 fine states) | −0.064 |
| Trustworthiness (high-dim → 3-D) | 0.757 |
| Best epoch | 16 |
| Best validation loss | 0.178 |
| Train / test windows | 1853 / 612 |

**A filter-leakage correction changed this result.** The first version of the
reimplementation applied the upstream **zero-phase** `filtfilt` to the whole
recording before splitting. That is acausal: future test frames entered earlier
training inputs. With that filter the same architecture and training budget scored
kNN 0.624, macro F1 0.474, silhouette 0.145. Replacing it with a one-pass causal Butterworth filter (same
0.05 normalized cutoff) drops kNN to 0.407 and silhouette below zero. The higher
number was leakage, not representation quality. Both numbers are reported here to
make the failure mode auditable.

This is a **partial, not faithful, reproduction**. Known deviations:

1. Upstream trains on all windows; we hold out a blocked test set, so our number
   is lower by construction.
2. Upstream Keras `Normalization` is unadapted (identity) in the released code;
   we reproduce that rather than "fixing" it silently.
3. Upstream uses a zero-phase 4th-order low-pass with effective cutoff ≈0.0725 Hz.
   We use a **causal one-pass** Butterworth filter at the same cutoff; the upstream
   zero-phase version is acausal and leaked future held-out frames into training
   inputs. The causal filter introduces a phase delay, which is a real deviation.
4. Upstream's 7-fold cross-validation is replaced by a single blocked holdout for
   comparability with the dynamics suite.
5. No commutativity/consistency loss term is included in `BunDLeLite`; the
   released training objective is approximated.

The reimplementation is therefore an **integration and stress test**, not a claim
that we reproduced BunDLe-Net's published performance. Its held-out 3-D geometry
is not behaviour-separated (silhouette −0.064); the failure is informative: most
of the apparent separation in the earlier run was attributable to acausal
filtering, not to the learned transition model.

---

## 5. NC-MCM verification

We did **not** retrain NC-MCM from raw data. We loaded the five distributed
`nc_mcm_model_worm_*.npz` artifacts, reimplemented the lag-2 surrogate
Markovianity test in Python, and compared at the paper's selected `K=3`.

| Worm | K | p (distributed) | p (Python port) | Decoder accuracy | Balanced accuracy |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 0.636 | 0.659 | 0.764 | 0.620 |
| 2 | 3 | 0.157 | 0.149 | 0.683 | 0.482 |
| 3 | 3 | 0.096 | 0.084 | 0.792 | 0.562 |
| 4 | 3 | 0.603 | 0.626 | 0.855 | 0.518 |
| 5 | 3 | 0.979 | 0.983 | 0.788 | 0.557 |

The port agrees within 0.023 in absolute p (worms 1 and 4; 1000 surrogates).
The port reuses the stored cluster labels, so k-means restarts do not contribute
to the difference; it comes from the stochastic surrogate draws and from any
difference between our sampler and the upstream Julia sampler. Two worms have
`p < 0.16`, i.e. the first-order Markov model is rejected at conventional levels
for those animals. The claim "cognitive states are Markovian" is **not** supported
uniformly across worms, and the distributed p-values themselves show this.

Decoder balanced accuracy is only 0.48–0.62 despite 0.68–0.86 raw accuracy,
reflecting strong class imbalance. A raw accuracy of 0.85 is not evidence of a
useful eight-state readout.

**Figure:** `figures/fig07_ncmcm_verification.png`.

---

## 6. Branching analysis (worm 0)

**Protocol.** 3-D PCA fitted on training frames; 328 held-out REV candidate
frames; `k=40` nearest neighbours excluding ±12 frames; 20-frame future horizon;
Shannon entropy over four macro outcomes.

| Quantity | Value |
|---|---:|
| Candidate frames | 328 |
| Mean local outcome entropy | **0.578 bits** |
| Median entropy | 0.669 bits |
| Mean majority-outcome probability | 0.789 |
| Shuffled-null mean entropy (100 permutations) | 0.960 bits |
| Null SD | 0.009 |
| Excess entropy | **−0.382 bits** |
| Permutation p (one-sided, observed ≥ null) | **1.0** |

The observed neighbourhoods are **far less diverse** than a null that preserves
the candidate geometry and marginal outcome distribution but destroys the
state→future association. There is no evidence of excess branching at these
reversal frames. The apparent branches in a 3-D plot are largely a consequence of
choosing a low-dimensional projection and looking at it.

Projection sensitivity: Spearman correlation of per-frame entropy with the 3-D
result is 0.765 (8-D), 0.707 (16-D), 0.722 (32-D). The ranking is moderately
stable, but the absolute entropy and the identity of top frames are
projection-dependent.

A model-based version of the same analysis predicts 88.75% REV and 8.75% TURN
from the same initial states, i.e. the fitted model is far more deterministic
than the observed neighbourhoods. This is a model-data mismatch, not a
bifurcation.

**Figure:** `figures/fig05_branching_worm0.png`.

**Interpretation.** A branch in a latent trajectory is not a bifurcation, not an
unobserved discrete state, and not evidence of causal structure. The correct
description is: "local outcome diversity in this projection at this horizon,
compared with a shuffled null." That comparison is negative.

---

## 7. Virtual perturbation (worm 0, MLP seed 17)

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

**Protocol.** Additive perturbation, ±2 SD, onset 0, duration = horizon = 12
frames (≈4.1 s), 30 held-out REV initial states, 3 simulation seeds (0,1,2),
common random numbers for baseline and intervention, `noise_std=0.04`. Target:
`ΔP(TURN)`.

### 7.1 Candidate scan

Top rows (13 named candidates × 2 amplitudes; sorted by |Δ|):

| Rank | Candidate | Amplitude | Baseline P(TURN) | Virtual P(TURN) | ΔP | 95% cluster bootstrap CI | OOD fraction |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | RIVL | +2 | 0.133 | 0.467 | **+0.333** | [0.167, 0.533] | 0.106 |
| 2 | RIVR | +2 | 0.133 | 0.367 | +0.233 | [0.100, 0.400] | 0.174 |
| 3 | AIBR | +2 | 0.133 | 0.300 | +0.167 | [0.033, 0.333] | 0.134 |
| 4 | DB01 | +2 | 0.133 | 0.300 | +0.167 | [0.033, 0.300] | 0.140 |
| 5 | AVEL | +2 | 0.133 | 0.300 | +0.167 | [0.033, 0.333] | 0.107 |
| 6 | AVBL | +2 | 0.133 | 0.289 | +0.156 | [0.050, 0.289] | 0.098 |
| 7 | RIML | −2 | 0.133 | 0.267 | +0.133 | [0.033, 0.267] | 0.115 |
| 8 | AVEL | −2 | 0.133 | 0.033 | −0.100 | [−0.200, 0.000] | 0.084 |

At face value RIVL looks like a strong candidate. It is not.

### 7.2 Randomly sampled non-candidate controls

Controls use the **same** amplitude (+2 SD), duration, horizon, initial states,
seeds, and OOD definition. They are drawn uniformly without replacement from
named predictor cells **not** in the hypothesis-driven candidate set (`seed=17`,
19 cells). This exclusion is essential: an earlier version sampled from the full
pool including candidates and was therefore not a control at all. That bug was
caught by red-teaming and corrected; the old numbers are not used.

| Statistic | Value |
|---|---:|
| n control cells | 19 |
| Mean ΔP(TURN) | +0.153 |
| Mean absolute ΔP | **0.194** |
| Max ΔP | **+0.667** (OLQVL) |
| Second / third | +0.633 (OLQVR), +0.589 (OLQDR) |
| Empirical one-sided p for RIVL (|Δ| ≥ 0.333) | **0.211** |
| RIVL OOD fraction | 0.106 |
| OLQVL OOD fraction | **0.377** |

The top candidate is at the **79th percentile** of the control distribution and
is exceeded in magnitude by 4/19 arbitrary non-candidate cells. The largest
control effects occur with the largest OOD fractions, i.e. where the model is
extrapolating. The candidate effect is **not specific**.

### 7.3 Interpretation

The scan is useful as a *simulator stress test* and as a generator of hypotheses.
It is not evidence that RIVL promotes turning in a living worm. The correct
statement is:

> "Under this fitted MLP, with these assumptions and this support, an additive
> +2 SD virtual edit to RIVL changed the model's predicted TURN probability by
> +0.333, which is within the range produced by arbitrary non-candidate cells and
> is accompanied by 10.6% out-of-support states."

The strong negative result — **nonspecificity** — is the scientifically important
one. It also motivates the biological experiment in
[`NEXT_EXPERIMENTS.md`](NEXT_EXPERIMENTS.md).

**Figure:** `figures/fig06_intervention_forest.png`.

---

## 8. Phase 2: robustness, specificity, and external data

Phase 2 is additive. It does **not** replace the Phase 1 tables above. Full
protocols and artifacts are in the Phase 2 reports linked at the end of this
section.

### 8.1 Branching: the one-sided null was read backwards

The Phase 1 `p = 1.0` was a one-sided test for **excess** entropy. Corrected
two-sided tests on all five worms show the opposite direction: observed local
entropy is **below** a geometry-preserving null in 4/5 worms (iid/block-10
p ≈ 0.005). That is local predictability, **not branching**.

| Worm | Mean entropy | iid null | block-10 null | p_deficit (iid / block-10) | min p_excess | shift p (two-sided) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.578 | 0.961 | 0.905 | 0.00498 / 0.00498 | 0.985 | 0.0398 |
| 1 | 0.625 | 0.978 | 0.933 | 0.00498 / 0.00498 | 0.965 | 0.0796 |
| 2 | 0.780 | 0.880 | 0.847 | 0.00498 / 0.0448 | 0.821 | 0.368 |
| 3 | 0.889 | 0.924 | 0.860 | 0.0249 / 0.677 | 0.0896 | 0.338 |
| 4 | 0.603 | 0.948 | 0.887 | 0.00498 / 0.00498 | 0.955 | 0.0995 |

No worm shows significant excess. Worm 3 is not significant under the block
null. Synthetic ground-truth fixtures show the test has little power for genuine
stochastic branching and can be fooled by projection width. Detail:
[`BRANCHING_STATISTICAL_AUDIT.md`](BRANCHING_STATISTICAL_AUDIT.md).

### 8.2 Intervention: RIVL is not specific

Phase 1's top effect (RIVL +2, ΔP(TURN) = 0.333 [0.167, 0.533]) was scanned across
6 models, 26 candidate cells and 19 matched non-candidate controls per model.

| Model | RIVL +2 ΔP | Controls ≥ RIVL | Empirical p | Control max |
|---|---:|---:|---:|---:|
| MLP 17 | 0.333 | 4/19 | 0.25 | 0.667 (OLQVL) |
| MLP 23 | 0.733 | 2/19 | 0.15 | 0.767 |
| MLP 31 | 0.267 | 6/19 | 0.35 | 0.567 |
| Residual MLP 17 | 0.000 | 19/19 | 1.00 | 0.000 |
| VAR 17 | 0.544 | 2/19 | 0.15 | 0.544 |
| GRU 17 | 0.367 | 1/19 | 0.10 | 0.422 |

No model survives `p < 0.05`. Amplitude sensitivity shows a threshold at +3 and
sign reversal at negative amplitudes; `OLQVL` and `OLQVR` exceed RIVL at +2. The
effect is an off-support stress response, not a specific causal candidate.
Detail:
[`INTERVENTION_GENERALIZATION_RESULTS.md`](INTERVENTION_GENERALIZATION_RESULTS.md).

### 8.3 Randi 2023 optogenetic atlas: no positive transfer

3,371 valid events, 110 animals, 259 stimulated cells, 305 recorded cells.
Repeated-stimulation response reliability is only median r = 0.090 (925 pairs).
Models fitted on pre-stimulus state, target identity, recorded cell, and
connectome edges were compared with a constant training mean.

| Protocol | Global-mean MAE | Best non-constant | MAE | Δ vs global |
|---|---:|---|---:|---:|
| held-out events | 6.1113 | connectome ridge | 6.1112 | −0.000086 (95% CI [−0.0177, +0.0165]) |
| held-out targets | 5.7251 | connectome ridge | 5.7467 | +0.0216 |
| held-out animals | 5.7249 | neuron mean | 5.7919 | +0.0670 |

The event-level tie is a coin flip (50.2% of bootstrap resamples favor the
connectome model). New targets and new animals are worse than the constant. This
is a negative benchmark result, not evidence that the atlas is uninformative.
Detail: [`INTERVENTION_DATASET_AUDIT.md`](INTERVENTION_DATASET_AUDIT.md).

### 8.4 Behaviour: neural activity does not add calibrated predictive value

Paired `state + neural − state` on five Kato worms:

| Protocol | Δ balanced accuracy | Δ log loss | Δ Brier | Δ transition accuracy |
|---|---:|---:|---:|---:|
| blocked | −0.0147 (0/5 wins) | +0.0114 (2/5) | +0.0126 (0/5) | +0.1789 (5/5) |
| forward | −0.1272 (0/5) | +0.2182 (0/5) | +0.0711 (0/5) | +0.2784 (4/5) |

Neural activity detects that a rare transition will occur, but every proper
scoring rule and balanced metric gets worse. This is not incremental predictive
value. Detail: [`BEHAVIORAL_GENERALIZATION.md`](BEHAVIORAL_GENERALIZATION.md).

### 8.5 Phase 2 verdicts

| Phase 1 statement | Phase 2 verdict | Evidence |
|---|---|---|
| No learned model robustly beats persistence | **Confirmed** | Phase 1 rerun + Phase 2 behaviour audit |
| Apparent branching is below shuffled null | **Confirmed, direction corrected**: significant local-entropy *deficit* in 4/5 worms, not branching | corrected nulls |
| Behaviour decoding below label persistence | **Confirmed**, and it adds no calibrated increment | incremental contrasts |
| Top RIVL virtual effect is nonspecific | **Confirmed and strengthened**: 0/6 models significant; controls larger; amplitude threshold | specificity audit |
| Real optogenetic responses are predictable | **Rejected** under this benchmark | Randi external test |
| A virtual perturbation is causal | **Unresolved and untested**; no biological validation | evidence gate |
| Neural activity improves prospective behaviour on independent animals | **Not supported**: primary CI includes 0; PCA secondary only | Baskoylu Phase 3 |
| A frozen identity representation adds robust value | **Not supported**: identified neurons weaker than population; quadratic control removes significance | Phase 3 representations |
| Randi target/context modelling rescues response prediction | **Targets no; context weak** (1.8% MAE, overlapping CI) | Phase 3 Randi audit |
| Dynamics generalise across recordings without transductive representations | **Behaviour yes; neural no** | cross-animal one-step probe |

### 8.6 Phase 3: independent freely-moving behaviour and frozen representations

Phase 3 adds Baskoylu & Flavell (2026): 27 animals, 1,162 reversals, 43,200
frames. The primary test is animal-disjoint prospective prediction of a reversal
onset in `(t, t+6 s]`, using original traces and causal normalisation.

| Model (H=6 s) | Mean animal log loss | Î” vs behaviour | 95% animal CI | Animals improved |
|---|---:|---:|---|---:|
| Behaviour history | 0.5927 | â€” | â€” | â€” |
| + population neural | 0.5856 | âˆ’0.00701 | [âˆ’0.01483, +0.00053] | 20/27 |
| + train-only PCA(10) | 0.5842 | âˆ’0.00841 | [âˆ’0.01475, âˆ’0.00220] | 20/27 |
| + identified 52 neurons | 0.5909 | âˆ’0.00172 | [âˆ’0.01054, +0.00699] | 13/27 |
| Causal delayed neural (211 frames) | 0.5892 | -0.00345 | [-0.00851, +0.00175] | 18/27 |

The prespecified criterion (CI upper < 0, â‰¥70% animals improved) is **not met**
for the primary model. The direction is consistent at 3/6/12 s, but the increment
shrinks against a quadratic behavioural baseline (6 s Î” = âˆ’0.00524
[âˆ’0.01334, +0.00301]), is absent for identity-aligned neurons, and reverses in
the within-animal forward task (+0.00288, 9/27 improved). The causal delayed
control also improves slightly (âˆ’0.00345), so it is not a clean null and the
small increment may reflect slow behavioural/arousal state. This is
**inconclusive/negative**, not evidence of absence.

Randi target reanalysis: peak/mean/integral reliability all r â‰ˆ 0.09; full
temporal shape per-neuron median r = 0.048; repeated-event averaging is worse
than a global mean for every target (signed peak 6.3238 vs 5.9008 MAE). Stimulus
context lowers MAE by 1.8% but the CI overlaps; the large RMSE/Pearson gain is
response-scale/outlier driven. A cross-animal one-step dynamics probe finds
behaviour ridge beats persistence (MAE 0.02587 vs 0.02722) but adding neural
activity does not (0.02590; next-state log loss 0.11631 â†’ 0.11655).

Phase 3 reproduces the Phase 2 negative **pattern** on independent data. Detail:
[`PHASE3_RESULTS.md`](PHASE3_RESULTS.md).

## 9. What would change our mind

| Claim | Falsifier | Current status |
|---|---|---|
| Neural dynamics add predictive value | Paired ΔR² > 0 with CI excluding 0 on held-out worms | Not supported |
| Behaviour is decodable above persistence | Accuracy > label persistence with CI | Not supported |
| 3-D branching reflects real multi-stability | Excess entropy > 0 vs shuffled null, replicated across worms and projections | Not supported (negative excess) |
| A named cell has a specific virtual effect | Effect exceeds matched random-neuron null and survives OOD stratification | Not supported |
| NC-MCM states are Markovian | p > 0.05 across worms with adequate power | Mixed; two worms reject |
| A perturbation is causal in the animal | Matched real intervention with controls | Not attempted |

See [`VALIDATION_PROTOCOL.md`](VALIDATION_PROTOCOL.md) for the pre-specified
decision rules and their outcomes.
