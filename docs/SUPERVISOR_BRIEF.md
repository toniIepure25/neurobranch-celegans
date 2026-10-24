# Supervisor Brief

**To:** Denis Kovačević (Neuroinformatics, University of Vienna)
**From:** NeuroBranch project
**Date:** 2026-10-24
**Status:** Phase 4 complete: independent replication and turn-direction
generalization both negative; no causal validation

---

## 1. What was built

A reproducible Python platform for *C. elegans* whole-brain calcium imaging:

- real-data loader and audit for the five `WT_NoStim` worms (15,662 frames, 109–135 traces each, 31–47 named);
- leakage-safe blocked and strict-forward splits, with purging and split-local scaling/PCA;
- baselines and dynamics models: persistence, VAR, MLP, residual MLP, GRU, PCA+MLP, PCA+kNN;
- BunDLe-Net artifact integration and a partial PyTorch reimplementation;
- NC-MCM artifact verification and an independent Python port of the lag-2 Markov test;
- branching analysis with a shuffled null and projection-stability check;
- paired virtual-perturbation engine with cluster bootstrap, OOD flags, and random-neuron controls;
- 7-tab Streamlit/Plotly app, Typer CLI, 43 passing tests, 8 figures, full provenance manifests.

**Phase 2 additions (this brief section 2.5):** corrected branching nulls on all
five worms; intervention specificity audit across 6 models; Randi 2023
optogenetic atlas (110 animals, 3,371 events) with external benchmark; incremental
behaviour benchmark; Phase 2 reports and tests.

**Phase 3 additions:** Baskoylu & Flavell 2026 freely-moving adapter (27 animals,
1,162 reversals, SHA256 manifest, JLD2 identity export); animal-disjoint
prospective benchmark with causal features and animal bootstrap; frozen
representation and negative-control comparison; Randi target/shape/context audit;
cross-animal one-step dynamics probe; intervention-limitations checklist; 8-tab
app (7 Phase 2 tabs preserved); 58 passing tests at the Phase 3 checkpoint.

**Phase 4 additions:** Kramer et al. (2026) directed-turning adapter (73
recordings, 107,131 frames, 2,670 reversals, SHA256-verified); independent 6 s
replication on 32 WT octanol animals; animal-disjoint dorsal/ventral benchmark
(56 animals, 945 events); dynamics/PCA/peak-time analysis; shifted-neural,
identity, label-permutation and matched-ablation controls; nonlinear
sensitivity; 9-tab app; Phase 4 manifest and figures.

Code: `D:\ComputaCenter\NeuroBranch\neurobranch-celegans`.
Primary commands: `neurobranch data audit`, `train dynamics`, `reproduce ncmcm`, `reproduce bundlenet`, `branching analyze`, `simulate intervention`, `report`, `app`.

---

## 2. Main results

### 2.1 No dynamics model clearly beats persistence

105 runs (5 worms × 3 seeds × 7 models), blocked middle-block holdout, 12-frame purge:

| Model | One-step R² | Paired ΔR² vs persistence | Wins/15 |
|---|---:|---:|---:|
| residual MLP | 0.372 | +0.011 [−0.008, +0.030] | 7 |
| VAR | 0.366 | +0.004 [−0.069, +0.077] | 6 |
| persistence | 0.361 | — | — |
| latent MLP | 0.251 | −0.110 | 3 |
| latent kNN | 0.250 | −0.111 | 3 |
| GRU | 0.214 | −0.148 | 5 |
| MLP | 0.210 | −0.152 | 3 |

The residual advantage is within seed noise. Behaviour accuracy is 0.80–0.83
versus **0.971** for repeating the current label.

Under a strict forward split (train first 70%, test final 15%), direct MLP/GRU/
latent models collapse on worms 0–1 (R² −1.0 to −4.3), while the residual model
merely tracks persistence (+0.009). This is a non-stationarity problem, not a
tuning success.

### 2.2 Apparent 3-D branching is not excess branching

Worm 0, 328 held-out REV frames, 3-D PCA, k=40, 20-frame horizon:

- observed mean local outcome entropy: **0.578 bits**
- iid shuffled null: **0.961 bits**
- block-10 null: **0.905 bits**
- excess: **−0.383 bits**, one-sided excess p = 1.0, **deficit p = 0.0050**

The Phase 1 p=1.0 was a one-sided excess test. The corrected two-sided audit
shows a significant **deficit**, not excess: local outcomes are *more*
predictable than chance. This is not branching. The deficit replicates in 4/5
worms under iid/block-10 nulls (p≈0.005), but not in worm 3 (block-10 p=0.68),
and is weaker under the circular-shift null (worms 0/4 p<0.05; worm 1 p=0.08;
worms 2/3 not significant). A 3-D branch is not a bifurcation.

### 2.3 Virtual perturbation effects are nonspecific

Worm 0 MLP, +2 SD, 12-frame horizon, 30 held-out REV states, 3 seeds:

- top candidate RIVL: ΔP(TURN)=+0.333, CI [0.167, 0.533], OOD 10.6%;
- RIVR: +0.233; AIBR/DB01/AVEL: +0.167.

Corrected random controls (19 non-candidate named cells, matched dose/timing):
mean |Δ|=0.194, max |Δ|=0.667 (OLQVL), 4/19 ≥ RIVL, empirical p≈0.21. The top
candidate is at the 79th percentile of arbitrary cells. The first control version
sampled candidates by mistake; it was fixed and re-run. The corrected result is a
strong negative result about specificity.

**Phase 2 extension.** Across 6 models (MLP seeds 17/23/31, residual MLP, VAR,
GRU), RIVL +2 is never significant against 19 matched controls (p=0.10–1.00). In
MLP 23 the effect is 0.733 but the control max is 0.767; in the residual MLP it is
exactly 0. The effect is threshold-like: RIVL is 0.011 at +1, 0.333 at +2, 0.667
at +3, and negative at −1/−2. `OLQVL` (0.667) and `OLQVR` (0.633) exceed RIVL at
+2. This is an off-support stress response, not a specific cell effect.

### 2.5 Phase 2 external data

- **Randi 2023:** archive verified against OSF metadata (SHA256
  `d6e7b3…6ce836`); 110 animals, 3,371 valid events, 259 stimulated / 305
  recorded cells. Repeated-stimulation reliability median r=0.090.
- Held-out events: connectome ridge MAE 6.1112 vs global mean 6.1113, paired
  Δ=−0.000086, 95% CI [−0.0177, +0.0165], 50.2% bootstrap wins. Held-out targets:
  5.7467 vs 5.7251. Held-out animals: 5.7919 vs 5.7249. **No model beats the
  constant mean.**
- **Behaviour increment:** state+neural is worse on balanced accuracy and every
  proper scoring rule (blocked Δlog loss +0.011, forward +0.218), while transition
  accuracy rises. No incremental calibrated value.

### 2.6 Phase 3 independent behaviour

Baskoylu & Flavell 2026: 27 animals, 43,200 frames, 1,162 reversals. Primary test:
animal-disjoint reversal onset in `(t, t+6 s]`, original traces, causal
normalisation, L2 logistic C=0.1, animal bootstrap.

| Model | Mean log loss | Δ vs behaviour | 95% CI | Animals improved |
|---|---:|---:|---|---:|
| Behaviour history | 0.5927 | — | — | — |
| + population neural | 0.5856 | −0.00701 | [−0.01483, +0.00053] | 20/27 |
| + train-only PCA(10) | 0.5842 | −0.00841 | [−0.01475, −0.00220] | 20/27 |
| + identified 52 neurons | 0.5909 | −0.00172 | [−0.01054, +0.00699] | 13/27 |
| Causal delayed neural (211 frames) | 0.5892 | −0.00345 | [−0.00851, +0.00175] | 18/27 |

The primary criterion (CI upper < 0, ≥70% animals) is **not met**. The direction
is stable at 3/6/12 s, but the increment shrinks against a quadratic behavioural
state (Δ=−0.00524 [−0.01334, +0.00301]), is absent for identity-aligned neurons,
and is negative within-animal (+0.00288, 9/27 improved). Calibration is good
(slope 0.963 for the neural model), so the null is not a probability-scale
artifact. The causal delayed control also improves slightly (−0.00345, CI includes
zero); delayed activity still carries slow state information, so it is not a clean
null. This is an honest inconclusive/negative result.

Randi Phase 3 target/shape audit: peak, mean and integral all have repeated
r≈0.09; per-neuron shape r=0.048; repeated-event averaging is worse than a global
mean (6.3238 vs 5.9008 MAE). Context reduces MAE by 1.8% but the CI overlaps and
the large RMSE/Pearson gain is response-scale driven. Cross-animal one-step
dynamics: behaviour ridge beats persistence (0.02587 vs 0.02722 MAE), adding
neural does not (0.02590; state log loss 0.11631 → 0.11655). No cross-animal
generative simulator is claimed.

Full report: [`PHASE3_SUPERVISOR_BRIEF.md`](PHASE3_SUPERVISOR_BRIEF.md).

### 2.7 Phase 4 turn-direction generalization

The independent Phase 3 replication on 32 WT octanol animals gives 6 s
behaviour 0.5452 and behaviour+neural 0.5395 log loss: Δ **−0.00570**
([−0.01133, +0.00047]), **22/32** animals improved. The prespecified criterion
is **not met**. The 3 s and 12 s horizons are also null. This is a
non-replication of a small, uncertain effect.

The turn-direction benchmark uses a four-frame mid-reversal segment ending
≥2 frames before the offset, a behaviour/context baseline, leave-one-animal-out
fitting, and animal-cluster bootstrap:

| Mode | AUROC | Δ vs behaviour | 95% CI |
|---|---:|---:|---|
| Behaviour | 0.595 | — | — |
| + population | 0.608 | +0.0123 | [−0.0166, +0.0425] |
| + SAAV | 0.591 | −0.0042 | [−0.0117, +0.0029] |
| + circuit | 0.586 | −0.0097 | [−0.0380, +0.0191] |
| + identified | 0.577 | −0.0176 | [−0.0991, +0.0574] |

No bounded neural increment. Per-animal change is broad (25/50 improve, 20/50
worsen, 5/50 tie), calibration is poor, nonlinear boosting also gives a
null-compatible increment, and controls do not support an identity-resolved
code. A WT diacetyl population point estimate (+0.0624, [−0.0040, +0.1241]) is
exploratory, secondary, includes zero, and has a near-chance behaviour
baseline.

No trial-level optogenetic recordings exist in the archive. tdc1 is treated as
a distribution-shift cohort, not an intervention. The defensible conclusion is
an absence of evidence for incremental prospective turn-direction
decodability, not evidence that the circuit is uninvolved. Full report:
[`PHASE4_SUPERVISOR_BRIEF.md`](PHASE4_SUPERVISOR_BRIEF.md).

### 2.4 Reproduction status

- **NC-MCM:** distributed p at K=3: 0.636, 0.157, 0.096, 0.603, 0.979; our Python port (1000 surrogates): 0.659, 0.149, 0.084, 0.626, 0.983. Agreement within 0.023. We verified artifacts and ported the test; we did not retrain the full pipeline.
- **BunDLe-Net:** official 3-D embeddings give kNN 0.949–0.984 and silhouette 0.446–0.565, but the encoder was fitted on all frames, so this is **embedding-level leakage**, not held-out generalization. Our blocked-holdout reimplementation gives kNN 0.407, macro F1 0.304, silhouette −0.064, trustworthiness 0.757. An earlier run scored kNN 0.624 only because it applied the upstream **zero-phase** filter before splitting, which let future test frames into training; after replacing it with a causal filter the held-out geometry is not behaviour-separated. This is a partial reproduction with documented deviations and a documented leakage correction.

---

## 3. What this means

The project does **not** support the claim that these models recover causal
population dynamics from Kato's data. It supports a narrower, useful claim:

> Under a leakage-audited blocked evaluation on five worms, standard latent and
> neural dynamics models do not beat persistence; local outcome diversity is
> *below* a geometry-preserving null (predictability, not branching); model-ranked
> perturbations are not distinguishable from arbitrary-cell perturbations; and
> pre-stimulus state does not predict external optogenetic responses better than
> a constant. These are benchmarks that a stronger method must beat before causal
> language is warranted.

The value for the collaboration is the negative benchmark, the artifact
verification, and the explicitly falsifiable next experiment — not a new
attractive 3-D plot.

---

## 4. Questions for Denis Kovačević

1. **Primary split.** Is the blocked middle-block interpolation benchmark the
   right primary comparison for your group, or should we invest in time-varying /
   switching models under strict forward evaluation? Our current negative result
   is conditional on a stationary model class.
2. **NC-MCM p-values.** Our port differs by up to 0.023 at K=3. Is that
   consistent with k-means restart/surrogate stochasticity, or does it indicate a
   mismatch in the lag-2 construction? Could we obtain the exact restart seeds and
   surrogate draws used for the distributed artifacts?
3. **BunDLe-Net evaluation.** We infer from the released `main.py` that the model
   is trained on all windows and that the Keras `Normalization` layer is
   unadapted. Is that the intended evaluation, or is there a held-out protocol we
   missed? If the encoder saw all frames, we should not quote its kNN accuracy as
   generalization.
4. **Causal claim.** How does your group define the "causal consistency" of
   NC-MCM? Is it the Markovianity test, the behaviour-decoding consistency, or an
   interventional criterion? We want to represent it accurately rather than
   overstate it.
5. **RIVL/RIVR experiment.** Is a cell-specific optogenetic test of RIVL/RIVR
   during reversal already feasible in your lab, or is it already known? If yes,
   would a preregistered test of the model's directional prediction be worth
   running? If not, which cell pair would you prioritize?
6. **Data access.** Do you have a held-out worm/lab dataset or the `WT_Stim` /
   `AVA_HisCl` files staged for a replication? We deliberately did not claim
   validation from AVA-HisCl.
7. **Compute and tuning.** We used a fixed 60-epoch budget and no per-worm tuning.
   Would you prefer an equal-compute hyperparameter sweep on validation, or is the
   fixed-budget comparison more informative for the collaboration?
8. **Framing.** Do you see this as (a) a methods/benchmark paper with negative
   results, (b) an internal audit before a larger grant, or (c) a stepping stone
   to the biological experiment? The writing and the next 3 months depend on it.

---

## 5. Recommended next step

The Phase 4 replication is the independent test that Phase 3 called for. It
does **not** meet the criterion, and the turn-direction target does not rescue
the signal. The priority is now an exact reproduction of the published RNN on
its intended inputs (if the derived CSVs can be obtained), a balanced
animal-disjoint replication, and trial-level perturbation data — not another
virtual scan or a larger 6-ECTS causal program. The current RIVL/RIVR experiment remains worth running
only with a pre-registered cell set, dose–response range, sham controls, and a new
cohort; the Phase 2 specificity audit and the Phase 3 negative prospective result
remove the original justification for a single-cell virtual test.

See [`PHASE4_NEXT_EXPERIMENTS.md`](PHASE4_NEXT_EXPERIMENTS.md) for falsifiable
next steps and the 6-ECTS scope.  The RIVL/RIVR experiment remains worth running
only with a pre-registered cell set, dose-response range, sham controls and a
new cohort.

These are designed to fail informatively. They do not require another
attractive 3-D visualization.
