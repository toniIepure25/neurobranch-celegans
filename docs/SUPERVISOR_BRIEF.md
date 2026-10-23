# Supervisor Brief

**To:** Denis Kovačević (Neuroinformatics, University of Vienna)
**From:** NeuroBranch project
**Date:** 2026-10-23
**Status:** working prototype on public Kato et al. (2015) data; no new animals, no causal validation

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
- 6-tab Streamlit/Plotly app, Typer CLI, 23 passing tests, 8 figures, full provenance manifests.

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
- shuffled null: **0.960 bits**
- excess: **−0.382 bits**, permutation p = **1.0**

The projection is moderately stable across dimensions (Spearman 0.71–0.77), but
the observed neighbourhoods are *less* diverse than chance. A 3-D branch is not a
bifurcation.

### 2.3 Virtual perturbation effects are nonspecific

Worm 0 MLP, +2 SD, 12-frame horizon, 30 held-out REV states, 3 seeds:

- top candidate RIVL: ΔP(TURN)=+0.333, CI [0.167, 0.533], OOD 10.6%;
- RIVR: +0.233; AIBR/DB01/AVEL: +0.167.

Corrected random controls (19 non-candidate named cells, matched dose/timing):
mean |Δ|=0.194, max |Δ|=0.667 (OLQVL), 4/19 ≥ RIVL, empirical p≈0.21. The top
candidate is at the 79th percentile of arbitrary cells. The first control version
sampled candidates by mistake; it was fixed and re-run. The corrected result is a
strong negative result about specificity.

### 2.4 Reproduction status

- **NC-MCM:** distributed p at K=3: 0.636, 0.157, 0.096, 0.603, 0.979; our Python port (1000 surrogates): 0.659, 0.149, 0.084, 0.626, 0.983. Agreement within 0.023. We verified artifacts and ported the test; we did not retrain the full pipeline.
- **BunDLe-Net:** official 3-D embeddings give kNN 0.949–0.984 and silhouette 0.446–0.565, but the encoder was fitted on all frames, so this is **embedding-level leakage**, not held-out generalization. Our blocked-holdout reimplementation gives kNN 0.407, macro F1 0.304, silhouette −0.064, trustworthiness 0.757. An earlier run scored kNN 0.624 only because it applied the upstream **zero-phase** filter before splitting, which let future test frames into training; after replacing it with a causal filter the held-out geometry is not behaviour-separated. This is a partial reproduction with documented deviations and a documented leakage correction.

---

## 3. What this means

The project does **not** support the claim that these models recover causal
population dynamics from Kato's data. It supports a narrower, useful claim:

> Under a leakage-audited blocked evaluation on five worms, standard latent and
> neural dynamics models do not beat persistence; apparent branching is below a
> shuffled null; and model-ranked perturbations are not distinguishable from
> arbitrary-cell perturbations. These are benchmarks that a stronger method must
> beat before causal language is warranted.

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

Run the preregistered RIVL/RIVR optogenetic experiment in
[`NEXT_EXPERIMENTS.md`](NEXT_EXPERIMENTS.md), with sham and random-neuron
controls, and pre-register the model's directional prediction. In parallel, run a
time-varying model class under the strict-forward split to test whether the
negative dynamics result is caused by non-stationarity.

Both are designed to fail informatively. Neither requires another attractive 3-D
visualization.
