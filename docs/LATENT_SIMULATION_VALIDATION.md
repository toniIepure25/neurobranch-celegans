# Latent simulation validation

This document separates two things that are often conflated: a **linear latent
projection with an inverse** (PCA) and a **representation/transition model
without a published inverse** (BunDLe-Net). Only the first can support a
latent-space dynamical claim in this repository.

> **Status.** PCA trajectories are invertible by construction. The train-only
> BunDLe encoder is unstable across seeds and is used for visualisation and
> descriptive comparison only. Neither representation establishes that a named
> neuron maps to one latent coordinate.

## 1. Why two representations

| Encoder | Fit | Inverse | Permitted use |
|---|---|---|---|
| PCA (3-D or 16-D) | training frames only | exact linear | latent rollout and decoding |
| BunDLeLite (3-D) | training windows only, blocked split | no published decoder | encoder/visualisation; representation comparison |

The official BunDLe-Net artifacts in
`references/external/nc-mcm/precomputed_results/` were fitted on all frames.
They are shown in the app as **in-sample artifact geometry**, not as held-out
validation. The code path used for this audit trains a fresh PyTorch
`BunDLeLite` with a causal low-pass and a blocked split.

## 2. What was trained

`scripts/simulator_latent.py` fits, per worm:

- `PCAEncoder(n_components=3)` on training frames;
- `BunDLeEncoder` with the upstream 50-30-25-10-3 encoder, 15-frame window,
  gamma 0.9 Markov consistency, behaviour cross-entropy weight 0.1, 80 epochs,
  best validation epoch restored.

Deviations from upstream are explicit in the model metadata: PyTorch rather than
TensorFlow/Keras, causal one-pass low-pass rather than zero-phase `filtfilt`,
blocked split rather than random K-fold, and best-validation rather than a fixed
final epoch. These deviations are necessary to avoid held-out leakage.

The 3-D PCA explains only part of the variance:

| Worm | PCA 3-D explained variance | BunDLe best validation loss | Best epoch | Train windows |
|---|---:|---:|---:|---:|
| 0 | 0.5305 | 0.1778 | 16 | 1853 |
| 1 | 0.4312 | 0.1390 | 8 | 1851 |
| 2 | 0.4201 | 0.1187 | 10 | 1806 |
| 3 | 0.5109 | 0.1432 | 11 | 1957 |
| 4 | 0.3861 | 0.2668 | 2 | 1783 |

A 3-D view is therefore a lossy summary, not the full population state.

## 3. Trajectory comparison

For each worm, 10 held-out test starts and a 20-frame horizon were rolled out
with the selected neuron-space model (`mlp_residual`) and with `latent_knn`.
Real and simulated states were projected through the same train-fitted encoder.
Metrics are averaged over starts:

- **latent energy distance** (lower is closer; scale differs between encoders);
- **path-length ratio** = simulated path length / observed path length;
- **curvature difference** in radians;
- **latent OOD fraction** = fraction of simulated points beyond the 99th
  percentile training nearest-neighbour distance.

| Kind | Encoder | Energy distance | Path-length ratio | Curvature difference | Latent OOD |
|---|---|---:|---:|---:|---:|
| `mlp_residual` | PCA | 1.4289 | 0.1860 | âˆ’1.1823 | 0.4571 |
| `mlp_residual` | BunDLe | 0.0335 | 0.6117 | +0.0514 | 0.2181 |
| `latent_knn` | PCA | 2.3174 | 1.1808 | âˆ’0.3301 | 0.5143 |
| `latent_knn` | BunDLe | 0.0219 | 1.5549 | +0.0127 | 0.2010 |

The selected model's PCA path is **too short** (ratio 0.19) and bends much less
than the observed path (curvature difference âˆ’1.18 rad). The latent kNN model
over-runs the path length (ratio 1.18) but is closer in curvature. These are
descriptive failures, not evidence that either model learned the true latent
manifold.

BunDLe energy distances are numerically smaller because its embedding scale is
different; energy distance is **not comparable across encoders**. The path-length
ratio and curvature difference are the interpretable columns.

## 4. Encoder stability is poor

A representation that changes substantially across training seeds cannot anchor
a scientific claim about a named-neuron perturbation. On 200 held-out frames of
worm 0:

| Seed pair | Linear CKA | Procrustes residual |
|---|---:|---:|
| 17 vs 23 | 0.102 | 1.021 |
| 17 vs 31 | 0.284 | 2.033 |

Both CKA values are far below the 0.9 that would indicate a stable embedding;
the Procrustes residuals are large relative to the centred embedding norm. The
BunDLe encoder is therefore **not** used as a validated latent coordinate system.
This is a negative result about the representation, not a bug in the plotting
code.

## 5. What a named-neuron perturbation means in latent space

A named neuron is one coordinate of `x`. PCA has a linear inverse, so a change in
one neuron maps to a direction in PCA space and back. BunDLe-Net has no published
inverse; a perturbation can be applied in neuron space and then encoded, but the
result is not a unique latent displacement. Any statement such as "AVEL lives on
latent dimension 2" would be an artefact of an arbitrary encoder.

The simulator therefore applies perturbations in **neuron space**, then projects
for display. Latent plots are descriptive summaries of a neuron-space rollout,
not a causal latent intervention.

## 6. Reproduction

```powershell
python scripts/simulator_latent.py --epochs 80 --starts 10 --horizon 20
python scripts/simulator_figures.py
```

Outputs: `reports/simulator/latent_metrics.csv`,
`latent_encoders.csv` (including each local encoder's SHA256),
`latent_stability.json`, local encoder weights under
`reports/simulator/latent_encoders/` (ignored by Git because they are binary),
and `figures/simulator/latent.png` plus `trajectory_pca.png`.

The local weights are provenance artifacts; metadata is committed. To reproduce
them exactly, use the command above and the frozen config. Do not commit the
`.pt` files: `.gitattributes` marks them binary and `.gitignore` excludes them to
prevent line-ending corruption.

## 7. Falsifiable next tests

1. Train BunDLe with multiple seeds and require CKA â‰¥ 0.9 before using it as a
   latent coordinate system.
2. Compare latent rollout error with the neuron-space rollout of the same model,
   not only with a different model.
3. Test whether a linear decoder from latent to the held-out neuron state is
   stable across seeds; if not, report encoder-only.
4. Add a held-out cohort. Within-worm latent geometry is not a population result.

