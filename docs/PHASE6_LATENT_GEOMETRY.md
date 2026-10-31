# Phase 6 latent geometry

No new latent generative model was fitted. This note records why the existing latent evidence is insufficient and what would be needed before latent-space perturbation claims.

## Existing evidence (Phase 5, unchanged)

- PCA encoders are used for visualization and for some latent baselines.
- Cross-seed linear CKA for the selected residual MLP is 0.102 (seeds 17 vs 23) and 0.284 (17 vs 31) on worm 0.
- Procrustes errors are 1.02 and 2.03.
- The PCA latent trajectories have large latent OOD fractions; BunDLe-Net is an encoder without a published inverse decoder.

The cross-seed CKA values are far below the level needed to treat a latent axis as reproducible, so no latent-space figure is presented as evidence.

## Consequences

1. A latent coordinate cannot be interpreted as a named neuron or as a stimulation target.
2. Low CKA means two fits of the same architecture learn different geometries; a "latent intervention" would not be reproducible across seeds.
3. Decoding a perturbed latent state through PCA is a linear projection, not a biological inverse.

## Why this was not expanded

The primary model comparison is in observed neural state space. Adding a latent generative model without a validated decoder would create an attractive visualization but no testable perturbation. It would also increase the number of models selected on the same five worms.

## Next step

A latent model becomes defensible only with:

- a held-out decoder with per-neuron error;
- cross-seed CKA above a predeclared threshold;
- support calibration in latent space;
- an independent cohort.

Until then, no latent intervention or latent branching claim is made.
