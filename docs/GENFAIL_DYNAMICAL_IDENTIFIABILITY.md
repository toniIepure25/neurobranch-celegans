# Dynamical identifiability — are the latent states and operators real?

Artifacts: `reports/simulator/failure_diagnostics/identifiability/`
Code: `src/neurobranch/diagnostics/identifiability.py`

## 1. Scope and ambiguity

A latent linear model is identifiable only up to an invertible transformation: rotating, permuting or sign-flipping latent coordinates leaves the observed predictions unchanged. We therefore report invariants:

- PCA participation ratio and variance concentration;
- principal angles between subspaces;
- eigenvalues of the fitted latent operator;
- observability rank and condition number;
- held-out reconstruction.

We do **not** interpret latent axes as anatomical populations, and we do not claim that a stable eigenvalue is a biological time constant.

## 2. Effective dimensionality

Train-only PCA on each session:

| worm | participation ratio | components for 90% variance | top-5 variance fraction |
|---:|---:|---:|---:|
| 0 | 8.76 | 36 | 0.613 |
| 1 | 12.63 | 52 | 0.505 |
| 2 | 13.29 | 61 | 0.470 |
| 3 | 10.05 | 38 | 0.583 |
| 4 | 16.47 | 63 | 0.445 |

The spectrum is broad. A 15-dimensional latent model retains substantial but not dominant variance; 36–63 components are needed for 90%. This is not a clean low-dimensional system in the observed traces, or the observation mixes many latent modes.

## 3. Are the subspaces stable?

Two tests:

1. **Half-split stability:** fit PCA on the first and second halves of the training frames, then compute principal angles.
2. **Bootstrap stability:** resample training frames within the same period.

| worm | half-split mean angle | half-split max angle | mean cos² overlap | bootstrap mean angle |
|---:|---:|---:|---:|---:|
| 0 | 56.2° | 88.4° | 0.352 | 4.45° |
| 1 | 56.0° | 88.4° | 0.355 | 3.41° |
| 2 | 61.7° | 89.9° | 0.283 | 5.11° |
| 3 | 59.3° | 89.1° | 0.304 | 2.83° |
| 4 | 60.4° | 89.9° | 0.306 | 5.77° |

The contrast is decisive. Within a fixed period, bootstrap subspaces agree closely (3–6°). Across the two halves of the same recording, the mean angle is 56–62° and the worst direction is nearly orthogonal (88–90°). The latent subspace is **not stable over time**, even though it is numerically stable under resampling.

This is evidence for non-stationarity or regime change, not merely finite-sample noise. It also explains why a model fitted on the early recording can fail on the late test tail.

## 4. Operator spectrum

For a 15-dimensional latent VAR(1), fitted on training data:

| worm | observability rank | observability condition | realized spectral radius | eigenvalue magnitude range |
|---:|---:|---:|---:|---|
| 0 | 15/15 | 4.03 | 0.9942 | 0.547–0.994 |
| 1 | 15/15 | 3.84 | 0.9923 | 0.556–0.992 |
| 2 | 15/15 | 3.61 | 0.9932 | 0.550–0.993 |
| 3 | 15/15 | 3.43 | 0.9905 | 0.596–0.993 |
| 4 | 15/15 | 4.61 | 0.9799 | 0.019–0.982 |

Observability is full rank with modest condition number: the fitted latent is observable in the linear systems sense. Most eigenvalues are close to the unit circle (0.98–0.99), i.e. slow modes. Worm 4 has one nearly zero mode, which is effectively unused.

Full observability does **not** imply that the operator generalises. The operator is observable but time-unstable.

## 5. Operator stability under resampling

Subsampling 80% of training starts and refitting gives:

| worm | eigenvalue magnitude MAE | spectral-radius difference |
|---:|---:|---:|
| 0 | 0.003–0.008 | 0.00006–0.002 |
| 1 | 0.004 | 0.00008–0.003 |
| 2 | 0.004–0.006 | 0.00006–0.002 |
| 3 | 0.003–0.004 | 0.0007–0.003 |
| 4 | 0.005–0.008 | 0.0005–0.002 |

The eigenvalue magnitudes are stable under resampling. Combined with the unstable half-split subspaces, this points to **stable within-period dynamics but a changing latent basis/regime**, not a random numerical fit.

## 6. Reconstruction versus forecast

Test reconstruction R² is negative for every session:

| worm | test reconstruction R² |
|---:|---:|
| 0 | −0.938 |
| 1 | −3.119 |
| 2 | −0.012 |
| 3 | −0.102 |
| 4 | −0.070 |

A negative R² means the latent model reconstructs held-out observations worse than the test mean. This is a distribution-shift/mean-shift result, not merely forecast error. It reinforces the half-split finding: the late test tail is not drawn from the same observation distribution as the training period.

## 7. What is identifiable, and what is not

**Supported:**

- A full-rank, moderately conditioned 15-dimensional linear latent state is observable within a period.
- The eigenvalue magnitudes are stable under resampling.
- Synthetic known-truth controls recover latent structure well (R² 0.77–0.99).

**Not supported:**

- A single time-invariant latent subspace across the recording.
- Anatomical identity of latent axes.
- A unique operator: rotation/permutation ambiguity remains, and the time-half instability is far larger than resampling noise.
- The claim that a stable spectral radius implies a valid generative simulator.

Stage verdict: `IDENTIFIABILITY_ASSESSED` — PASS for the assessment; the latent state is locally observable but **not globally/time-invariantly identifiable**. This is a substantive negative result.
