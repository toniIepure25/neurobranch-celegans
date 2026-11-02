# Calcium observation-model stability

## Diagnosis

The frozen `CausalCalciumLDS` uses a causal high-pass transform, a PCA latent
drive, a lag-one latent operator, and a first-order calcium observation. Its
one-step prediction is well behaved. Its autonomous rollout is not: at H20 the
mean MSE reaches 37 to 4.9×10⁴ across real sessions, at H50 up to 3.7×10¹⁷, and
at H100 up to 1.7×10³⁹. The trajectories remain finite in IEEE arithmetic but
are numerically meaningless.

The defect is localized to the rollout, not to the fitted spectral radius. The
operators are already capped:

| Session | alpha | spectral radius | observation condition |
|---:|---:|---:|---:|
| 0 | 0.3 | 0.947 | 4.75 |
| 1 | 0.5 | 0.973 | 4.53 |
| 2 | 0.7 | 0.984 | 9.99 |
| 3 | 0.2 | 0.761 | 9.13 |
| 4 | 0.7 | 0.973 | 6.04 |

The original rollout re-encodes each predicted calcium value as if it were a
new high-pass drive, then feeds that encoding back into the latent recursion.
The high-pass transform has gain `(1-alpha)/alpha` (up to 4.0 here), the PCA
inverse map adds another amplification, and the loop compounds both. The
stability of `A` alone cannot protect against this observation-filter feedback.
This is a **correctable implementation defect**, not evidence that the latent
dynamics are intrinsically explosive.

## Corrected model

`causal_calcium_lds_corrected_v2` (artifact version
`causal_calcium_lds_corrected_v1`) keeps the same fitted PCA, `A`, `b`, and
alpha, but separates the recursions:

```
z_{t+1} = A z_t + b
c_{t+1} = (1 - alpha) c_t + alpha (D z_{t+1} + mean)
```

The latent drive is never re-encoded from predicted calcium. `A` is uniformly
scaled to spectral radius 0.995 if needed; no trajectory clipping is applied.
This is a new versioned configuration and result artifact; the frozen checkpoint
is unchanged.

## Real-data outcome

All corrected trajectories are finite for all 175 test landmarks. The numerical
explosion is eliminated.

| Session | Old max abs | Corrected max abs | Old H20 MSE | Corrected H20 MSE | Persistence H20 MSE |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.77×10²⁰ | 3.20 | 4.92×10⁴ | 1.424 | 0.582 |
| 1 | 1.06×10¹⁵ | 6.77 | 9.56×10² | 3.131 | 0.586 |
| 2 | 9.55×10⁸ | 3.48 | 37.4 | 2.397 | 2.526 |
| 3 | 2.05×10¹⁷ | 3.29 | 2.82×10⁴ | 1.290 | 0.601 |
| 4 | 4.46×10⁹ | 2.56 | 41.9 | 1.240 | 1.089 |

Stability is bought at the cost of fidelity. The corrected model beats
persistence at H20 only in session 2, and by a small margin. In the other four
sessions it is worse. At H100 it remains finite but decays toward a narrow
trajectory and does not recover the observed variance.

## Synthetic control

A known stable latent AR system observed through a first-order calcium filter
was fitted with the same code. The old rollout is acceptable at short horizons
but diverges at H50/H100 (MSE 1.21 and 143.6; maximum absolute value 84.5). The
corrected rollout stays finite (MSE 0.0237 and 0.0238) and beats persistence at
both horizons. This confirms that the implementation can recover a stable
latent system when the data-generating process is stable and the observation
model is correct.

## Decision

- Gate B: **PASS** for numerical stability of the corrected model.
- The corrected model is **not** credible for virtual-intervention
  interpretation: it is stable but underfits and is worse than persistence in
  four of five sessions.
- The original model remains frozen and is marked unusable for autonomous
  rollout. No exploding trajectory is clipped or presented as a result.
- Any future use must report both stability and predictive fidelity; stability
  alone is not evidence of a useful dynamical model.
