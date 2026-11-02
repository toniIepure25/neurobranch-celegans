# Dynamical modes and operator stability

The fitted object is a 103-dimensional lag-one linear operator. Its eigenvalues
describe linear decay and rotation; they do **not** identify biological
oscillators, synapses, or command neurons.

## Frozen training operators

| Session | Spectral radius | Leading decay (frames) | Non-normality | Transient peak | Eigenvector condition |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.995 | 199.5 | 1.55 | 3.57 | 116 |
| 1 | 0.995 | 199.5 | 1.96 | 4.67 | 1,096 |
| 2 | 0.994 | 173.2 | 1.47 | 3.78 | 107 |
| 3 | 0.995 | 199.5 | 2.70 | 4.28 | 234 |
| 4 | 0.991 | 112.3 | 1.83 | 3.63 | 552 |

The spectral cap binds in four sessions. The fitted decay time is therefore a
cap-limited quantity, not a measured biological timescale. Non-normality is
large, and the largest transient amplification is 3.6-4.7 times the asymptotic
mode scale. A stable spectral radius does not imply a stable transient response.

## What is stable and what is not

Three chronological training segments and five moving-block bootstrap fits were
compared within each session.

- Spectral radius is tightly constrained by the cap: median segment CV `0.0015`.
- Leading decay is less stable: median segment CV `0.224`, with session 2 ranging
  from 62 to 186 frames across segments.
- Non-normality has median segment CV `0.113`.
- The off-diagonal operator structure is **not stable**: median pairwise
  correlation of off-diagonal coefficients across chronological segments is
  `0.052`, and relative Frobenius differences are around `1.2-1.5`.

The joint pattern is the central mode result: the **radius** is stable because it
is regularized, while the **cross-neuron coefficient pattern** is not. A mode
summary that reports only the spectral radius hides this instability.

Gate G3 is `FAIL` because the prespecified off-diagonal correlation threshold
(`> 0.50`) is not met, even though the radius and non-normality thresholds are.

## Complex modes are not oscillations

There are 35-49 eigenvalues with positive imaginary part, out of 103. Their
median periods are 4.7-20.3 seconds and their median decay is below one frame for
the leading complex pair. The observed training spectra have median dominant
periods of 252-756 seconds, dominated by slow drift rather than a matched 5-20 s
rhythm.

Complex eigenvalues are therefore common but not evidence for a reproducible
oscillation. No oscillatory biological claim is made. A future test would need a
pre-specified frequency, a held-out session, and a noise/observation model that
can distinguish a rotation from a non-normal transient.

## Interpretation boundary

- The operator is an inferred predictive object, not a connectome.
- A stable eigenvalue does not identify a cell or a synapse.
- A transient peak is a linear-model property; it can arise from non-normal
  coupling without any biological pulse.
- The instability of off-diagonal coefficients is a reason not to interpret
  single coefficients as interactions.

See [cross-neuron information](KATO_INTERACTIONS_CROSS_NEURON_INFORMATION.md) for
the predictive ablation and [sensitivity](KATO_INTERACTIONS_SENSITIVITY.md) for
the bounded virtual edits.
