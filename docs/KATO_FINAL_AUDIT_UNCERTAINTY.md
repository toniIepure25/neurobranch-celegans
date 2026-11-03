# Kato final audit: uncertainty and dependence

## 1. What the interval is

The primary interval is a moving-block bootstrap over 175 overlapping H20 forecast
landmarks, pooled with equal session weight. Landmarks are spaced 10 frames apart
but each forecast covers 20 frames, so adjacent landmarks share future
observations and fitted histories. The five recordings are repeated sessions, not
five confirmed independent animals.

The interval therefore describes variability **within and across these five
recordings**. It is not an animal-level confidence interval, and 175 is not 175
independent biological replicates.

## 2. Block-length sensitivity

Contrast: `B5-10 − B3` PCA-10 H20 energy; positive means B3 is better. Official
seed `1700 + block`, 1,000 draws.

| Block | Mean | 95% interval | Positive |
| ---: | ---: | --- | :---: |
| 3 | +0.2644 | `[+0.0972, +0.4018]` | yes |
| 5 (primary) | +0.2644 | `[+0.0750, +0.4031]` | yes |
| 10 | +0.2644 | `[+0.0350, +0.3994]` | yes |
| 20 | +0.2644 | `[+0.0921, +0.2939]` | yes |

All four intervals remain positive, so the result is not dependent on choosing
block 5. The block-20 interval is nevertheless fragile: with 33–38 landmarks per
session, it resamples only two blocks per session. Its narrower width is not
evidence of greater precision.

## 3. Session heterogeneity

| Session | Landmarks | Mean `B5−B3` | Fraction B3 better | Within-session block-5 interval |
| ---: | ---: | ---: | ---: | --- |
| 0 | 35 | +0.3003 | 0.686 | `[+0.0680, +0.5056]` |
| 1 | 35 | +0.3292 | 0.686 | `[-0.0353, +0.7672]` |
| 2 | 34 | −0.1931 | 0.500 | `[-0.6045, +0.2185]` |
| 3 | 38 | +0.4075 | 0.579 | `[-0.1538, +0.6696]` |
| 4 | 33 | +0.4645 | 0.758 | `[+0.1263, +0.8581]` |

Four of five sessions have a positive point estimate; session 2 reverses. Only
sessions 0 and 4 have within-session intervals excluding zero. This is meaningful
heterogeneity, not a uniform effect.

Leave-one-session-out pooled means remain positive:

| Excluded | Mean advantage | n |
| ---: | ---: | ---: |
| 0 | +0.2555 | 140 |
| 1 | +0.2483 | 140 |
| 2 | +0.3748 | 141 |
| 3 | +0.2248 | 137 |
| 4 | +0.2179 | 142 |

No single session drives the pooled sign.

## 4. Serial dependence and landmark subsampling

Within-session lag-1 autocorrelation of the paired difference ranges from 0.179
to 0.597; session 1 is the most dependent. Lag-2 and lag-3 values are also
non-zero in several sessions. The moving-block bootstrap is an attempt to respect
this, but the blocks are short relative to the 20-frame forecast overlap.

Taking every 2nd, 3rd or 5th landmark gives means `+0.208`, `+0.260` and `+0.401`
with 89, 60 and 36 landmarks. The direction is stable, but these are
dependence-reducing sensitivity analyses, not independent tests. The every-5th
estimate is larger, so it should not be presented as confirmation.

## 5. Horizon dependence

| Horizon | Mean `B5−B3` | Block-5 95% interval | B3 better |
| ---: | ---: | --- | :---: |
| 1 | −0.0637 | `[-0.0967, -0.0311]` | no; B5 better |
| 5 | +0.0835 | `[+0.0339, +0.1145]` | yes |
| 10 | +0.1471 | `[+0.0629, +0.2037]` | yes |
| 20 | +0.2644 | `[+0.0750, +0.4031]` | yes |

The sign reverses at H1. The positive result is a medium-to-long-horizon energy
result, not a general short-horizon forecasting improvement.

## 6. Representation and influence

The B5−B3 contrast is positive in PCA-10 energy (`+0.2644`, interval excludes
zero) and neuron-space energy (`+0.1617`), but the neuron-space interval includes
zero. PCA-10 MSE is `+0.0545` with an interval including zero; neuron MSE is
`−0.0354` with an interval including zero. The result is representation-specific
and metric-specific.

Descriptive energy strata show the advantage is largest for the lowest-B3-energy
landmarks (`+0.473`) and becomes slightly negative in the highest quartile
(`−0.122`). This is exploratory influence, not a new test.

## 7. Verdict

The positive point estimate survives block lengths 3, 5, 10 and 20 and
leave-one-session-out pooling. It does **not** survive as a uniform effect:
session 2 reverses, most within-session intervals include zero, H1 reverses, and
MSE/neuron-space outcomes are mixed.

The uncertainty statement should be:

> A modest, horizon-dependent exploratory energy advantage is present under all
> tested block lengths, but the intervals reflect overlapping landmarks within
> five repeated recordings and do not establish animal-level or confirmatory
> robustness.

It should not be:

> The effect is robustly confirmed across animals and metrics.
