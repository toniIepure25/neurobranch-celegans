# Cross-neuron information

## Question

Is the joint H20 advantage of a population VAR carried by time-lagged information
shared between recorded neurons, or can an independent per-neuron model with the
same innovations reproduce it?

## Matched experiment

Every model in this comparison uses the **same 100 J5 innovation draws** at each
landmark. Only the conditional mean differs. The fixed 10-D PCA is used for
scoring only; the models are fitted in the 103-neuron standardized space.

| Model | What changes |
| --- | --- |
| B0 persistence | no dynamics |
| B1 independent AR, window 200 | own history only |
| B2 local mean, window 500 | no lag structure |
| B3 population VAR, window 500 | full cross-neuron operator |
| B4 diagonalized B3 | cross coefficients removed from the fitted full operator |
| B4r independent AR, window 500 | own history only, matched window |
| B5 circular / block | B3 family fitted on column-wise temporal surrogates |

B4 and B4r are different ablations. B4 keeps the diagonal of the full-fit
operator; B4r refits an own-neuron-only model on the same window. Neither is a
biological lesion.

## Scores

| Model | H20 energy | H20 trajectory | Mean MSE | 80% coverage | Neural covariance error |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 persistence | 3.657 | 2.726 | 2.056 | 0.965 | 0.853 |
| B1 independent (common innovations) | 3.071 | 2.380 | 1.956 | 0.611 | 0.521 |
| B1 independent (exact frozen sampler) | 3.146 | 2.435 | 1.961 | 0.554 | 0.505 |
| B2 local mean | 4.615 | 4.589 | 3.539 | 0.406 | 0.519 |
| B3 population | **2.700** | **2.060** | **1.707** | 0.830 | 0.604 |
| B4 diagonalized | 4.568 | 4.296 | 3.515 | 0.428 | 0.509 |
| B4r independent AR, window 500 | 3.247 | 2.526 | 2.135 | 0.639 | 0.554 |
| B5 circular | 3.949 | 3.500 | 2.991 | 0.566 | 0.560 |
| B5 block | 4.130 | 3.736 | 3.366 | 0.601 | 0.588 |

## Paired contrasts

Primary endpoint, block-5 within-session interval, positive means B3 is better:

| Contrast | Mean improvement | 95% interval | Fraction of landmarks better |
| --- | ---: | ---: | ---: |
| B3 vs B1, energy | `+0.371` | `[0.128, 0.530]` | 0.606 |
| B3 vs B4r, energy | `+0.547` | `[0.355, 0.698]` | 0.720 |
| B3 vs B4r, mean MSE | `+0.428` | `[0.191, 0.621]` | 0.731 |
| B3 vs B5 circular, energy | `+1.249` | `[0.888, 1.537]` | 0.840 |
| B3 vs B1, mean MSE | `+0.250` | `[-0.029, 0.432]` | 0.611 |

The energy result is positive; the B3-versus-B1 mean-MSE result is not
conclusive. Energy and MSE can diverge because the energy score also rewards
ensemble spread and shape. The audit therefore does not claim a pure
conditional-mean improvement over the best independent baseline. Against the
matched 500-frame independent refit, B3 does improve both energy and mean MSE,
so window length and cross-neuron structure are partly confounded.

## Surrogate controls

Circular shifts and block permutations were applied column-wise inside the causal
500-frame window. They preserve each neuron's marginal values and (for circular
shifts) lag-1 autocorrelation while destroying cross-neuron alignment.

| Method | Diagonal AR correlation with B3 | Lag-1 ACF absolute error | H20 energy | H20 trajectory |
| --- | ---: | ---: | ---: | ---: |
| Circular | 0.903 | 0.0021 | 3.949 | 3.500 |
| Block, 25 frames | 0.890 | 0.0248 | 4.130 | 3.736 |

B3 beats both surrogates by about one energy unit. The null is deliberately
destructive, so this is evidence that cross-neuron alignment matters, not a
decomposition into direct and indirect pathways. The surrogates are **worse than
B1**, which is expected: fitting a full operator to shuffled data creates noisy
cross terms. Their role is a negative control, not a realistic generative model.

## Session heterogeneity

B3 versus B1 energy by session:

| Session | Mean improvement | Landmarks |
| --- | ---: | ---: |
| 0 | `+0.570` | 35 |
| 1 | `-0.148` | 35 |
| 2 | `+0.143` | 34 |
| 3 | `+0.660` | 38 |
| 4 | `+0.613` | 33 |

Four of five sessions are positive, but session 1 reverses. The pooled interval
is not an animal-level interval and does not establish consistency across
animals. This is a repeated-session result.

## Conclusion

The recorded population contains predictive cross-neuron information beyond
independent own-history dynamics: the same innovations produce a lower H20
energy when the full lag-one operator is used. The result is not a clean MSE gain,
not stable in every session, and not identifiable as direct coupling. It is a
**predictive dependence** result.

**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**
