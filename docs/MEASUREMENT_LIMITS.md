# Measurement limits of the C. elegans perturbation atlas

This note explains what a reliability ceiling is, which facet each metric
belongs to, and how to use the ceiling when comparing models. The Phase 10
numbers below are retained for derivation, but the headline pooled ceiling is
superseded by the coverage-aware Phase 11 value **0.560** (see
[`PHASE11_RESULTS.md`](PHASE11_RESULTS.md) and C8).

## The attenuation ceiling in one paragraph

If a measured profile $m$ equals a true profile $\theta$ plus independent noise,
the correlation between $m$ and any predictor $p$ that is independent of that
noise cannot exceed $\sqrt{\rho}$, where $\rho$ is the reliability of $m$. A
model that predicts $\theta$ perfectly still correlates only $\sqrt{\rho}$ with
the observed data. Comparing a model's correlation to 1 therefore understates
how well it is doing; comparing it to $\sqrt{\rho}$ is the fair test.

The ceiling must be estimated on the same facet as the claim. There are three
facets in this project, and they answer different questions.

## Three facets, three questions

**Repeat reliability** asks whether stimulating the same animal twice produces
the same profile. It bounds claims about a single animal's repeatability.

**Single-animal reliability** asks whether one animal's profile reproduces the
shared target profile. It bounds per-animal claims, because the animal
contributes both sampling noise and its own idiosyncratic profile variance.

**Pooled reliability** asks how well a profile averaged over $n$ animals
estimates the shared target profile. It bounds the Phase 6-8 pair metrics,
because those metrics are computed on pair means pooled over the roughly 22
animals of a test fold.

The single-animal ceiling is lower than the pooled ceiling because pooling
averages away both sampling noise and part of the animal x responder
interaction. Using the single-animal ceiling for a pooled metric overstates the
measurement limitation.

## Current estimates

From `reports/phase10/reliability_summary.json` (historical) and the Phase 11
coverage sensitivity:

| Facet | Reliability | Ceiling $\sqrt{\rho}$ | Targets |
| --- | ---: | ---: | ---: |
| Repeat (within animal) | 0.243 | 0.493 | 152 |
| Single animal ($n=1$, sparse coverage) | 0.162 | 0.403 | 126 |
| Pooled over 22 animals, sparse coverage | 0.693 | 0.833 | 126 |
| Pooled over 22 animals, coverage >= 0.8 | **0.314** | **0.560** | 126 |
| Pooled over all 110 animals | 0.873 | 0.934 | 126 |

The single-animal ceiling is near 0.4. The coverage-aware fold-level ceiling is
0.560, not 0.833. A model reaching a correlation of 0.10 with the observed
pooled profile is at about **17.6%** of the coverage-aware ceiling (about 11.8%
of the Phase 10 sparse-coverage value), not 64% and not 21%.

## Why the old split-half ceiling was wrong

The Phase 6-8 estimator split events into two halves, computed Spearman-Brown,
and truncated negative values to zero. Three problems:

1. **Small-sample bias.** With a handful of events per target, half-splits are
   noisy and the Spearman-Brown correction amplifies that noise. Truncation at
   zero then biases the average ceiling downward.
2. **Event-level splitting ignores animal clustering.** Splitting events within
   an animal is not the same as splitting animals. The resulting number answers
   a within-animal question while being used to bound an across-animal claim.
3. **It produced impossible normalised scores.** Phase 7 divided by a truncated
   ceiling and reported values above 1, which is a mathematical signal that the
   denominator was not a valid ceiling.

The variance-corrected estimator removes the truncation, uses all events, and
separates the facets. It is still a moment estimator: it assumes additive noise
and does not model response dynamics.

## Why the Phase 9 single-animal number changed

Phase 9 reported 0.223 (ceiling 0.473). Two defects were found:

1. The pooled sampling variance divided by the total animal count squared
   instead of the per-responder count squared. Responders measured in fewer
   animals were under-corrected, inflating the signal.
2. The signal $V-S$ still contained `interaction / A`, and the reliability
   denominator then added `interaction / A` again.

With per-responder counts and the interaction correction, the single-animal
reliability is 0.162 (ceiling 0.403). The repeat facet is unaffected because it
does not pool across animals.

## What the ceiling does not mean

- It is **not** evidence that the response is random or mechanistically
  uninteresting.
- It is **not** a causal effect or a statement about intervention.
- It is **not** a property of the worm alone; it is a property of this atlas,
  this response window and this summary statistic.
- A low ceiling does not excuse a weak model; it changes what counts as strong.
- A high pooled ceiling does not validate a model; it raises the bar for calling
  the model good.

## Design implications

For the repeat facet, reliability at $n$ stimulations is approximately
$\tau^2 / (\tau^2 + \sigma^2/n)$. Rearranging, the number of stimulations needed
to reach a target reliability $\rho^\*$ is

$$
n^\* \approx n_0 \frac{\rho^\*}{1-\rho^\*} \frac{1-\rho_0}{\rho_0},
$$

where $\rho_0$ is the reliability observed at $n_0$ stimulations. With
$\rho_0 = 0.243$ at $n_0 = 2$, reaching 0.8 requires roughly 16 further
stimulations per animal per target.

For the animal facet, the analogous formula is

$$
n^\* = \frac{(\text{noise} + \text{interaction})\,\rho^\*}{(1-\rho^\*)\,\text{signal}}.
$$

`neurobranch.design.animals_for_reliability` implements it. On this atlas the
median budget for a target reliability of 0.8 is 25 animals, with a 90th
percentile of 104. The Phase 8 fold of 22 animals is near the median but far
below the p90.

For a two-arm animal-randomised comparison, the minimum detectable effect is

$$
\text{MDE} = (z_{1-\alpha/2} + z_{\text{power}})\,\sigma\sqrt{2/n_{\text{arm}}}.
$$

With the observed between-animal spread, a difference of 0.02 Spearman needs
substantially more animals per arm than Phase 8 had. `neurobranch.benchmark.power`
provides these calculations and is tested against the normal approximation.

## Practical rules

1. Report the ceiling next to every held-out score, on the matching facet.
2. Report the MDE next to every claimed difference.
3. Never compare a model to 1.0; compare it to the ceiling for its metric.
4. Do not interpret a difference smaller than the MDE, even if its interval
   excludes zero.
5. Treat a low ceiling as a prompt to redesign the measurement, not to lower
   the bar for models.
6. Do not treat the animal jackknife interval as a calibrated confidence
   interval; its coverage study failed.
