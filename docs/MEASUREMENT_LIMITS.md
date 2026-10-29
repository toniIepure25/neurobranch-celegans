# Measurement limits of the C. elegans perturbation atlas

This note explains what a reliability ceiling is, why the corrected numbers
differ from the Phase 6-8 numbers, and how to use the ceiling when comparing
models.

## The attenuation ceiling in one paragraph

If a measured profile $m$ equals a true profile $\theta$ plus independent noise,
the correlation between $m$ and any predictor $p$ that is independent of that
noise cannot exceed $\sqrt{\rho}$, where $\rho$ is the reliability of $m$. A
model that predicts $\theta$ perfectly still correlates only $\sqrt{\rho}$ with
the observed data. Comparing a model's correlation to 1 therefore understates
how well it is doing; comparing it to $\sqrt{\rho}$ is the fair test.

This is why the benchmark reports a ceiling next to every predictive score. It
is also why the ceiling must be estimated on the same facet as the claim:
within-animal repetition and across-animal generalisation are different
measurements with different ceilings.

## Two facets, two questions

**Repeat reliability** asks whether stimulating the same animal twice produces
the same profile. It bounds claims about a single animal's response.

**Single-animal reliability** asks whether one animal's profile reproduces the
shared target profile. It bounds animal-held-out claims, because the held-out
animal contributes both sampling noise and its own idiosyncratic profile
variance. This is the relevant ceiling for Phases 6-8.

The second ceiling is lower because the denominator includes animal × responder
interaction: if animals have different response profiles, a held-out animal is
genuinely harder to predict even with infinite data per animal.

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
separates the two facets. It is still a moment estimator: it assumes additive
noise and does not model response dynamics.

## Current estimates

From `reports/phase9/reliability_summary.json`:

| Facet | Reliability | Ceiling $\sqrt{\rho}$ | Targets |
| --- | ---: | ---: | ---: |
| Repeat (within animal) | 0.243 | 0.493 | 152 |
| Single animal (held out) | 0.223 | 0.473 | 126 |

The ceilings are close to 0.5. A model reaching a correlation of 0.10 with the
observed profile is at about 21% of the single-animal ceiling, not 64%.

## What the ceiling does not mean

- It is **not** evidence that the response is random or mechanistically
  uninteresting.
- It is **not** a causal effect or a statement about intervention.
- It is **not** a property of the worm alone; it is a property of this atlas,
  this response window and this summary statistic.
- A low ceiling does not excuse a weak model; it changes what counts as strong.

## Design implications

For the repeat facet, reliability at $n$ stimulations is approximately
$\tau^2 / (\tau^2 + \sigma^2/n)$. Rearranging, the number of stimulations needed
to reach a target reliability $\rho^\*$ is

$$
n^\* \approx n_0 \frac{\rho^\*}{1-\rho^\*} \frac{1-\rho_0}{\rho_0},
$$

where $\rho_0$ is the reliability observed at $n_0$ stimulations. With
$\rho_0 = 0.243$ at $n_0 = 2$, reaching 0.8 requires roughly 25 stimulations per
animal per target — an order of magnitude beyond the current atlas for most
targets. This is a measurement-design conclusion, not a modelling one.

For a two-arm animal-randomised comparison, the minimum detectable effect is

$$
\text{MDE} = (z_{1-\alpha/2} + z_{\text{power}})\,\sigma\sqrt{2/n_{\text{arm}}}.
$$

With the observed between-animal spread, a difference of 0.02 Spearman needs
substantially more animals per arm than Phase 8 had. `neurobranch.benchmark.power`
provides these calculations and is tested against the normal approximation.

## Practical rules

1. Report the ceiling next to every held-out score.
2. Report the MDE next to every claimed difference.
3. Never compare a model to 1.0; compare it to the ceiling.
4. Do not interpret a difference smaller than the MDE, even if its interval
   excludes zero.
5. Treat a low ceiling as a prompt to redesign the measurement, not to lower
   the bar for models.
