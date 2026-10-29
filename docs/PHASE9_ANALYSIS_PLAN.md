# Phase 9 analysis plan: bounded evaluation

**Status:** internal analysis plan, not a registered report. It was written after
exploratory reliability estimates had been computed, so it cannot carry the
authority of a preregistration. It is recorded here so that the decision rules
used by `reports/phase9/claims.json` are inspectable and fixed before the
remaining audit conclusions are drawn.

## Why this phase exists

Phases 6-8 compared predictive scores against a split-half "ceiling" of about
0.152 and reported that the CRO model reached 64% of it. That ceiling came from
a split-half Spearman-Brown estimator that is biased downward when targets have
few events and that produced normalised scores above 1 in Phase 7. A benchmark
whose ceiling is wrong cannot say whether a model is close to the measurement
limit or far from it.

Phase 9 replaces the ceiling estimator, quantifies design resolution, and
attaches an explicit status to every quantitative claim.

## Estimands

For target $t$, animal $a$ and responder neuron $r$, let $m_{a,r}$ be the mean
response over repeated stimulations, $n_{a,r}$ the number of stimulations and
$v_{a,r}$ their sample variance.

**Repeat reliability (within animal).** For each animal with at least three
responders:

$$
V_a = \operatorname{Var}_r(m_{a,r}), \qquad
S_a = \operatorname{mean}_r\!\left(\frac{v_{a,r}}{n_{a,r}}\right), \qquad
\rho_{\text{repeat}} = \frac{\sum_a \max(V_a - S_a, 0)}{\sum_a V_a}.
$$

**Single-animal reliability (across animals).** With $\bar m_r$ the mean profile
over animals and $A$ animals:

$$
V = \operatorname{Var}_r(\bar m_r), \qquad
S = \operatorname{mean}_r\!\left(\frac{1}{A^2}\sum_a \frac{v_{a,r}}{n_{a,r}}\right),
\qquad \sigma^2_{\text{signal}} = \max(V - S, 0).
$$

The between-animal profile variance is
$\tau^2_r = \max(\operatorname{Var}_a(m_{a,r}) - \operatorname{mean}_a(v_{a,r}/n_{a,r}), 0)$.
Then

$$
\rho_{\text{single}} =
\frac{\sigma^2_{\text{signal}}}
{\sigma^2_{\text{signal}} + \operatorname{mean}_r(v_{a,r}/n_{a,r}) + \operatorname{mean}_r(\tau^2_r)}.
$$

The attenuation ceiling is $\sqrt{\rho}$: the maximum attainable correlation
between the observed profile and any error-free predictor.

Both estimators are variance-corrected moment estimators. They are **not**
REML, not a causal model, and not a digital twin.

## Hypotheses and decision rules

| ID | Statement | Test | Decision rule |
| --- | --- | --- | --- |
| H9.1 | Single-animal reliability is below 0.5 | upper bound | supported iff the 95% CI upper limit < 0.5 |
| H9.2 | Repeat-stimulation reliability is below 0.5 | upper bound | supported iff the 95% CI upper limit < 0.5 |
| H9.3 | Phase 8 pair-level advantage is a resolved positive effect | superiority | supported iff CI excludes 0 **and** MDE ≤ SESOI = 0.02 |
| H9.4 | EIG selection differs from random by more than 0.01 | equivalence | `bounded_null` iff the 95% CI lies inside ±0.01 |
| H9.5 | Profiles transfer across cohorts | superiority | deferred; no second cohort has been harmonised |

Superiority statuses follow this fixed order: `underpowered` if MDE > SESOI,
then `supported` if the interval excludes zero and the estimate reaches the MDE,
then `bounded_null` if the interval lies inside ±SESOI, otherwise
`not_supported`.

MDE for an existing result is reconstructed from its reported 95% interval:

$$
\text{MDE} = \left(z_{1-\alpha/2} + z_{\text{power}}\right)
\frac{\text{CI width}}{2 z_{1-\alpha/2}},
$$

with $\alpha = 0.05$ and power $= 0.8$. This is a statement about design
resolution, not post-hoc power for the observed effect.

## Multiplicity and clustering

- The unit of resampling and of generalisation is the animal, not the event.
- Target-level intervals use a normal approximation across targets. Targets
  share animals, so these intervals are descriptive and are labelled as such.
- Per-target intervals are not reported: a variance ratio computed from a
  handful of animals has an unstable interval, and a cluster bootstrap that
  duplicates animals inflates split-half agreement.
- No multiplicity correction is applied to the seven fixed claims; they are a
  small, pre-specified family, and each is reported with its interval.

## Deviations from the original Phase 9 plan

1. **Split-half reliability for the across-animal facet was abandoned.** A
   cluster bootstrap placed resampled animals in both halves, inflating the
   apparent ceiling (mean bootstrap upper limit 0.86 against a point estimate
   of 0.08). The variance-corrected estimator replaced it.
2. **Per-target bootstrap intervals were abandoned** for the reason above.
   Aggregate intervals are target-level and descriptive.
3. **The design engine (EIG portfolio, SBC, replay) was not implemented** in
   this phase. The measurement audit had to come first; a design optimiser
   built on an unverified ceiling would optimise noise.
4. **H9.5 was deferred.** Cross-cohort transfer needs a harmonised second
   dataset; none is present in the repository.

## What would falsify the Phase 9 conclusion

The claim that the atlas is measurement-limited would be weakened if a
re-analysis with longer per-animal stimulation blocks, or with a different
response summary (integral instead of mean), produced single-animal reliability
above 0.5. That experiment is specified in `docs/NEXT_EXPERIMENTS.md`.
