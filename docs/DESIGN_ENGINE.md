# Design engine: ceiling-targeted animal budgets

## What this is

`neurobranch.design` answers a narrow design question:

> Given the fitted signal, noise and interaction components, how many animals
> must be pooled before a target's profile reaches reliability `r`?

It does **not** select which target to stimulate next. The approved Phase 10
plan originally proposed an expected-information-gain portfolio. That was
replaced after the ceiling analysis showed that the binding design variable in
this atlas is the number of evaluation animals, not target selection. An EIG
optimiser built on components whose interval calibration had failed would have
optimised noise.

## Model

For a target profile `theta_r` shared across animals, animal-specific
deviations with variance `interaction`, and cell-mean sampling variance
`noise`:

```
reliability(n) = signal / (signal + (noise + interaction) / n)
```

Inverting gives the animal budget:

```
n = (noise + interaction) * r / ((1 - r) * signal)
```

`animals_for_reliability` returns `inf` when the goal is unattainable
(`signal <= 0` or `r >= 1`) and `1.0` when the error variance is zero.

## Results on the Randi atlas

| Goal | Median animals | p90 animals | Targets feasible |
| ---: | ---: | ---: | ---: |
| 0.5 | 6 | 26 | 97% |
| 0.7 | 15 | 61 | 97% |
| 0.8 | 25 | 104 | 97% |
| 0.9 | 56 | 234 | 97% |

Twenty-two animals per fold is close to the median budget for 0.8 but far below
the p90. This is a design statement about measurement precision, not a
biological claim.

## Calibration study

`coverage_study` simulates atlases with known components and checks whether the
animal jackknife interval covers the known fold-level reliability. It does not:
nominal 95% intervals covered about 60% of synthetic truths in the default
configuration, and lower fractions in other configurations. The interval
captures animal resampling but not responder sampling or component bias.

**The coverage study is a negative result.** It is retained as evidence that the
jackknife interval must not be presented as a calibrated confidence interval.
The design calculator uses point estimates only; its output should be treated as
a planning approximation with substantial uncertainty.

## What would make this a real design engine

1. A two-way (animal x responder) resampling scheme with nominal coverage on
   synthetic atlases.
2. A prospective validation on a new cohort, where the predicted budget is
   compared with the reliability actually achieved.
3. An explicit cost model if targets have different stimulation costs.

None of these is implemented. The module is deliberately a calculator, not an
optimiser.

## Reproduction

```powershell
python scripts/phase10_design.py
```

Artifacts: `reports/phase10/design_budget.csv`,
`reports/phase10/coverage_study.json`.

## Phase 12 posterior design table

`neurobranch.design.posterior_design` propagates the hierarchical posterior
uncertainty into the probability that a target reaches reliability 0.5 at a given
animal count. The output is `reports/phase12/design_posterior.csv`: for each
eligible target it reports the minimum number of animals at which the posterior
probability is at least 0.8, or `not_reached_within_budget` within 60 animals.

This is **not** a validated design engine. The hierarchical model failed its SBC
rank test, held-out predictive check and negative control, so the table is
conditional on a misspecified model. It is retained for transparency and for the
next modelling attempt, not for recommending a cohort size.

```powershell
python scripts/phase12_hierarchy.py
```

The script writes the design table together with the calibration artifacts; see
`docs/PHASE12_RESULTS.md`.
