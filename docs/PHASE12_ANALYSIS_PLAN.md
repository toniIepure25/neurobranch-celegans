# Phase 12 analysis plan: hierarchical reliability and calibration gates

Status: executed. This file records the contract used for the Phase 12 artifact
run. It is an analysis plan for a measurement audit, not a registered clinical
trial and not a systematic review.

## 1. Question

Does an explicit hierarchical measurement model reproduce the coverage-aware
pooled reliability (0.314) and ceiling (0.560) at 80% responder coverage, and can
its uncertainty be calibrated well enough to support a target-level ranking or a
prospective animal-budget statement?

## 2. Estimand and model

For target $r$, animal $a$, and event $e$:

$$
y_{are} = \mu_r + u_{ar} + \varepsilon_{are},
\quad \mu_r \sim N(0,\sigma_r^2),
\quad u_{ar} \sim N(0,\tau_r^2),
\quad \varepsilon_{are} \sim N(0,v_{are}).
$$

$v_{are}$ is the known cell-level sampling variance. The reliability of an
animal-mean profile at $n$ animals is

$$
R_r(n) = \frac{\sigma_r^2}{\sigma_r^2 + (\tau_r^2 + \nu_r)/n},
$$

where $\nu_r$ is the mean sampling variance over the target's cells. The pooled
reliability is the median of $R_r(22)$ over eligible targets; the pooled ceiling
is the median of $\sqrt{R_r(22)}$. This is the same 22-animal fold size used in
Phases 6--11.

Model fitting details, including animal centering, the deterministic grid
posterior, and the rank-one marginal likelihood, are in
[`METHODS_HIERARCHICAL.md`](METHODS_HIERARCHICAL.md).

## 3. Eligibility and missingness

- Primary coverage: a responder must be observed in at least 80% of the
  target's animals.
- A target is eligible if it has at least two animals, two responders, and two
  events per responder after filtering.
- Missing cells are not imputed and do not contribute a sampling variance.
- The coverage curve is reported at 0.5, 0.8, and 1.0; no point is selected as a
  headline after seeing the result.
- The Phase 11 moment estimates (0.314, ceiling 0.560) are the reference values,
  not a target to be matched by tuning.

## 4. Calibration gates

All gates are evaluated on the full run. A single failure blocks adoption; no
threshold is relaxed after inspection.

| Gate | Requirement | Rationale |
|---|---|---|
| G1a SBC coverage | 22-animal coverage in [0.93, 0.97] | nominal 95% posterior interval |
| G1b SBC rank | KS test of rank uniformity p > 0.01 | detects biased posterior draws |
| G1c grid edge | maximum edge posterior mass < 0.05 | excludes boundary-dominated fits |
| G2 held-out PPC | animal-holdout interval coverage >= 0.90 and median absolute error <= 0.10 | predictive, not just internal |
| G3 negative control | animal-wise cell-profile permutation median R(22) <= 0.05 and q95 <= 0.15 | no reliability without true pairing |
| G4 prior sensitivity | range of pooled R(22) across HalfNormal scales <= 0.05 | conclusions not prior-driven |

G1a--G1c operationalise the synthetic-coverage requirement in
[`NEXT_EXPERIMENTS.md`](NEXT_EXPERIMENTS.md) section 10. G2 operationalises the
animal-holdout requirement. G3 and G4 are adversarial audits added before the
full run because the pilot showed posterior boundary and null-calibration
problems; they are reported whether they pass or fail.

## 5. Decision rule

Adopt the hierarchical posterior as the primary inferential model only if all of
G1--G4 pass **and** the hierarchical pooled estimate is within 0.05 of 0.314.

If any gate fails:

1. retain 0.560 as the only descriptive coverage-aware ceiling;
2. do not publish a calibrated reliability interval;
3. do not rank targets or recommend an animal budget from the posterior;
4. report the failed gates and the negative-control bias as a model limitation;
5. do not tune the prior, grid, or coverage filter to recover a pass.

## 6. Benchmark inference

The benchmark comparison uses animals, not folds or events, as the resampling
unit. Pair means are computed within each animal; animals are resampled with
replacement; the paired Spearman difference is recomputed. Multiplicity is
controlled with Holm across the five non-baseline configurations. A
configuration is labelled `supported` only when its paired interval excludes
zero, its Holm-adjusted p-value is below 0.05, and its 80% minimum detectable
effect is at or below the 0.05 smallest effect of interest. Otherwise it is
`underpowered` or `inconclusive`. Fold-level leaderboard values remain
descriptive.

## 7. Reporting

The generated artifacts are under `reports/phase12/`:

- `summary.json` -- gates, estimates, decision;
- `hierarchical_by_target.csv` -- per-target posterior summaries;
- `coverage_sensitivity.csv` -- coverage curve;
- `sbc.json` -- simulation-based calibration;
- `heldout_ppc.json` -- animal-holdout predictive check;
- `negative_control.json` -- permutation null;
- `prior_sensitivity.json` -- prior-scale audit;
- `benchmark_cluster.csv` / `.json` -- animal-cluster comparisons;
- `design_posterior.csv` -- planning table, conditional on the model;
- `manifest.json` / `manifest.sha256` -- provenance.

The input table is referenced by hash and is not redistributed.
