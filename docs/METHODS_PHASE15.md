# Phase 15 methods: three reliability estimands without extrapolation

**Frozen protocol:** `research/phase15_protocol.md` (SHA256 recorded in
`reports/phase15/manifest.json`).
**Status:** measurement audit; no model is fitted and no causal claim is made.

## 1. Why the old headline was replaced

Phase 11 reported `R(22) = 0.3136077301426884` and
`sqrt(R(22)) = 0.5600069018705827` for 126 targets at 80% responder coverage.
That calculation evaluates every target at 22 animals. Only 39 of the 126
targets have at least 22 animals; for the other 87 the value is a design
counterfactual. `R(22)` is therefore not an identified observed-data quantity,
and its square root is not a bound. Phase 15 keeps the arithmetic as a
historical reference but separates the estimands.

## 2. Input and population

- Input table: `reports/phase3/randi_trial_table_phase3.csv.gz`.
- Response: `response_mean`.
- Animal unit: `dataset_id`.
- Minimum 1 event, at least 6 animals, and each responder observed in at least
  80% of the target's animals.
- The responder set is fixed before uncertainty is computed; the jackknife is
  conditional on that set.

## 3. Variance treatment

The primary treatment is V1. For every target x responder cell with at least two
events and positive sample variance, compute the cell variance. The common
one-event variance is the median of those cell variances:

`omega^2 = median(s^2)`.

The cell sampling variance is `omega^2 / n_events`. The cell total variance used
by the estimator is `signal + noise + interaction`, with the sampling component
inserted as the one-event term. V0 (historical zero one-event variance) and V1h
(one-event variance applied to every cell) are sensitivity analyses; V2
(`min_events=2`) changes the target population and is labelled as such.

## 4. Estimands

For target `t` with `n_t` observed animals:

```
R_t(n) = signal_t / (signal_t + (noise_t + interaction_t) / n)
```

clipped to `[0, 1]`.

| Estimand | Definition | Population |
| --- | --- | --- |
| `R_obs` | `mean_t R_t(n_t)` | all 126 eligible targets |
| `R_22` | `mean_t R_t(22)` | all 126; 87 counterfactual |
| `R_id` | `mean_{t: n_t >= 22} R_t(n_t)` | 39 identified targets |

`R_obs` and `R_id` are not interchangeable: `R_id` conditions on larger targets
and is expected to be higher if reliability grows with animal count. Ceilings
are `sqrt(R)` and are derived descriptive quantities, not bounds.

## 5. Uncertainty

The primary interval is a delete-one-animal jackknife:

1. Delete one animal from every target containing it.
2. Recompute the target components with the remaining animals.
3. Recompute the aggregate estimand; targets without the deleted animal keep
   their full-sample value.
4. For `R_obs` and `R_id`, reduce `n` by one; for `R_22`, keep `n = 22`.
5. Use the finite-sample factor `(A - 1) / A` and a normal 95% interval.

The procedure is conditional on the 80%-coverage responder set. Replicates with
fewer than three valid animals are excluded; the valid fraction is reported and
must be at least 0.95.

Secondary diagnostics:

- A BCa diagnostic was prototyped during the run but **not reported**: the
  delete-one-animal values are not a bootstrap distribution, so percentile
  calibration on them is not valid. The primary jackknife interval is unchanged;
  this failed diagnostic is recorded as a deviation rather than quietly dropped.
- Dirichlet Bayesian bootstrap over animals, 2,000 draws.
- Animal-half replication and held-out half-correlation prediction.
- Two-way bootstrap: diagnostic only; never a reported interval.

## 6. Calibration

The simulation uses the real target sizes, missingness masks, and cell sampling
variances. For each target, draw `mu_r ~ N(0, signal_t)`,
`u_ar ~ N(0, interaction_t)` for observed cells, and
`e_ar ~ N(0, sampling_var_ar)`, then form `y_ar = mu_r + u_ar + e_ar`. The truth
is the same functional at the fixed target counts.

- 1,000 replicates for the primary coverage study.
- 500 replicates for a Student-t / heteroskedastic misspecification check.
- Pass: coverage in `[0.93, 0.97]` for each estimand at nominal 95%.

The interval is called `not_calibrated` if any estimand fails. Calibration is a
property of the interval, not of the biological signal.

## 7. Null controls

Two controls are run for each estimand:

- Permutation of responder labels within target, preserving missingness.
- Sign flip of animal-level contributions.

Report the median and 95th percentile. Pass requires median `<= 0.05` and q95
`<= 0.15` for both controls. A null control that exceeds these thresholds means
the estimator can manufacture apparent reliability from structureless data.

## 8. Extrapolation and ranking checks

- Animal-half extrapolation: fit components on half A, predict the reliability
  of both half-profiles, and compare with the observed half-to-half Pearson
  correlation. Pass: median absolute error `<= 0.10` and `|bias| <= 0.05`.
- Ranking stability: split animals in half, compute `R_obs` per target in each
  half, and compare Spearman correlation and top-10 Jaccard overlap. Pass:
  Spearman lower CI `> 0.50` and median Jaccard `> 0.60`.

## 9. Decision labels

The audit fails closed:

- `calibrated_descriptive`: calibration and nulls pass; interval is descriptive.
- `counterfactual_only`: calibration and nulls pass, but extrapolation fails.
- `not_calibrated`: calibration or nulls fail.
- `no_ranking`: ranking stability fails.

No label authorizes a causal, transport, digital-twin, foundation-model,
animal-budget, or target-ranking claim.

## 10. Reproducibility

```bash
python scripts/phase15_reliability.py
python scripts/phase15_calibrate.py --heavy --reps 500
python scripts/phase15_nulls.py
python scripts/phase15_card.py
python scripts/phase15_manifest.py
python scripts/phase15_validate.py
```

The manifest hashes every artifact except itself and `validation.json`. The
validator checks the schema, estimand counts, jackknife consistency, gate logic,
decision consistency, and artifact hashes.
