"""Calibration checks for the hierarchical reliability model.

The functions here are deliberately separate from the fitter.  A hierarchical
posterior is not "calibrated" because it has credible intervals; it is
calibrated only if simulated data generated from the prior recover the true
reliability at the declared rate.  The real-data predictive check and the
negative control are reported alongside SBC so a passing synthetic check cannot
stand alone.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from neurobranch.benchmark.hierarchical import (
    _normalize_target,
    _reliability_draws,
    _TargetData,
    fit_hierarchical_cells,
)
from neurobranch.benchmark.variance import DEFAULT_ANIMAL_COLUMN, cell_summaries


def prepare_templates(
    frame: pd.DataFrame,
    *,
    coverage: float = 0.8,
    response: str = "response_mean",
    animal: str = DEFAULT_ANIMAL_COLUMN,
    min_events: int = 1,
    min_animals: int = 6,
) -> list[_TargetData]:
    """Eligible real-target templates used to preserve missingness patterns."""
    cells = cell_summaries(frame, response=response, animal=animal)
    templates: list[_TargetData] = []
    for _target, group in cells.groupby("_target", sort=False):
        data = _normalize_target(
            group,
            coverage=coverage,
            min_events=min_events,
            min_animals=min_animals,
            standardize=True,
        )
        if data is not None:
            templates.append(data)
    return templates


def _cells_from_template(
    data: _TargetData,
    y: np.ndarray,
    *,
    target_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for animal_index in range(data.n_animals):
        for responder_index in range(data.n_responders):
            if not bool(data.mask[animal_index, responder_index]):
                continue
            rows.append(
                {
                    "_animal": f"a{animal_index}",
                    "_target": target_id,
                    "_neuron": f"n{responder_index}",
                    "n": 1,
                    "mean": float(y[animal_index, responder_index]),
                    "var": float(data.v[animal_index, responder_index]),
                }
            )
    return pd.DataFrame(rows)


def _simulate_template(
    data: _TargetData,
    *,
    rng: np.random.Generator,
    sigma: float,
    tau: float,
) -> np.ndarray:
    mu = rng.normal(0.0, sigma, size=data.n_responders)
    interaction = rng.normal(0.0, tau, size=(data.n_animals, data.n_responders))
    noise = rng.normal(0.0, np.sqrt(data.v))
    y = mu[None, :] + interaction + noise
    return np.where(data.mask, y, np.nan)


def simulation_based_calibration(
    templates: list[_TargetData],
    *,
    n_reps: int = 200,
    targets_per_rep: int = 3,
    n_eval: tuple[int, ...] = (1, 22),
    draws: int = 1500,
    grid_resolution: int = 50,
    seed: int = 31,
) -> dict[str, Any]:
    """Rank-based SBC for the hierarchical reliability posterior.

    Parameters are drawn from the fitting prior (``sigma, tau ~ HalfNormal(1)``
    on the standardized scale).  Each replicate uses a real target's coverage
    mask and sampling variances.  Ranks should be uniform if the posterior is
    calibrated; finite-replicate noise is reported through the KS test.
    """
    if not templates:
        raise ValueError("no templates")
    if n_reps < 10:
        raise ValueError("n_reps must be >= 10")
    rng = np.random.default_rng(seed)
    ranks: dict[int, list[float]] = {n: [] for n in n_eval}
    covered: dict[int, list[bool]] = {n: [] for n in n_eval}
    widths: dict[int, list[float]] = {n: [] for n in n_eval}
    failures = 0
    for rep in range(n_reps):
        for target_index in range(targets_per_rep):
            template = templates[int(rng.integers(0, len(templates)))]
            sigma = float(abs(rng.normal(0.0, 1.0)))
            tau = float(abs(rng.normal(0.0, 1.0)))
            y = _simulate_template(template, rng=rng, sigma=sigma, tau=tau)
            target_id = f"sbc-{rep}-{target_index}"
            cells = _cells_from_template(template, y, target_id=target_id)
            result = fit_hierarchical_cells(
                cells,
                coverage=0.0,
                n_eval=n_eval,
                draws=draws,
                grid_resolution=grid_resolution,
                seed=int(rng.integers(0, 2**31 - 1)),
                standardize=False,
                center_animals=True,
            )
            draws_matrix = result.draws_by_target.get(target_id)
            if draws_matrix is None or not result.by_target["status"].eq("ok").any():
                failures += 1
                continue
            reliability = _reliability_draws(draws_matrix[:, 0], draws_matrix[:, 1], template.nu, n_eval)
            for n in n_eval:
                truth = sigma**2 / (sigma**2 + (tau**2 + template.nu) / float(n))
                values = reliability[n][np.isfinite(reliability[n])]
                if values.size < 50:
                    failures += 1
                    continue
                ranks[n].append(float(np.mean(values < truth)))
                low, high = np.quantile(values, [0.025, 0.975])
                covered[n].append(bool(low <= truth <= high))
                widths[n].append(float(high - low))
    output: dict[str, Any] = {
        "n_reps": int(n_reps),
        "targets_per_rep": int(targets_per_rep),
        "n_fits": int(n_reps * targets_per_rep),
        "n_failures": int(failures),
        "prior": "sigma, tau ~ HalfNormal(1) on the standardized scale",
        "parameters": {},
    }
    for n in n_eval:
        values = np.asarray(ranks[n], dtype=float)
        if values.size < 10:
            output["parameters"][f"n{n}"] = {
                "status": "insufficient_draws",
                "n": int(values.size),
            }
            continue
        ks = stats.kstest(values, "uniform")
        output["parameters"][f"n{n}"] = {
            "status": "ok",
            "n": int(values.size),
            "coverage": float(np.mean(covered[n])),
            "mean_width": float(np.mean(widths[n])),
            "mean_rank": float(values.mean()),
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "ranks": values.tolist(),
        }
    return output


def _pearson_rows(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    keep = mask & np.isfinite(left) & np.isfinite(right)
    if int(keep.sum()) < 3:
        return float("nan")
    x = left[keep]
    y = right[keep]
    if np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def heldout_predictive_check(
    frame: pd.DataFrame,
    *,
    coverage: float = 0.8,
    n_splits: int = 5,
    test_fraction: float = 0.2,
    max_targets: int = 40,
    draws: int = 1200,
    grid_resolution: int = 50,
    seed: int = 41,
    response: str = "response_mean",
    animal: str = DEFAULT_ANIMAL_COLUMN,
) -> dict[str, Any]:
    """Animal-holdout posterior predictive check.

    For each split, the model is fitted on training animals.  For each held-out
    animal and target, the observed held-out profile is compared with the model
    prediction; the observed correlation must fall inside the 95% posterior
    predictive interval.  Splits are repeated because a single 20% animal split
    is noisy.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    cells_all = cell_summaries(frame, response=response, animal=animal)
    animals = np.asarray(sorted(cells_all["_animal"].unique()), dtype=object)
    targets = np.asarray(sorted(cells_all["_target"].unique()), dtype=object)
    rng = np.random.default_rng(seed)
    selected = rng.choice(targets, size=min(max_targets, targets.size), replace=False)
    records: list[dict[str, Any]] = []
    for split in range(n_splits):
        n_test = max(2, int(round(float(test_fraction) * animals.size)))
        n_test = min(n_test, animals.size - 2)
        test_animals = set(rng.choice(animals, size=n_test, replace=False))
        train_cells = cells_all[~cells_all["_animal"].isin(test_animals)]
        test_cells = cells_all[cells_all["_animal"].isin(test_animals)]
        fit = fit_hierarchical_cells(
            train_cells,
            coverage=coverage,
            n_eval=(1, 22),
            draws=draws,
            grid_resolution=grid_resolution,
            seed=seed + split,
        )
        for target in selected:
            target_key = str(target)
            train_group = train_cells[train_cells["_target"] == target_key]
            data = _normalize_target(
                train_group,
                coverage=coverage,
                min_events=1,
                min_animals=6,
                standardize=True,
            )
            if data is None:
                continue
            draws_matrix = fit.draws_by_target.get(target_key)
            if draws_matrix is None:
                continue
            test_group = test_cells[test_cells["_target"] == target_key]
            if test_group.empty:
                continue
            test_mean = test_group.pivot_table(
                index="_animal", columns="_neuron", values="mean", aggfunc="mean"
            ).reindex(columns=list(data.responders))
            test_var = (
                test_group.assign(v=test_group["var"] / np.maximum(test_group["n"], 1.0))
                .pivot_table(index="_animal", columns="_neuron", values="v", aggfunc="mean")
                .reindex(index=test_mean.index, columns=list(data.responders))
            )
            if test_mean.shape[0] < 2 or test_mean.notna().sum().sum() < 3:
                continue
            test_values = test_mean.to_numpy(dtype=float)
            finite_test = np.isfinite(test_values)
            counts = finite_test.sum(axis=0)
            observed = np.divide(
                np.where(finite_test, test_values, 0.0).sum(axis=0),
                counts,
                out=np.full(counts.shape, np.nan, dtype=float),
                where=counts > 0,
            )
            mask_obs = np.isfinite(observed)
            if mask_obs.sum() < 3:
                continue
            v_test = test_var.to_numpy(dtype=float)
            v_test = np.where(np.isfinite(v_test), v_test, data.nu)
            predictive: list[float] = []
            mu_draws: list[np.ndarray] = []
            for sigma2, tau2 in draws_matrix:
                d = tau2 + data.v
                s = np.where(data.mask, 1.0 / d, 0.0).sum(axis=0)
                precision = 1.0 / sigma2 + s
                linear = np.where(data.mask, data.y / d, 0.0).sum(axis=0)
                mu_mean = linear / precision
                mu_sd = np.sqrt(1.0 / precision)
                mu = rng.normal(mu_mean, mu_sd)
                mu_draws.append(mu)
                simulated = []
                for animal_index in range(test_mean.shape[0]):
                    u = rng.normal(0.0, np.sqrt(tau2), size=data.n_responders)
                    e = rng.normal(0.0, np.sqrt(v_test[animal_index]), size=data.n_responders)
                    simulated.append(mu + u + e)
                simulated_mean = np.nanmean(np.asarray(simulated), axis=0)
                predictive.append(_pearson_rows(mu, simulated_mean, mask_obs))
            finite = np.asarray(predictive, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size < 50:
                continue
            mu_mean = np.nanmean(np.asarray(mu_draws, dtype=float), axis=0)
            observed_corr = _pearson_rows(mu_mean, observed, mask_obs)
            if not np.isfinite(observed_corr):
                continue
            low, high = np.quantile(finite, [0.025, 0.975])
            records.append(
                {
                    "split": split,
                    "target": target_key,
                    "observed_corr": float(observed_corr),
                    "predictive_low": float(low),
                    "predictive_high": float(high),
                    "predictive_median": float(np.median(finite)),
                    "covered": bool(low <= observed_corr <= high),
                    "abs_error": float(abs(np.median(finite) - observed_corr)),
                }
            )
    if not records:
        return {"status": "no_eligible_targets", "n": 0}
    frame_out = pd.DataFrame(records)
    return {
        "status": "ok",
        "n": int(len(frame_out)),
        "n_splits": int(n_splits),
        "coverage": float(frame_out["covered"].mean()),
        "median_abs_error": float(frame_out["abs_error"].median()),
        "records": frame_out.to_dict(orient="records"),
    }


def negative_control(
    frame: pd.DataFrame,
    *,
    n_targets: int = 40,
    coverage: float = 0.8,
    draws: int = 1200,
    grid_resolution: int = 50,
    seed: int = 43,
    response: str = "response_mean",
    animal: str = DEFAULT_ANIMAL_COLUMN,
) -> dict[str, Any]:
    """Profile-permutation negative control.

    Neuron labels are permuted within each animal and target, destroying the
    shared responder profile while preserving the marginal distribution.  A
    credible high reliability would indicate that the model is fitting noise.
    """
    cells_all = cell_summaries(frame, response=response, animal=animal)
    rng = np.random.default_rng(seed)
    permuted: list[pd.DataFrame] = []
    for _target, group in cells_all.groupby("_target", sort=False):
        means = group.pivot_table(index="_animal", columns="_neuron", values="mean", aggfunc="mean")
        variances = group.pivot_table(
            index="_animal", columns="_neuron", values="var", aggfunc="mean"
        ).reindex(index=means.index, columns=means.columns)
        rows: list[dict[str, Any]] = []
        for animal_index in means.index:
            values = means.loc[animal_index].to_numpy(dtype=float)
            finite = np.isfinite(values)
            shuffled = values.copy()
            shuffled[finite] = rng.permutation(values[finite])
            cell_var = variances.loc[animal_index].to_numpy(dtype=float)
            for responder_index, column in enumerate(means.columns):
                if not finite[responder_index]:
                    continue
                rows.append(
                    {
                        "_animal": str(animal_index),
                        "_target": str(_target),
                        "_neuron": str(column),
                        "n": 1,
                        "mean": float(shuffled[responder_index]),
                        "var": float(cell_var[responder_index])
                        if np.isfinite(cell_var[responder_index])
                        else 0.0,
                    }
                )
        permuted.append(pd.DataFrame(rows))
    cells = pd.concat(permuted, ignore_index=True)
    targets = np.asarray(sorted(cells["_target"].unique()), dtype=object)
    selected = set(rng.choice(targets, size=min(n_targets, targets.size), replace=False))
    result = fit_hierarchical_cells(
        cells[cells["_target"].isin(selected)],
        coverage=coverage,
        n_eval=(22,),
        draws=draws,
        grid_resolution=grid_resolution,
        seed=seed,
    )
    values = result.by_target.loc[result.by_target["status"] == "ok", "reliability_n22"]
    values = values[np.isfinite(values)]
    if values.empty:
        return {"status": "no_eligible_targets", "n": 0}
    return {
        "status": "ok",
        "n_targets": int(len(values)),
        "median_reliability_n22": float(values.median()),
        "q95_reliability_n22": float(np.quantile(values, 0.95)),
        "max_reliability_n22": float(values.max()),
    }


def prior_sensitivity(
    frame: pd.DataFrame,
    *,
    coverage: float = 0.8,
    prior_scales: tuple[float, ...] = (0.5, 1.0, 2.0),
    draws: int = 1500,
    grid_resolution: int = 50,
    seed: int = 47,
    response: str = "response_mean",
    animal: str = DEFAULT_ANIMAL_COLUMN,
) -> dict[str, Any]:
    """Refit the real data under different HalfNormal prior scales."""
    cells = cell_summaries(frame, response=response, animal=animal)
    output: dict[str, Any] = {"coverage": coverage, "fits": {}}
    for scale in prior_scales:
        result = fit_hierarchical_cells(
            cells,
            coverage=coverage,
            n_eval=(1, 22),
            draws=draws,
            grid_resolution=grid_resolution,
            seed=seed,
            prior_scale=float(scale),
        )
        output["fits"][f"{scale:g}"] = {
            "status": result.status,
            "pooled_reliability_median": result.pooled_reliability_median,
            "single_animal_reliability_median": result.single_animal_reliability_median,
            "pooled_ceiling_median": result.pooled_ceiling_median,
            "n_ok": int(result.by_target["status"].eq("ok").sum()),
        }
    return output


__all__ = [
    "heldout_predictive_check",
    "negative_control",
    "prepare_templates",
    "prior_sensitivity",
    "simulation_based_calibration",
]
