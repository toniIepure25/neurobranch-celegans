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
    TargetData,
    _normalize_target,
    _reliability_draws,
    fit_hierarchical_cells,
    fit_hierarchical_target,
    transform_target,
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
) -> list[TargetData]:
    """Eligible real-target templates used to preserve missingness patterns."""
    cells = cell_summaries(frame, response=response, animal=animal)
    templates: list[TargetData] = []
    for _target, group in cells.groupby("_target", sort=False):
        data = _normalize_target(
            group,
            coverage=coverage,
            min_events=min_events,
            min_animals=min_animals,
            standardize=True,
            center_animals=False,
        )
        if data is not None:
            templates.append(data)
    return templates


def _simulate_template(
    data: TargetData,
    *,
    rng: np.random.Generator,
    sigma: float,
    tau: float,
    target_id: str,
) -> TargetData:
    """Simulate one uncentered target on the template's standardized scale."""
    mu = rng.normal(0.0, sigma, size=data.n_responders)
    interaction = rng.normal(0.0, tau, size=(data.n_animals, data.n_responders))
    noise = rng.normal(0.0, np.sqrt(data.v))
    y = np.where(data.mask, mu[None, :] + interaction + noise, np.nan)
    return TargetData(
        target=target_id,
        y=y,
        v=data.v.copy(),
        mask=data.mask.copy(),
        n_animals=data.n_animals,
        n_responders=data.n_responders,
        coverage=data.coverage,
        nu=data.nu,
        responders=data.responders,
    )


def _aligned_truth(sigma: float, tau: float, nu: float, n_responders: int, n: int) -> float:
    """Reliability truth after the fitter's animal centering.

    Row centering removes the responder mean of each animal.  For a Gaussian
    profile of ``R`` responders, the centering factor is ``1 - 1/R`` for the
    between-target and interaction variances.  This is the deterministic
    counterpart of simulating the same centered process; using the uncentered
    formula would compare the posterior to a different estimand.
    """
    if n_responders < 2:
        raise ValueError("aligned truth requires at least two responders")
    factor = max(1.0 - 1.0 / float(n_responders), 0.0)
    sigma_centered = float(sigma) ** 2 * factor
    tau_centered = float(tau) ** 2 * factor
    denominator = sigma_centered + (tau_centered + float(nu)) / float(n)
    return float(np.clip(sigma_centered / denominator, 0.0, 1.0)) if denominator > 0 else 0.0


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / float(total)
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def simulation_based_calibration(
    templates: list[TargetData],
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
    mask, sampling variances and standardized scale.  The simulated target is
    passed through the same centering as the fit, and the truth uses the
    centered effective variance rather than the uncentered closed form.
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
    edge_failures = 0
    edge_masses: list[float] = []
    for rep in range(n_reps):
        for target_index in range(targets_per_rep):
            template = templates[int(rng.integers(0, len(templates)))]
            sigma = float(abs(rng.normal(0.0, 1.0)))
            tau = float(abs(rng.normal(0.0, 1.0)))
            target_id = f"sbc-{rep}-{target_index}"
            simulated = _simulate_template(
                template,
                rng=rng,
                sigma=sigma,
                tau=tau,
                target_id=target_id,
            )
            simulated = transform_target(simulated, center_animals=True, standardize=False)
            fit = fit_hierarchical_target(
                simulated,
                n_eval=n_eval,
                draws=draws,
                grid_resolution=grid_resolution,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            if fit.draws is None or fit.status != "ok":
                failures += 1
                continue
            edge_mass = float(fit.row["grid_edge_mass"])
            edge_masses.append(edge_mass)
            if edge_mass >= 0.05:
                edge_failures += 1
            reliability = _reliability_draws(
                fit.draws[:, 0], fit.draws[:, 1], simulated.nu, n_eval
            )
            for n in n_eval:
                truth = _aligned_truth(sigma, tau, simulated.nu, simulated.n_responders, n)
                values = reliability[n][np.isfinite(reliability[n])]
                if values.size < 50:
                    failures += 1
                    continue
                below = float(np.sum(values < truth))
                equal = float(np.sum(values == truth))
                ranks[n].append((below + 0.5 * equal) / float(values.size))
                low, high = np.quantile(values, [0.025, 0.975])
                covered[n].append(bool(low <= truth <= high))
                widths[n].append(float(high - low))
    output: dict[str, Any] = {
        "n_reps": int(n_reps),
        "targets_per_rep": int(targets_per_rep),
        "n_fits": int(n_reps * targets_per_rep),
        "n_failures": int(failures),
        "n_edge_failures": int(edge_failures),
        "max_edge_mass": float(max(edge_masses)) if edge_masses else None,
        "mean_edge_mass": float(np.mean(edge_masses)) if edge_masses else None,
        "truth_method": "centered effective variance; same centering as fit",
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
        successes = int(np.sum(covered[n]))
        low, high = _wilson_interval(successes, len(covered[n]))
        output["parameters"][f"n{n}"] = {
            "status": "ok",
            "n": int(values.size),
            "coverage": float(np.mean(covered[n])),
            "coverage_ci_low": low,
            "coverage_ci_high": high,
            "mean_width": float(np.mean(widths[n])),
            "mean_rank": float(values.mean()),
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "ranks": values.tolist(),
        }
    return output


def _center_profile(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Center a profile over observed responders, preserving the mask."""
    out = np.asarray(values, dtype=float).copy()
    keep = mask & np.isfinite(out)
    if int(keep.sum()) == 0:
        return out
    out[keep] = out[keep] - float(np.mean(out[keep]))
    return out


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
            row_means = np.nanmean(np.where(finite_test, test_values, np.nan), axis=1, keepdims=True)
            test_values = test_values - row_means
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
                mu_centered = _center_profile(mu, mask_obs)
                simulated = []
                for animal_index in range(test_mean.shape[0]):
                    u = rng.normal(0.0, np.sqrt(tau2), size=data.n_responders)
                    e = rng.normal(0.0, np.sqrt(v_test[animal_index]), size=data.n_responders)
                    simulated.append(mu + u + e)
                simulated_mean = np.nanmean(np.asarray(simulated), axis=0)
                simulated_centered = _center_profile(simulated_mean, mask_obs)
                predictive.append(_pearson_rows(mu_centered, simulated_centered, mask_obs))
            finite = np.asarray(predictive, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size < 50:
                continue
            mu_mean = np.nanmean(np.asarray(mu_draws, dtype=float), axis=0)
            mu_mean = _center_profile(mu_mean, mask_obs)
            observed = _center_profile(observed, mask_obs)
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
        return {
            "status": "no_eligible_targets",
            "n_requested": int(len(selected)),
            "n_targets": 0,
            "n_excluded": int(len(selected)),
        }
    return {
        "status": "ok",
        "n_requested": int(len(selected)),
        "n_targets": int(len(values)),
        "n_excluded": int(len(selected) - len(values)),
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
