"""Calibration and adversarial checks for the Phase 13 measurement models."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import stats

from neurobranch.benchmark.hierarchical import TargetData, _reliability_draws
from neurobranch.benchmark.hierarchical_calibration import _aligned_truth, _wilson_interval, prepare_templates
from neurobranch.benchmark.measurement import (
    MeasurementConfig,
    fit_measurement_reliability,
    fit_measurement_target,
)
from neurobranch.benchmark.variance import DEFAULT_ANIMAL_COLUMN


def _projection_nu(data: TargetData) -> float:
    mask = np.asarray(data.mask, dtype=bool)
    observed = mask.sum(axis=1).astype(float)
    factor = np.clip(1.0 - 1.0 / np.maximum(observed, 1.0), 0.0, 1.0)
    v = np.asarray(data.v, dtype=float) * factor[:, None]
    return float(np.mean(v[mask]))


def _simulate_template(
    template: TargetData,
    *,
    rng: np.random.Generator,
    sigma: float,
    tau: float,
    omega: float,
    model: str,
    target_id: str,
) -> TargetData:
    """Simulate one target under the candidate's generative model."""
    mask = np.asarray(template.mask, dtype=bool)
    n_responders = int(template.n_responders)
    n_animals = int(template.n_animals)
    mu = rng.normal(0.0, sigma, size=n_responders)
    interaction = rng.normal(0.0, tau, size=(n_animals, n_responders))
    cell_var: np.ndarray | None
    if model == "m2_event_variance":
        if template.n_events is None:
            raise ValueError("event-variance SBC requires event counts")
        n = np.asarray(template.n_events, dtype=float)
        v = np.where(mask, omega**2 / np.maximum(n, 1.0), 1.0)
        cell_var = np.zeros_like(v)
        variance_cells = mask & (n >= 2.0)
        if bool(variance_cells.any()):
            k = n[variance_cells] - 1.0
            cell_var[variance_cells] = omega**2 * rng.chisquare(k) / k
    else:
        v = np.asarray(template.v, dtype=float).copy()
        cell_var = None if template.cell_var is None else np.asarray(template.cell_var, dtype=float).copy()
    noise = rng.normal(0.0, np.sqrt(v))
    y = np.where(mask, mu[None, :] + interaction + noise, np.nan)
    return TargetData(
        target=target_id,
        y=y,
        v=v,
        mask=mask,
        n_animals=n_animals,
        n_responders=n_responders,
        coverage=template.coverage,
        nu=float(np.mean(v[mask])),
        responders=template.responders,
        n_variance_floored=0,
        n_events=None if template.n_events is None else np.asarray(template.n_events, dtype=float).copy(),
        cell_var=cell_var,
    )


def simulation_based_calibration(
    templates: list[TargetData],
    *,
    model: str,
    n_reps: int = 200,
    targets_per_rep: int = 3,
    n_eval: tuple[int, ...] = (1, 22),
    draws: int = 1500,
    grid_resolution: int = 50,
    event_grid: int = 100,
    seed: int = 31,
) -> dict[str, Any]:
    """Rank-based SBC for a Phase 13 candidate."""
    if not templates:
        raise ValueError("no templates")
    if n_reps < 10:
        raise ValueError("n_reps must be >= 10")
    rng = np.random.default_rng(seed)
    ranks: dict[int, list[float]] = {n: [] for n in n_eval}
    covered: dict[int, list[bool]] = {n: [] for n in n_eval}
    widths: dict[int, list[float]] = {n: [] for n in n_eval}
    one_event_ranks: list[float] = []
    one_event_covered: list[bool] = []
    omega_covered: list[bool] = []
    failed_fits = 0
    edge_failures = 0
    edge_masses: list[float] = []
    for rep in range(n_reps):
        for target_index in range(targets_per_rep):
            template = templates[int(rng.integers(0, len(templates)))]
            sigma = float(abs(rng.normal(0.0, 1.0)))
            tau = float(abs(rng.normal(0.0, 1.0)))
            omega = float(abs(rng.normal(0.0, 1.0)))
            simulated = _simulate_template(
                template,
                rng=rng,
                sigma=sigma,
                tau=tau,
                omega=omega,
                model=model,
                target_id=f"sbc-{rep}-{target_index}",
            )
            config = MeasurementConfig(
                model=model,
                n_eval=n_eval,
                draws=draws,
                grid_resolution=grid_resolution,
                event_grid=event_grid,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            fit = fit_measurement_target(simulated, config=config)
            if fit.draws is None or fit.status != "ok":
                failed_fits += 1
                continue
            edge_mass = float(fit.row["grid_edge_mass"])
            edge_masses.append(edge_mass)
            if edge_mass >= 0.05:
                edge_failures += 1
            # The fit's sigma/tau are on the centered scale.  Use the same
            # projection-scaled nu for the truth and the posterior reliability.
            centered_v = np.asarray(simulated.v, dtype=float)
            mask = np.asarray(simulated.mask, dtype=bool)
            observed = mask.sum(axis=1).astype(float)
            factor = np.clip(1.0 - 1.0 / np.maximum(observed, 1.0), 0.0, 1.0)
            nu_centered = float(np.mean((centered_v * factor[:, None])[mask]))
            reliability = _reliability_draws(fit.draws[:, 0], fit.draws[:, 1], nu_centered, n_eval)
            n_one_fraction = (
                float(np.mean(np.asarray(simulated.n_events)[mask] < 2.0))
                if simulated.n_events is not None
                else 0.0
            )
            for n in n_eval:
                truth = _aligned_truth(sigma, tau, nu_centered, simulated.n_responders, n)
                values = reliability[n][np.isfinite(reliability[n])]
                if values.size < 50:
                    continue
                below = float(np.sum(values < truth))
                equal = float(np.sum(values == truth))
                rank = (below + 0.5 * equal) / float(values.size)
                ranks[n].append(rank)
                low, high = np.quantile(values, [0.025, 0.975])
                is_covered = bool(low <= truth <= high)
                covered[n].append(is_covered)
                widths[n].append(float(high - low))
                if n_one_fraction >= 0.5:
                    one_event_ranks.append(rank)
                    one_event_covered.append(is_covered)
            if model == "m2_event_variance" and np.isfinite(omega):
                low = float(fit.row.get("event_variance_ci_low", np.nan))
                high = float(fit.row.get("event_variance_ci_high", np.nan))
                if np.isfinite(low) and np.isfinite(high):
                    omega_covered.append(bool(low <= omega**2 <= high))
    output: dict[str, Any] = {
        "model": model,
        "n_reps": int(n_reps),
        "targets_per_rep": int(targets_per_rep),
        "n_fits": int(n_reps * targets_per_rep),
        "n_failed_fits": int(failed_fits),
        "n_edge_failures": int(edge_failures),
        "max_edge_mass": float(max(edge_masses)) if edge_masses else None,
        "mean_edge_mass": float(np.mean(edge_masses)) if edge_masses else None,
        "truth_method": "projection-scaled variance; same centering as fit",
        "parameters": {},
    }
    for n in n_eval:
        values = np.asarray(ranks[n], dtype=float)
        if values.size < 10:
            output["parameters"][f"n{n}"] = {"status": "insufficient_draws", "n": int(values.size)}
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
    if one_event_ranks:
        values = np.asarray(one_event_ranks, dtype=float)
        output["one_event_stratum"] = {
            "n": int(values.size),
            "coverage": float(np.mean(one_event_covered)),
            "mean_rank": float(values.mean()),
            "ks_pvalue": float(stats.kstest(values, "uniform").pvalue),
        }
    if model == "m2_event_variance":
        output["event_variance_coverage"] = float(np.mean(omega_covered)) if omega_covered else float("nan")
        output["event_variance_n"] = int(len(omega_covered))
    return output


def _permuted_template(
    template: TargetData, rng: np.random.Generator, *, sign_flip: bool = False
) -> TargetData:
    y = np.asarray(template.y, dtype=float).copy()
    mask = np.asarray(template.mask, dtype=bool)
    for animal in range(y.shape[0]):
        finite = mask[animal] & np.isfinite(y[animal])
        if not finite.any():
            continue
        if sign_flip:
            y[animal, finite] *= rng.choice([-1.0, 1.0])
        else:
            y[animal, finite] = rng.permutation(y[animal, finite])
    return replace(template, y=y)


def permutation_control(
    templates: list[TargetData],
    *,
    model: str,
    n_targets: int = 100,
    draws: int = 1200,
    grid_resolution: int = 50,
    event_grid: int = 100,
    seed: int = 43,
    sign_flip: bool = False,
) -> dict[str, Any]:
    """Within-animal responder permutation or sign-flip null."""
    if not templates:
        raise ValueError("no templates")
    rng = np.random.default_rng(seed)
    selected = list(rng.choice(len(templates), size=min(n_targets, len(templates)), replace=False))
    rows: list[dict[str, Any]] = []
    failures = 0
    for index in selected:
        template = templates[int(index)]
        null_template = _permuted_template(template, rng, sign_flip=sign_flip)
        config = MeasurementConfig(
            model=model,
            n_eval=(22,),
            draws=draws,
            grid_resolution=grid_resolution,
            event_grid=event_grid,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        fit = fit_measurement_target(null_template, config=config)
        if fit.status != "ok":
            failures += 1
            continue
        rows.append(
            {
                "target": template.target,
                "reliability_n22": float(fit.row["reliability_n22"]),
                "grid_edge_mass": float(fit.row["grid_edge_mass"]),
            }
        )
    values = np.asarray([row["reliability_n22"] for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return {
        "model": model,
        "control": "sign_flip" if sign_flip else "responder_permutation",
        "n_requested": int(len(selected)),
        "n_eligible": int(values.size),
        "n_failed": int(failures),
        "median_reliability_n22": float(np.median(values)) if values.size else float("nan"),
        "q95_reliability_n22": float(np.quantile(values, 0.95)) if values.size else float("nan"),
        "max_reliability_n22": float(values.max()) if values.size else float("nan"),
        "values": values.tolist(),
    }


def prior_sensitivity(
    frame: pd.DataFrame,
    *,
    model: str,
    coverage: float = 0.8,
    draws: int = 1200,
    grid_resolution: int = 50,
    seed: int = 47,
    prior_scales: tuple[float, ...] = (0.5, 1.0, 2.0),
) -> dict[str, Any]:
    """Range of pooled R(22) across pre-declared prior scales."""
    output: dict[str, Any] = {"model": model, "fits": {}}
    values: list[float] = []
    for scale in prior_scales:
        result = fit_measurement_reliability(
            frame,
            model=model,
            coverage=coverage,
            n_eval=(22,),
            draws=draws,
            grid_resolution=grid_resolution,
            seed=seed,
            prior_scale=float(scale),
        )
        value = cast(float, result.summary.get("pooled_reliability_median", float("nan")))
        values.append(value)
        output["fits"][f"{scale:g}"] = {
            "status": result.status,
            "pooled_reliability_median": value,
            "n_ok": cast(int, result.summary.get("n_ok", 0)),
        }
    finite = [value for value in values if np.isfinite(value)]
    output["range"] = float(max(finite) - min(finite)) if finite else float("nan")
    return output


def prepare_measurement_templates(
    frame: pd.DataFrame,
    *,
    coverage: float = 0.8,
    response: str = "response_mean",
    animal: str = DEFAULT_ANIMAL_COLUMN,
    min_events: int = 1,
    min_animals: int = 6,
) -> list[TargetData]:
    """Real-target templates preserving masks, event counts and cell variances."""
    return prepare_templates(
        frame,
        coverage=coverage,
        response=response,
        animal=animal,
        min_events=min_events,
        min_animals=min_animals,
    )


__all__ = [
    "permutation_control",
    "prepare_measurement_templates",
    "prior_sensitivity",
    "simulation_based_calibration",
]
