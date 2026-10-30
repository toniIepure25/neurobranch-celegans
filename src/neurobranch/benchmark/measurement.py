"""Projection-consistent reliability models for Phase 13.

The Phase 12R model centers each animal's responder profile but leaves the
sampling variance ``v`` unchanged.  For equal variances the centered noise
variance is ``v * (1 - 1/R)``; treating the post-centering noise as
``diag(v)`` overstates it.  This module implements the pre-registered Phase 13
candidates:

``m0_reference``
    The Phase 12R fit, retained only as a historical control.

``m1_projection_scaled``
    Center ``y`` and scale each cell's known ``v`` by the first-order factor
    ``1 - 1/R_a``.  This corrects the omitted noise scaling.  Off-diagonal
    covariance induced by unequal variances is not modelled; the limitation is
    reported, not hidden.

``m2_event_variance``
    Replace the ``1e-8`` floor with a marginal event-level variance model.  The
    event variance is estimated from the scaled chi-square likelihood of the
    observed within-cell sample variances, then used as
    ``v_ar = omega_e^2 / n_ar`` for the cell means.  The same projection scaling
    is applied afterwards.

The event-variance model is deliberately marginal and homogeneous.  It removes
the floor and restores a measured variance to one-event cells, but it does not
model cell-level variance heterogeneity.  That limitation is part of the claim
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy import special
from scipy.special import logsumexp

from neurobranch.benchmark.hierarchical import (
    MIN_TARGET_ANIMALS,
    HierarchicalResult,
    TargetData,
    TargetFit,
    _grid_posterior,
    _normalize_target_with_reason,
    _reliability_draws,
    _validate_target,
    fit_hierarchical_target,
    posterior_ceiling,
    transform_target,
)
from neurobranch.benchmark.variance import DEFAULT_ANIMAL_COLUMN, cell_summaries

MODELS = ("m0_reference", "m1_projection_scaled", "m2_event_variance", "m4_spike_slab")


@dataclass(frozen=True)
class MeasurementConfig:
    """Frozen Phase 13 configuration for one target."""

    model: str = "m2_event_variance"
    n_eval: tuple[int, ...] = (1, 22)
    draws: int = 2500
    grid_resolution: int = 80
    seed: int = 12
    prior_scale: float = 1.0
    event_prior_scale: float = 1.0
    event_grid: int = 200
    pi0: float = 0.5
    slab_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"unknown measurement model {self.model!r}")
        if self.draws < 100:
            raise ValueError("draws >= 100 is required")
        if self.grid_resolution < 20:
            raise ValueError("grid_resolution >= 20 is required")
        if self.event_grid < 40:
            raise ValueError("event_grid >= 40 is required")
        if self.prior_scale <= 0 or self.event_prior_scale <= 0:
            raise ValueError("prior scales must be positive")
        if not 0.0 < self.pi0 < 1.0:
            raise ValueError("pi0 must be strictly between zero and one")
        if self.slab_scale <= 0:
            raise ValueError("slab_scale must be positive")


@dataclass(frozen=True)
class EventVariancePosterior:
    """Marginal posterior for the homogeneous event variance scale."""

    omega2_median: float
    omega2_mean: float
    omega2_ci_low: float
    omega2_ci_high: float
    n_variance_cells: int
    n_zero_variance_cells: int
    n_one_event_cells: int
    grid_n_eff: float


@dataclass(frozen=True)
class SpikeSlabPosterior:
    """Posterior summary for the exact zero/spike-and-slab model M4.

    ``p_zero`` is the posterior probability that the target has exactly zero
    shared signal, i.e. ``sigma = 0``.  The draws include that point mass, so a
    reliability draw is exactly zero whenever the spike is selected.
    """

    p_zero: float
    log_evidence: float
    slab_log_evidence: float
    spike_log_evidence: float
    slab_edge_mass: float
    slab_n_eff: float
    sigma2: np.ndarray
    tau2: np.ndarray


def _spike_slab_grid(
    data: TargetData,
    *,
    resolution: int,
    draws: int,
    seed: int,
    prior_scale: float,
    pi0: float,
) -> SpikeSlabPosterior:
    """Exact point mass at ``sigma=0`` plus the Phase 13 slab.

    The slab is the existing ``(log sigma, log tau)`` grid with HalfNormal
    priors.  The spike has ``sigma=0`` and a HalfNormal prior on ``tau`` only.
    Both components share the same Gaussian likelihood and projection-scaled
    sampling variances.
    """
    y, v, mask = data.y, data.v, data.mask
    log_sigma = np.linspace(-6.0, float(np.log(20.0)), int(resolution))
    log_tau = np.linspace(-6.0, float(np.log(20.0)), int(resolution))
    sigma2_grid = np.exp(2.0 * log_sigma)
    tau2_grid = np.exp(2.0 * log_tau)

    slab = np.empty((log_sigma.size, log_tau.size), dtype=float)
    spike_tau = np.empty(log_tau.size, dtype=float)
    for j, tau2 in enumerate(tau2_grid):
        d = tau2 + v
        inv = np.where(mask, 1.0 / d, 0.0)
        q = np.where(mask, y * y * inv, 0.0).sum(axis=0)
        linear = np.where(mask, y * inv, 0.0).sum(axis=0)
        log_det_base = np.where(mask, np.log(d), 0.0).sum(axis=0)
        denominator = 1.0 + sigma2_grid[:, None] * inv.sum(axis=0)[None, :]
        slab[:, j] = -0.5 * (
            float(log_det_base.sum())
            + np.log(denominator).sum(axis=1)
            + float(q.sum())
            - sigma2_grid * np.sum(linear[None, :] ** 2 / denominator, axis=1)
        ) - 0.5 * (sigma2_grid + tau2) / (prior_scale**2) + log_sigma + float(log_tau[j])
        # sigma = 0: mu is exactly zero, so the likelihood has no linear term.
        spike_tau[j] = (
            -0.5 * (float(log_det_base.sum()) + float(q.sum()))
            - 0.5 * tau2 / (prior_scale**2)
            + float(log_tau[j])
        )

    finite_slab = np.isfinite(slab)
    if not finite_slab.any() or not np.isfinite(spike_tau).any():
        nan = np.array([np.nan])
        return SpikeSlabPosterior(
            p_zero=float("nan"),
            log_evidence=float("nan"),
            slab_log_evidence=float("nan"),
            spike_log_evidence=float("nan"),
            slab_edge_mass=1.0,
            slab_n_eff=0.0,
            sigma2=nan,
            tau2=nan,
        )

    slab_log_evidence = float(logsumexp(slab[finite_slab]))
    spike_log_evidence = float(logsumexp(spike_tau[np.isfinite(spike_tau)]))
    log_slab_joint = float(np.log1p(-pi0)) + slab_log_evidence
    log_spike_joint = float(np.log(pi0)) + spike_log_evidence
    log_evidence = float(np.logaddexp(log_slab_joint, log_spike_joint))
    p_zero = float(np.exp(log_spike_joint - log_evidence))
    p_zero = float(np.clip(p_zero, 0.0, 1.0))

    slab_weights = np.exp(slab - slab_log_evidence)
    slab_weights = np.where(finite_slab, slab_weights, 0.0)
    slab_weights /= slab_weights.sum()
    edge = float(
        slab_weights[0, :].sum()
        + slab_weights[-1, :].sum()
        + slab_weights[:, 0].sum()
        + slab_weights[:, -1].sum()
    )
    n_eff = float(1.0 / np.sum(slab_weights**2))

    spike_weights = np.exp(spike_tau - spike_log_evidence)
    spike_weights = np.where(np.isfinite(spike_tau), spike_weights, 0.0)
    spike_weights /= spike_weights.sum()

    rng = np.random.default_rng(seed)
    n_spike = int(rng.binomial(int(draws), p_zero))
    n_slab = int(draws) - n_spike
    sigma_parts: list[np.ndarray] = []
    tau_parts: list[np.ndarray] = []
    step_sigma = float(log_sigma[1] - log_sigma[0])
    step_tau = float(log_tau[1] - log_tau[0])
    if n_spike:
        cols = rng.choice(spike_weights.size, size=n_spike, p=spike_weights)
        tau_spike = np.exp(log_tau[cols] + rng.uniform(-0.5, 0.5, size=cols.size) * step_tau)
        sigma_parts.append(np.zeros(n_spike, dtype=float))
        tau_parts.append(tau_spike**2)
    if n_slab:
        flat = rng.choice(slab_weights.size, size=n_slab, p=slab_weights.ravel())
        rows, cols = np.divmod(flat, slab_weights.shape[1])
        sigma_slab = np.exp(log_sigma[rows] + rng.uniform(-0.5, 0.5, size=rows.size) * step_sigma)
        tau_slab = np.exp(log_tau[cols] + rng.uniform(-0.5, 0.5, size=cols.size) * step_tau)
        sigma_parts.append(sigma_slab**2)
        tau_parts.append(tau_slab**2)
    sigma2 = np.concatenate(sigma_parts) if sigma_parts else np.array([np.nan])
    tau2 = np.concatenate(tau_parts) if tau_parts else np.array([np.nan])
    order = rng.permutation(sigma2.size)
    return SpikeSlabPosterior(
        p_zero=p_zero,
        log_evidence=log_evidence,
        slab_log_evidence=float(log_slab_joint),
        spike_log_evidence=float(log_spike_joint),
        slab_edge_mass=edge,
        slab_n_eff=n_eff,
        sigma2=sigma2[order],
        tau2=tau2[order],
    )


def _drop_single_responder_animals(data: TargetData) -> tuple[TargetData, int]:
    """Remove animals with fewer than two observed responders.

    A one-responder animal has no within-animal profile contrast, so centering
    sets its row to zero and it cannot inform ``sigma`` or ``tau``.  Keeping it
    would change ``n_animals`` without adding information.
    """
    mask = np.asarray(data.mask, dtype=bool)
    observed = mask.sum(axis=1)
    keep = observed >= 2
    dropped = int((~keep).sum())
    if dropped == 0:
        return data, 0
    if int(keep.sum()) < MIN_TARGET_ANIMALS:
        raise ValueError("fewer than two animals with at least two responders")
    n_events = None if data.n_events is None else np.asarray(data.n_events, dtype=float)[keep]
    cell_var = None if data.cell_var is None else np.asarray(data.cell_var, dtype=float)[keep]
    y = np.asarray(data.y, dtype=float)[keep]
    v = np.asarray(data.v, dtype=float)[keep]
    mask = mask[keep]
    return (
        replace(
            data,
            y=y,
            v=v,
            mask=mask,
            n_animals=int(y.shape[0]),
            nu=float(np.mean(v[mask])),
            n_events=n_events,
            cell_var=cell_var,
        ),
        dropped,
    )


def _center_projection_scaled(data: TargetData) -> tuple[TargetData, np.ndarray]:
    """Center profiles and apply the first-order noise projection factor."""
    y = np.asarray(data.y, dtype=float).copy()
    v = np.asarray(data.v, dtype=float).copy()
    mask = np.asarray(data.mask, dtype=bool)
    observed = mask.sum(axis=1).astype(float)
    factor = np.clip(1.0 - 1.0 / np.maximum(observed, 1.0), 0.0, 1.0)
    row_mean = np.nanmean(np.where(mask, y, np.nan), axis=1, keepdims=True)
    y = y - row_mean
    v = v * factor[:, None]
    centered = replace(
        data,
        y=y,
        v=v,
        nu=float(np.mean(v[mask])),
    )
    return centered, factor


def estimate_event_variance(
    data: TargetData,
    *,
    prior_scale: float = 1.0,
    grid_size: int = 200,
) -> EventVariancePosterior:
    """Marginal posterior for a homogeneous event-level variance.

    For a cell with ``n >= 2`` events and sample variance ``s^2``::

        (n - 1) * s^2 / omega^2 ~ ChiSquare(n - 1)

    Cells with one event contribute no sample-variance likelihood.  They are
    retained in the mean likelihood with ``omega^2 / n`` rather than floored.
    """
    if data.n_events is None or data.cell_var is None:
        raise ValueError("event-variance model requires n_events and cell_var")
    n = np.asarray(data.n_events, dtype=float)
    s2 = np.asarray(data.cell_var, dtype=float)
    mask = np.asarray(data.mask, dtype=bool)
    variance_mask = mask & (n >= 2.0) & (s2 > 0.0)
    n_one = int((mask & (n < 2.0)).sum())
    n_zero = int((mask & (n >= 2.0) & (s2 <= 0.0)).sum())
    log_omega = np.linspace(np.log(1e-4), np.log(10.0), int(grid_size))
    omega2 = np.exp(2.0 * log_omega)
    log_likelihood = np.zeros_like(omega2)
    if bool(variance_mask.any()):
        k = n[variance_mask] - 1.0
        values = s2[variance_mask]
        # Vectorized over the omega grid.  The constant terms are retained so
        # the reported density is a real likelihood, not an unnormalized score.
        log_likelihood = np.sum(
            0.5 * k[:, None] * np.log(k[:, None] / (2.0 * omega2[None, :]))
            - special.gammaln(0.5 * k)[:, None]
            + (0.5 * k[:, None] - 1.0) * np.log(values)[:, None]
            - k[:, None] * values[:, None] / (2.0 * omega2[None, :]),
            axis=0,
        )
    # HalfNormal(prior_scale) on omega; the log grid contributes a Jacobian.
    log_prior = -0.5 * (omega2 / (prior_scale**2)) + log_omega
    log_post = log_likelihood + log_prior
    finite = np.isfinite(log_post)
    if not finite.any():
        return EventVariancePosterior(
            omega2_median=float("nan"),
            omega2_mean=float("nan"),
            omega2_ci_low=float("nan"),
            omega2_ci_high=float("nan"),
            n_variance_cells=int(variance_mask.sum()),
            n_zero_variance_cells=n_zero,
            n_one_event_cells=n_one,
            grid_n_eff=0.0,
        )
    shifted = np.where(finite, log_post - np.max(log_post[finite]), -np.inf)
    weights = np.exp(shifted)
    weights /= weights.sum()
    cdf = np.cumsum(weights)
    median = float(omega2[int(np.searchsorted(cdf, 0.5))])
    mean = float(np.sum(weights * omega2))
    low = float(omega2[int(np.searchsorted(cdf, 0.025))])
    high = float(omega2[int(np.searchsorted(cdf, 0.975))])
    return EventVariancePosterior(
        omega2_median=median,
        omega2_mean=mean,
        omega2_ci_low=low,
        omega2_ci_high=high,
        n_variance_cells=int(variance_mask.sum()),
        n_zero_variance_cells=n_zero,
        n_one_event_cells=n_one,
        grid_n_eff=float(1.0 / np.sum(weights**2)),
    )


def _event_variance_data(
    data: TargetData,
    *,
    omega2: float,
) -> TargetData:
    if data.n_events is None:
        raise ValueError("event-variance model requires event counts")
    n = np.asarray(data.n_events, dtype=float)
    v = np.where(data.mask, omega2 / np.maximum(n, 1.0), 1.0)
    return replace(data, v=v, nu=float(np.mean(v[data.mask])), n_variance_floored=0)


def _fit_grid(
    data: TargetData,
    *,
    config: MeasurementConfig,
    seed: int,
) -> TargetFit:
    _validate_target(data)
    fit = _grid_posterior(
        data,
        resolution=config.grid_resolution,
        draws=config.draws,
        seed=seed,
        prior_scale=config.prior_scale,
    )
    row: dict[str, Any] = {
        "target": data.target,
        "model": config.model,
        "status": "ok",
        "reason": "",
        "coverage": float(data.coverage),
        "n_animals": data.n_animals,
        "n_responders": data.n_responders,
        "n_variance_floored": int(data.n_variance_floored),
        "n_observed_cells": int(data.mask.sum()),
        "mean_sampling_var": data.nu,
        "grid_edge_mass": float(fit.edge_mass),
        "grid_n_eff": float(fit.n_eff),
        "n_draws": int(fit.sigma2.size),
    }
    if fit.sigma2.size == 0 or not np.isfinite(fit.sigma2).any() or not np.isfinite(fit.tau2).any():
        row.update({"status": "fit_failed", "reason": "no_finite_posterior"})
        return TargetFit(target=data.target, status="fit_failed", row=row, draws=None)
    sigma2 = fit.sigma2
    tau2 = fit.tau2
    reliability = _reliability_draws(sigma2, tau2, data.nu, config.n_eval)
    row["sigma2_median"] = float(np.median(sigma2))
    row["tau2_median"] = float(np.median(tau2))
    for n in config.n_eval:
        values = reliability[n]
        median_value = float(np.nanmedian(values))
        row[f"reliability_n{n}"] = median_value
        row[f"reliability_ci_low_n{n}"] = float(np.nanquantile(values, 0.025))
        row[f"reliability_ci_high_n{n}"] = float(np.nanquantile(values, 0.975))
        row[f"ceiling_n{n}"] = posterior_ceiling(median_value)
    return TargetFit(
        target=data.target,
        status="ok",
        row=row,
        draws=np.column_stack([sigma2, tau2]),
    )


def _fit_spike_slab(
    data: TargetData,
    *,
    config: MeasurementConfig,
    seed: int,
) -> TargetFit:
    _validate_target(data)
    fit = _spike_slab_grid(
        data,
        resolution=config.grid_resolution,
        draws=config.draws,
        seed=seed,
        prior_scale=config.slab_scale,
        pi0=config.pi0,
    )
    row: dict[str, Any] = {
        "target": data.target,
        "model": config.model,
        "status": "ok",
        "reason": "",
        "coverage": float(data.coverage),
        "n_animals": data.n_animals,
        "n_responders": data.n_responders,
        "n_variance_floored": int(data.n_variance_floored),
        "n_observed_cells": int(data.mask.sum()),
        "mean_sampling_var": data.nu,
        "grid_edge_mass": float(fit.slab_edge_mass),
        "grid_n_eff": float(fit.slab_n_eff),
        "n_draws": int(fit.sigma2.size),
        "p_zero": float(fit.p_zero),
        "spike_probability": float(fit.p_zero),
        "log_evidence": float(fit.log_evidence),
        "slab_log_evidence": float(fit.slab_log_evidence),
        "spike_log_evidence": float(fit.spike_log_evidence),
        "slab_edge_mass": float(fit.slab_edge_mass),
        "slab_n_eff": float(fit.slab_n_eff),
    }
    if (
        fit.sigma2.size == 0
        or not np.isfinite(fit.sigma2).any()
        or not np.isfinite(fit.tau2).any()
        or not np.isfinite(fit.p_zero)
    ):
        row.update({"status": "fit_failed", "reason": "no_finite_posterior"})
        return TargetFit(target=data.target, status="fit_failed", row=row, draws=None)
    sigma2 = fit.sigma2
    tau2 = fit.tau2
    reliability = _reliability_draws(sigma2, tau2, data.nu, config.n_eval)
    row["sigma2_median"] = float(np.median(sigma2))
    row["tau2_median"] = float(np.median(tau2))
    for n in config.n_eval:
        values = reliability[n]
        median_value = float(np.nanmedian(values))
        row[f"reliability_n{n}"] = median_value
        row[f"reliability_ci_low_n{n}"] = float(np.nanquantile(values, 0.025))
        row[f"reliability_ci_high_n{n}"] = float(np.nanquantile(values, 0.975))
        row[f"ceiling_n{n}"] = posterior_ceiling(median_value)
    return TargetFit(
        target=data.target,
        status="ok",
        row=row,
        draws=np.column_stack([sigma2, tau2]),
    )


def fit_measurement_target(
    data: TargetData,
    *,
    config: MeasurementConfig | None = None,
    seed: int | None = None,
) -> TargetFit:
    """Fit one pre-registered Phase 13/14 candidate to a target."""
    cfg = config or MeasurementConfig()
    fit_seed = cfg.seed if seed is None else int(seed)
    if cfg.model == "m0_reference":
        centered = transform_target(data, center_animals=True, standardize=False)
        fit = fit_hierarchical_target(
            centered,
            n_eval=cfg.n_eval,
            draws=cfg.draws,
            grid_resolution=cfg.grid_resolution,
            seed=fit_seed,
            prior_scale=cfg.prior_scale,
        )
        fit.row["model"] = "m0_reference"
        return fit

    data, dropped = _drop_single_responder_animals(data)
    if cfg.model == "m1_projection_scaled":
        centered, _factor = _center_projection_scaled(data)
        fit = _fit_grid(centered, config=cfg, seed=fit_seed)
        fit.row["n_single_responder_animals_dropped"] = dropped
        return fit

    event_post = estimate_event_variance(
        data,
        prior_scale=cfg.event_prior_scale,
        grid_size=cfg.event_grid,
    )
    if not np.isfinite(event_post.omega2_median) or event_post.omega2_median <= 0.0:
        return TargetFit(
            target=data.target,
            status="fit_failed",
            row={
                "target": data.target,
                "model": cfg.model,
                "status": "fit_failed",
                "reason": "event_variance_failed",
                "n_animals": data.n_animals,
                "n_responders": data.n_responders,
            },
            draws=None,
        )
    event_data = _event_variance_data(data, omega2=event_post.omega2_median)
    centered, _factor = _center_projection_scaled(event_data)
    if cfg.model == "m4_spike_slab":
        fit = _fit_spike_slab(centered, config=cfg, seed=fit_seed)
    else:
        fit = _fit_grid(centered, config=cfg, seed=fit_seed)
    fit.row.update(
        {
            "n_single_responder_animals_dropped": dropped,
            "event_variance_median": event_post.omega2_median,
            "event_variance_mean": event_post.omega2_mean,
            "event_variance_ci_low": event_post.omega2_ci_low,
            "event_variance_ci_high": event_post.omega2_ci_high,
            "event_variance_cells": event_post.n_variance_cells,
            "event_zero_variance_cells": event_post.n_zero_variance_cells,
            "event_one_event_cells": event_post.n_one_event_cells,
            "event_variance_grid_n_eff": event_post.grid_n_eff,
        }
    )
    return fit


def fit_measurement_cells(
    cells: pd.DataFrame,
    *,
    model: str = "m2_event_variance",
    coverage: float = 0.8,
    n_eval: tuple[int, ...] = (1, 22),
    draws: int = 2500,
    grid_resolution: int = 80,
    seed: int = 12,
    min_events: int = 1,
    min_animals: int = 6,
    prior_scale: float = 1.0,
    event_prior_scale: float = 1.0,
    event_grid: int = 200,
    pi0: float = 0.5,
    slab_scale: float = 1.0,
) -> HierarchicalResult:
    """Fit a Phase 13/14 candidate independently for every target."""
    cfg = MeasurementConfig(
        model=model,
        n_eval=n_eval,
        draws=draws,
        grid_resolution=grid_resolution,
        seed=seed,
        prior_scale=prior_scale,
        event_prior_scale=event_prior_scale,
        event_grid=event_grid,
        pi0=pi0,
        slab_scale=slab_scale,
    )
    rows: list[dict[str, Any]] = []
    draws_by_target: dict[str, np.ndarray] = {}
    # M0 must reproduce the historical Phase 12R path: center first, then
    # standardize by the pooled SD of centered profiles.  M1/M2 standardize the
    # uncentered profiles and then apply their own projection/centering step.
    center_for_normalization = model == "m0_reference"
    for target, tgroup in cells.groupby("_target", sort=False):
        data, reason = _normalize_target_with_reason(
            tgroup,
            coverage=coverage,
            min_events=min_events,
            min_animals=min_animals,
            standardize=True,
            center_animals=center_for_normalization,
        )
        if data is None:
            rows.append(
                {
                    "target": str(target),
                    "model": model,
                    "status": "ineligible",
                    "reason": reason,
                    "coverage": float(coverage),
                    "n_animals": 0,
                    "n_responders": 0,
                    "n_variance_floored": 0,
                }
            )
            continue
        target_seed = seed + int(pd.util.hash_pandas_object(pd.Series([str(target)])).iloc[0] % 10_000)
        fit = fit_measurement_target(data, config=cfg, seed=target_seed)
        rows.append(fit.row)
        if fit.draws is not None and fit.status == "ok":
            draws_by_target[str(target)] = fit.draws

    by_target = pd.DataFrame(rows)
    eligible = by_target[by_target["status"] == "ok"] if not by_target.empty else by_target
    if eligible.empty:
        summary: dict[str, object] = {
            "status": "no_eligible_targets",
            "model": model,
            "coverage": float(coverage),
            "n_eval": list(n_eval),
            "n_targets": int(len(by_target)),
            "n_ok": 0,
        }
        return HierarchicalResult(
            by_target=by_target,
            summary=summary,
            draws_by_target=draws_by_target,
            status="no_eligible_targets",
            pooled_reliability_median=float("nan"),
            single_animal_reliability_median=float("nan"),
            pooled_ceiling_median=float("nan"),
        )
    max_n = max(n_eval)
    pooled = float(eligible[f"reliability_n{max_n}"].median())
    single = float(eligible[f"reliability_n{min(n_eval)}"].median())
    status = "ok" if len(eligible) == len(by_target) else "ok_with_failures"
    total_cells = float(eligible["n_observed_cells"].sum())
    total_floored = float(eligible["n_variance_floored"].sum())
    summary = {
        "status": status,
        "model": model,
        "coverage": float(coverage),
        "n_eval": list(n_eval),
        "n_targets": int(len(by_target)),
        "n_ok": int(len(eligible)),
        "n_ineligible": int((by_target["status"] != "ok").sum()),
        "pooled_reliability_median": pooled,
        "single_animal_reliability_median": single,
        "pooled_ceiling_median": posterior_ceiling(pooled),
        "max_grid_edge_mass": float(eligible["grid_edge_mass"].max()),
        "min_grid_n_eff": float(eligible["grid_n_eff"].min()),
        "grid_resolution": int(grid_resolution),
        "prior_scale": float(prior_scale),
        "n_variance_floored": int(total_floored),
        "n_observed_cells": int(total_cells),
        "variance_floor_fraction": (total_floored / total_cells) if total_cells else 0.0,
    }
    if model in {"m2_event_variance", "m4_spike_slab"} and "event_variance_median" in eligible.columns:
        summary["event_variance_median"] = float(eligible["event_variance_median"].median())
        summary["event_one_event_cell_fraction"] = float(
            eligible["event_one_event_cells"].sum() / max(eligible["n_observed_cells"].sum(), 1)
        )
    if model == "m4_spike_slab" and "p_zero" in eligible.columns:
        summary["p_zero_median"] = float(eligible["p_zero"].median())
        summary["p_zero_mean"] = float(eligible["p_zero"].mean())
        summary["event_spike_fraction"] = float((eligible["p_zero"] >= 0.5).mean())
    return HierarchicalResult(
        by_target=by_target,
        summary=summary,
        draws_by_target=draws_by_target,
        status=status,
        pooled_reliability_median=pooled,
        single_animal_reliability_median=single,
        pooled_ceiling_median=posterior_ceiling(pooled),
    )


def fit_measurement_reliability(
    frame: pd.DataFrame,
    *,
    response: str = "response_mean",
    animal: str = DEFAULT_ANIMAL_COLUMN,
    model: str = "m2_event_variance",
    coverage: float = 0.8,
    n_eval: tuple[int, ...] = (1, 22),
    draws: int = 2500,
    grid_resolution: int = 80,
    seed: int = 12,
    min_events: int = 1,
    min_animals: int = 6,
    prior_scale: float = 1.0,
    event_prior_scale: float = 1.0,
    event_grid: int = 200,
    pi0: float = 0.5,
    slab_scale: float = 1.0,
) -> HierarchicalResult:
    """Fit a Phase 13/14 candidate from an event-level trial table."""
    cells = cell_summaries(frame, response=response, animal=animal)
    return fit_measurement_cells(
        cells,
        model=model,
        coverage=coverage,
        n_eval=n_eval,
        draws=draws,
        grid_resolution=grid_resolution,
        seed=seed,
        min_events=min_events,
        min_animals=min_animals,
        prior_scale=prior_scale,
        event_prior_scale=event_prior_scale,
        event_grid=event_grid,
        pi0=pi0,
        slab_scale=slab_scale,
    )


__all__ = [
    "MODELS",
    "EventVariancePosterior",
    "MeasurementConfig",
    "SpikeSlabPosterior",
    "estimate_event_variance",
    "fit_measurement_cells",
    "fit_measurement_reliability",
    "fit_measurement_target",
]
