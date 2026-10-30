"""Hierarchical measurement model for target-profile reliability.

The Phase 11 moment estimator is transparent but its resampling interval is not
calibrated on the sparse atlas.  This module implements the pre-specified
alternative: a small Bayesian hierarchical model with an explicit responder
coverage filter and known cell-level sampling variance.

For target ``t`` and standardized cell means ``y_ar`` (animal ``a``, responder
``r``)::

    y_ar = mu_r + u_ar + e_ar
    mu_r ~ Normal(0, sigma^2)
    u_ar ~ Normal(0, tau^2)
    e_ar ~ Normal(0, v_ar)

``v_ar`` is the within-cell variance divided by the event count.  Integrating
``mu`` analytically gives, for each responder, a rank-one covariance across
animals.  The resulting likelihood is O(animals x responders) and avoids a
general matrix inverse.

The reliability of a profile pooled over ``n`` animals is::

    R(n) = sigma^2 / (sigma^2 + (tau^2 + mean(v)) / n)

This is a model-based estimand.  It is not a causal effect, and it is
conditional on the observed responder-coverage pattern.  The companion
calibration module is responsible for deciding whether the posterior interval
is calibrated; this module only fits and reports diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from neurobranch.benchmark.variance import DEFAULT_ANIMAL_COLUMN, cell_summaries

MIN_RESPONDERS = 3


@dataclass(frozen=True)
class _TargetData:
    target: str
    y: np.ndarray
    v: np.ndarray
    mask: np.ndarray
    n_animals: int
    n_responders: int
    coverage: float
    nu: float
    responders: tuple[str, ...]


@dataclass(frozen=True)
class _SampleOutput:
    sigma2: np.ndarray
    tau2: np.ndarray
    edge_mass: float
    n_eff: float


def posterior_ceiling(reliability: float) -> float:
    """Attenuation ceiling ``sqrt(reliability)``, clipped to [0, 1]."""
    if not np.isfinite(reliability):
        return float("nan")
    return float(np.sqrt(np.clip(reliability, 0.0, 1.0)))


def _normalize_target(
    cells: pd.DataFrame,
    *,
    coverage: float,
    min_events: int,
    min_animals: int,
    standardize: bool,
    center_animals: bool = True,
) -> _TargetData | None:
    """Build the masked, standardized animal x responder matrix for one target."""
    subset = cells[cells["n"] >= min_events]
    if subset.empty:
        return None
    means = subset.pivot_table(index="_animal", columns="_neuron", values="mean", aggfunc="mean")
    variances = subset.assign(v=subset["var"] / np.maximum(subset["n"], 1.0)).pivot_table(
        index="_animal", columns="_neuron", values="v", aggfunc="mean"
    )
    variances = variances.reindex(index=means.index, columns=means.columns)
    if means.shape[1] < MIN_RESPONDERS:
        return None
    coverage_by_responder = means.notna().sum(axis=0) / max(means.shape[0], 1)
    keep = coverage_by_responder >= float(coverage)
    if int(keep.sum()) < MIN_RESPONDERS:
        return None
    means = means.loc[:, keep]
    variances = variances.loc[:, keep]
    keep_animals = means.notna().any(axis=1)
    means = means.loc[keep_animals]
    variances = variances.loc[keep_animals]
    if means.shape[0] < max(min_animals, MIN_RESPONDERS):
        return None

    y = means.to_numpy(dtype=float)
    v = variances.to_numpy(dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < MIN_RESPONDERS * MIN_RESPONDERS:
        return None
    # A cell without a finite variance contributes only its mean; floor the
    # sampling variance so the likelihood remains finite.
    v = np.where(mask & np.isfinite(v), v, 1.0)
    v = np.maximum(v, 1e-8)
    if center_animals:
        row_means = np.nanmean(np.where(mask, y, np.nan), axis=1, keepdims=True)
        y = y - row_means
    if standardize:
        finite_y = y[mask]
        center = float(np.mean(finite_y))
        scale = float(np.std(finite_y, ddof=1))
        if not np.isfinite(scale) or scale <= 0:
            return None
        y = (y - center) / scale
        v = v / (scale**2)
    return _TargetData(
        target=str(subset["_target"].iloc[0]),
        y=y,
        v=v,
        mask=mask,
        n_animals=int(y.shape[0]),
        n_responders=int(y.shape[1]),
        coverage=float(coverage),
        nu=float(np.mean(v[mask])),
        responders=tuple(str(column) for column in means.columns),
    )


def _log_posterior(
    theta: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray,
    *,
    prior_scale: float = 1.0,
) -> float:
    """Log posterior for ``theta = (log sigma, log tau)`` up to a constant."""
    log_sigma, log_tau = float(theta[0]), float(theta[1])
    if not np.isfinite(log_sigma) or not np.isfinite(log_tau):
        return -np.inf
    sigma2 = float(np.exp(2.0 * log_sigma))
    tau2 = float(np.exp(2.0 * log_tau))
    if sigma2 <= 0 or tau2 <= 0 or sigma2 > 1e6 or tau2 > 1e6:
        return -np.inf

    d = tau2 + v
    inv = 1.0 / d
    inv_masked = np.where(mask, inv, 0.0)
    s = inv_masked.sum(axis=0)
    q = np.where(mask, y * y * inv, 0.0).sum(axis=0)
    linear = np.where(mask, y * inv, 0.0).sum(axis=0)
    log_det = np.where(mask, np.log(d), 0.0).sum(axis=0)
    denominator = 1.0 + sigma2 * s
    log_det = log_det + np.log(denominator)
    quadratic = q - (sigma2 * linear * linear) / denominator
    log_likelihood = -0.5 * float(np.sum(log_det + quadratic))

    # HalfNormal(prior_scale) priors on sigma and tau; the log transforms add
    # the Jacobian terms log(sigma) and log(tau).
    log_prior = -0.5 * ((sigma2 + tau2) / (prior_scale**2)) + log_sigma + log_tau
    return float(log_likelihood + log_prior)


def _grid_posterior(
    data: _TargetData,
    *,
    resolution: int,
    draws: int,
    seed: int,
    prior_scale: float = 1.0,
) -> _SampleOutput:
    """Sample the exact 2-D grid posterior for ``(log sigma, log tau)``.

    A grid is used instead of MCMC because the model has only two global
    parameters per target and the marginal likelihood is cheap.  This avoids
    the convergence failures of a random-walk sampler on weakly identified
    targets, and makes SBC reproducible.  The cost is discretization: the
    posterior mass on the grid boundary is reported, and a fit with more than
    5% boundary mass is flagged.
    """
    y, v, mask = data.y, data.v, data.mask
    log_sigma = np.linspace(-6.0, float(np.log(20.0)), int(resolution))
    log_tau = np.linspace(-6.0, float(np.log(20.0)), int(resolution))
    log_post = np.empty((log_sigma.size, log_tau.size), dtype=float)
    for i, sigma in enumerate(log_sigma):
        for j, tau in enumerate(log_tau):
            log_post[i, j] = _log_posterior(np.array([sigma, tau]), y, v, mask, prior_scale=prior_scale)
    finite = np.isfinite(log_post)
    if not finite.any():
        return _SampleOutput(
            sigma2=np.array([np.nan]),
            tau2=np.array([np.nan]),
            edge_mass=1.0,
            n_eff=0.0,
        )
    shifted = np.where(finite, log_post - np.max(log_post[finite]), -np.inf)
    weights = np.exp(shifted)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        return _SampleOutput(
            sigma2=np.array([np.nan]),
            tau2=np.array([np.nan]),
            edge_mass=1.0,
            n_eff=0.0,
        )
    weights /= total
    edge = weights[0, :].sum() + weights[-1, :].sum() + weights[:, 0].sum() + weights[:, -1].sum()
    n_eff = float(1.0 / np.sum(weights**2))
    rng = np.random.default_rng(seed)
    flat = rng.choice(weights.size, size=int(draws), p=weights.ravel())
    rows, cols = np.divmod(flat, weights.shape[1])
    step_sigma = float(log_sigma[1] - log_sigma[0])
    step_tau = float(log_tau[1] - log_tau[0])
    sigma = np.exp(log_sigma[rows] + rng.uniform(-0.5, 0.5, size=rows.size) * step_sigma)
    tau = np.exp(log_tau[cols] + rng.uniform(-0.5, 0.5, size=cols.size) * step_tau)
    return _SampleOutput(
        sigma2=sigma**2,
        tau2=tau**2,
        edge_mass=float(edge),
        n_eff=n_eff,
    )


def _reliability_draws(
    sigma2: np.ndarray,
    tau2: np.ndarray,
    nu: float,
    n_eval: tuple[int, ...],
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for n in n_eval:
        if n < 1:
            raise ValueError("n_eval entries must be >= 1")
        denominator = sigma2 + (tau2 + nu) / float(n)
        out[n] = np.clip(sigma2 / np.where(denominator > 0, denominator, np.nan), 0.0, 1.0)
    return out


def fit_hierarchical_cells(
    cells: pd.DataFrame,
    *,
    coverage: float = 0.0,
    n_eval: tuple[int, ...] = (1, 22),
    chains: int = 4,
    warmup: int = 750,
    draws: int = 4000,
    grid_resolution: int = 100,
    seed: int = 12,
    min_events: int = 1,
    min_animals: int = 6,
    standardize: bool = True,
    center_animals: bool = True,
    prior_scale: float = 1.0,
) -> HierarchicalResult:
    """Fit the hierarchical model independently for every eligible target.

    Args:
        frame: event-level trial table.
        response: response column.
        animal: animal identifier column.
        coverage: minimum fraction of animals in which a responder must be
            observed.  The model is conditional on this observed pattern.
        n_eval: animal counts at which to report reliability.
        chains: number of independent Metropolis chains.
        warmup: warm-up iterations per chain.
        draws: retained iterations per chain.
        seed: seed for chain initialization and proposals.
        min_events: minimum events per animal/target/responder cell.
        min_animals: minimum animals with any observation for a target.
    """
    if coverage < 0 or coverage > 1:
        raise ValueError("coverage must be in [0, 1]")
    del chains, warmup  # retained for API compatibility; the grid has no chains
    if draws < 100:
        raise ValueError("draws >= 100 is required")
    if grid_resolution < 20:
        raise ValueError("grid_resolution >= 20 is required")
    rows: list[dict[str, object]] = []
    draws_by_target: dict[str, np.ndarray] = {}
    for target, tgroup in cells.groupby("_target", sort=False):
        data = _normalize_target(
            tgroup,
            coverage=coverage,
            min_events=min_events,
            min_animals=min_animals,
            standardize=standardize,
            center_animals=center_animals,
        )
        if data is None:
            rows.append(
                {
                    "target": str(target),
                    "status": "ineligible",
                    "coverage": float(coverage),
                    "n_animals": 0,
                    "n_responders": 0,
                }
            )
            continue
        fit = _grid_posterior(
            data,
            resolution=grid_resolution,
            draws=draws,
            seed=seed + int(pd.util.hash_pandas_object(pd.Series([target])).iloc[0] % 10_000),
            prior_scale=prior_scale,
        )
        sigma2 = fit.sigma2
        tau2 = fit.tau2
        reliability = _reliability_draws(sigma2, tau2, data.nu, n_eval)
        row: dict[str, object] = {
            "target": str(target),
            "status": "ok",
            "coverage": float(coverage),
            "n_animals": data.n_animals,
            "n_responders": data.n_responders,
            "mean_sampling_var": data.nu,
            "sigma2_median": float(np.median(sigma2)),
            "tau2_median": float(np.median(tau2)),
            "grid_edge_mass": fit.edge_mass,
            "grid_n_eff": fit.n_eff,
            "n_draws": int(sigma2.size),
        }
        for n in n_eval:
            values = reliability[n]
            median_value = float(np.nanmedian(values))
            row[f"reliability_n{n}"] = median_value
            row[f"reliability_ci_low_n{n}"] = float(np.nanquantile(values, 0.025))
            row[f"reliability_ci_high_n{n}"] = float(np.nanquantile(values, 0.975))
            row[f"ceiling_n{n}"] = posterior_ceiling(median_value)
        rows.append(row)
        draws_by_target[str(target)] = np.column_stack([sigma2, tau2])

    by_target = pd.DataFrame(rows)
    eligible = by_target[by_target["status"] == "ok"] if not by_target.empty else by_target
    if eligible.empty:
        summary = {
            "status": "no_eligible_targets",
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
    pooled_col = f"reliability_n{max_n}"
    single_col = f"reliability_n{min(n_eval)}"
    pooled = float(eligible[pooled_col].median())
    single = float(eligible[single_col].median())
    status = "ok" if len(eligible) == len(by_target) else "ok_with_failures"
    summary = {
        "status": status,
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
    }
    return HierarchicalResult(
        by_target=by_target,
        summary=summary,
        draws_by_target=draws_by_target,
        status=status,
        pooled_reliability_median=pooled,
        single_animal_reliability_median=single,
        pooled_ceiling_median=posterior_ceiling(pooled),
    )


def fit_hierarchical_reliability(
    frame: pd.DataFrame,
    *,
    response: str = "response_mean",
    animal: str = DEFAULT_ANIMAL_COLUMN,
    coverage: float = 0.0,
    n_eval: tuple[int, ...] = (1, 22),
    chains: int = 4,
    warmup: int = 750,
    draws: int = 4000,
    grid_resolution: int = 100,
    seed: int = 12,
    min_events: int = 1,
    min_animals: int = 6,
    prior_scale: float = 1.0,
    center_animals: bool = True,
) -> HierarchicalResult:
    """Fit the hierarchical model from an event-level trial table."""
    cells = cell_summaries(frame, response=response, animal=animal)
    return fit_hierarchical_cells(
        cells,
        coverage=coverage,
        n_eval=n_eval,
        chains=chains,
        warmup=warmup,
        draws=draws,
        grid_resolution=grid_resolution,
        seed=seed,
        min_events=min_events,
        min_animals=min_animals,
        prior_scale=prior_scale,
        center_animals=center_animals,
    )


class HierarchicalResult:
    """Fit output with per-target rows, summary and posterior draws."""

    def __init__(
        self,
        *,
        by_target: pd.DataFrame,
        summary: dict[str, object],
        draws_by_target: dict[str, np.ndarray],
        status: str,
        pooled_reliability_median: float,
        single_animal_reliability_median: float,
        pooled_ceiling_median: float,
    ) -> None:
        self.by_target = by_target
        self.summary = summary
        self.draws_by_target = draws_by_target
        self.status = status
        self.pooled_reliability_median = pooled_reliability_median
        self.single_animal_reliability_median = single_animal_reliability_median
        self.pooled_ceiling_median = pooled_ceiling_median


__all__ = [
    "HierarchicalResult",
    "fit_hierarchical_cells",
    "fit_hierarchical_reliability",
    "posterior_ceiling",
]
