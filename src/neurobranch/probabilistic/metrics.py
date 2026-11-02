"""Proper probabilistic scores and calibration utilities.

All ensemble scores use the fair finite-ensemble estimator.  Lower is better.
The implementation is deliberately independent of any neural model so that it
can be tested on known distributions.
"""

from __future__ import annotations

import numpy as np


def _as_ensemble(ensemble: np.ndarray) -> np.ndarray:
    arr = np.asarray(ensemble, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("ensemble must be (members, dimensions)")
    if not np.isfinite(arr).all():
        raise ValueError("non-finite ensemble")
    return arr


def energy_score(ensemble: np.ndarray, observation: np.ndarray) -> float:
    """Multivariate energy score, lower is better.

    ``E||X-y|| - 0.5 E||X-X'||`` with the second term estimated from
    distinct ensemble pairs only.  The score is proper for the ensemble
    distribution.
    """
    ens = _as_ensemble(ensemble)
    obs = np.asarray(observation, dtype=np.float64).ravel()
    if ens.shape[1] != obs.size:
        raise ValueError("dimension mismatch")
    term1 = float(np.linalg.norm(ens - obs, axis=1).mean())
    m = ens.shape[0]
    if m < 2:
        return term1
    diff = ens[:, None, :] - ens[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    term2 = 0.5 * float(dist[np.triu_indices(m, k=1)].mean())
    return term1 - term2


def crps_ensemble(ensemble: np.ndarray, observation: np.ndarray) -> np.ndarray:
    """Per-dimension fair CRPS for a finite ensemble.

    Returns one value per dimension.  For each dimension
    ``E|X-y| - 0.5 E|X-X'|``.
    """
    ens = _as_ensemble(ensemble)
    obs = np.asarray(observation, dtype=np.float64).ravel()
    if ens.shape[1] != obs.size:
        raise ValueError("dimension mismatch")
    term1 = np.abs(ens - obs).mean(axis=0)
    ordered = np.sort(ens, axis=0)
    m = ens.shape[0]
    weights = 2.0 * np.arange(1, m + 1) - m - 1
    term2 = (weights[:, None] * ordered).sum(axis=0) / float(m * m)
    return term1 - term2


def interval_coverage(
    ensemble: np.ndarray, observation: np.ndarray, level: float
) -> tuple[np.ndarray, np.ndarray]:
    """Central predictive interval coverage and width per dimension."""
    ens = _as_ensemble(ensemble)
    obs = np.asarray(observation, dtype=np.float64).ravel()
    lo = np.quantile(ens, (1.0 - level) / 2.0, axis=0)
    hi = np.quantile(ens, 1.0 - (1.0 - level) / 2.0, axis=0)
    covered = (obs >= lo) & (obs <= hi)
    return covered.astype(np.float64), (hi - lo)


def pit_values(ensemble: np.ndarray, observation: np.ndarray) -> np.ndarray:
    """Probability integral transform values for each dimension."""
    ens = _as_ensemble(ensemble)
    obs = np.asarray(observation, dtype=np.float64).ravel()
    return (ens <= obs[None, :]).mean(axis=0)


def pit_histogram(pit: np.ndarray, bins: int = 10) -> tuple[np.ndarray, float]:
    values = np.asarray(pit, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    counts, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    density = counts / max(1, counts.sum())
    calibration_error = float(np.abs(density - 1.0 / bins).mean())
    return density, calibration_error


def variogram_score(
    ensemble: np.ndarray, observation: np.ndarray, *, gamma: float = 0.5
) -> float:
    """Ensemble variogram score over all dimension pairs at one time point.

    The score is the squared difference between transformed absolute pairwise
    differences of forecast members (averaged) and of the observation.
    """
    ens = _as_ensemble(ensemble)
    obs = np.asarray(observation, dtype=np.float64).ravel()
    if ens.shape[1] != obs.size:
        raise ValueError("dimension mismatch")
    i, j = np.triu_indices(obs.size, k=1)
    obs_diff = np.abs(obs[i] - obs[j]) ** gamma
    member_diff = np.abs(ens[:, i] - ens[:, j]) ** gamma
    return float(np.mean((member_diff.mean(axis=0) - obs_diff) ** 2))


def coverage_summary(
    ensemble: np.ndarray,
    observation: np.ndarray,
    levels: tuple[float, ...] = (0.5, 0.8, 0.95),
) -> dict[str, float]:
    out: dict[str, float] = {}
    for level in levels:
        covered, width = interval_coverage(ensemble, observation, level)
        tag = f"{int(level * 100)}"
        out[f"coverage_{tag}"] = float(covered.mean())
        out[f"width_{tag}"] = float(width.mean())
    return out


def effective_rank(covariance: np.ndarray) -> tuple[float, float]:
    eig = np.linalg.eigvalsh(np.asarray(covariance, dtype=np.float64))
    eig = np.clip(eig, 0.0, None)
    total = float(eig.sum())
    if total <= 0:
        return float("nan"), float("nan")
    participation = float(total**2 / np.sum(eig**2))
    positive = eig[eig > 1e-12 * max(eig.max(), 1e-12)]
    condition = float(positive.max() / positive.min()) if positive.size else float("nan")
    return participation, condition


def covariance_relative_error(predicted: np.ndarray, observed: np.ndarray) -> float:
    p = np.asarray(predicted, dtype=np.float64)
    o = np.asarray(observed, dtype=np.float64)
    denom = float(np.linalg.norm(p) + np.linalg.norm(o))
    return float(np.linalg.norm(p - o) / denom) if denom > 0 else float("nan")
