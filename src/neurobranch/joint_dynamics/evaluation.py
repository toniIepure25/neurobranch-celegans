"""Evaluation summaries specific to joint neural trajectories."""

from __future__ import annotations

import numpy as np

from neurobranch.probabilistic.metrics import covariance_relative_error


def pooled_neural_covariance(paths: np.ndarray) -> np.ndarray:
    """Covariance across features after pooling every path and time point."""
    arr = np.asarray(paths, dtype=np.float64).reshape(-1, paths.shape[-1])
    arr = arr - arr.mean(axis=0, keepdims=True)
    return arr.T @ arr / max(1, len(arr) - 1)


def path_covariance(paths: np.ndarray) -> np.ndarray:
    """Average within-path temporal covariance across members and landmarks."""
    arr = np.asarray(paths, dtype=np.float64)
    covs = []
    for path in arr.reshape(-1, arr.shape[-2], arr.shape[-1]):
        c = path - path.mean(axis=0, keepdims=True)
        covs.append(c.T @ c / max(1, len(c) - 1))
    return np.mean(covs, axis=0)


def neural_covariance_error(ensemble: np.ndarray, truth: np.ndarray) -> float:
    """Relative error of pooled cross-neuron covariance."""
    return covariance_relative_error(pooled_neural_covariance(ensemble), pooled_neural_covariance(truth))


def path_covariance_error(ensemble: np.ndarray, truth: np.ndarray) -> float:
    """Relative error of within-trajectory temporal covariance."""
    return covariance_relative_error(path_covariance(ensemble), path_covariance(truth))


def autocorrelation_profile(paths: np.ndarray, max_lag: int = 10) -> np.ndarray:
    """Mean autocorrelation per lag, averaged over features."""
    arr = np.asarray(paths, dtype=np.float64)
    arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
    centered = arr - arr.mean(axis=1, keepdims=True)
    rows = []
    for lag in range(1, int(max_lag) + 1):
        vals = []
        for j in range(centered.shape[-1]):
            a, b = centered[:, :-lag, j].ravel(), centered[:, lag:, j].ravel()
            if np.std(a) > 1e-10 and np.std(b) > 1e-10:
                vals.append(np.corrcoef(a, b)[0, 1])
        rows.append(float(np.nanmean(vals)) if vals else np.nan)
    return np.asarray(rows)


def autocorrelation_error(ensemble: np.ndarray, truth: np.ndarray) -> float:
    p = autocorrelation_profile(ensemble)
    t = autocorrelation_profile(truth)
    return float(np.nanmean(np.abs(p - t)))


def occupancy_and_recurrence(paths: np.ndarray, centroids: np.ndarray, *, horizon: int = 20) -> tuple[np.ndarray, float]:
    arr = np.asarray(paths, dtype=np.float64)[:, :horizon]
    dist = ((arr[:, :, None, :] - centroids[None, None, :, :]) ** 2).sum(axis=-1)
    labels = dist.argmin(axis=-1)
    counts = np.bincount(labels.ravel(), minlength=len(centroids)).astype(np.float64)
    occupancy = counts / max(counts.sum(), 1.0)
    recurrence = float((labels[:, 1:] == labels[:, :-1]).mean()) if labels.shape[1] > 1 else float("nan")
    return occupancy, recurrence


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())
