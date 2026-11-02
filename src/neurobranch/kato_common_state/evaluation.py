"""Evaluation helpers for matched Kato predictive controls."""

from __future__ import annotations

import numpy as np

from neurobranch.joint_dynamics.evaluation import neural_covariance_error
from neurobranch.kato_interactions.metrics import ensemble_metrics
from neurobranch.probabilistic.metrics import (
    coverage_summary,
    energy_score,
)
from neurobranch.uncertainty_calibration.trajectory import trajectory_energy_score


def _prefixed(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in values.items()}


def _neuron_metrics(ensemble: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Neuron-space scores that avoid an expensive within-path 103x103 covariance."""
    h = min(20, ensemble.shape[1], truth.shape[0])
    cov = coverage_summary(ensemble[:, h - 1], truth[h - 1])
    return {
        "energy_h20": float(energy_score(ensemble[:, h - 1], truth[h - 1])),
        "trajectory_energy_h20": float(trajectory_energy_score(ensemble[:, :h], truth[:h])),
        "mse_mean_h20": float(np.mean((ensemble[:, h - 1].mean(axis=0) - truth[h - 1]) ** 2)),
        "coverage_50": float(cov["coverage_50"]),
        "coverage_80": float(cov["coverage_80"]),
        "coverage_95": float(cov["coverage_95"]),
        "neural_cov_error": float(neural_covariance_error(ensemble[:, :h], truth[:h])),
    }


def score_ensemble(
    ensemble: np.ndarray,
    truth: np.ndarray,
    pca,
    *,
    factor_k: int,
) -> dict[str, float]:
    """Score a matched ensemble in PCA-10, neuron, factor and residual spaces."""
    ens = np.asarray(ensemble, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    ens_proj = pca.transform(ens.reshape(-1, ens.shape[-1])).reshape(ens.shape[0], ens.shape[1], -1)
    truth_proj = pca.transform(truth)
    out: dict[str, float] = {}
    out.update(_prefixed("pca10", ensemble_metrics(ens_proj, truth_proj)))
    out.update(_prefixed("neuron", _neuron_metrics(ens, truth)))

    components = np.asarray(pca.components_[:factor_k], dtype=np.float64)
    ens_centered = ens - pca.mean_
    truth_centered = truth - pca.mean_
    ens_factor = ens_centered @ components.T
    truth_factor = truth_centered @ components.T
    ens_residual = ens_centered - ens_factor @ components
    truth_residual = truth_centered - truth_factor @ components
    h = 20
    out["factor_energy_h20"] = float(energy_score(ens_factor[:, h - 1], truth_factor[h - 1]))
    out["residual_energy_h20"] = float(
        energy_score(ens_residual[:, h - 1], truth_residual[h - 1])
    )
    out["factor_mse_h20"] = float(
        np.mean((ens_factor[:, h - 1].mean(axis=0) - truth_factor[h - 1]) ** 2)
    )
    out["residual_mse_h20"] = float(
        np.mean((ens_residual[:, h - 1].mean(axis=0) - truth_residual[h - 1]) ** 2)
    )
    for step in (1, 5, 10, 20):
        out[f"pca10_energy_h{step}"] = float(
            energy_score(ens_proj[:, step - 1], truth_proj[step - 1])
        )
        out[f"pca10_mse_h{step}"] = float(
            np.mean((ens_proj[:, step - 1].mean(axis=0) - truth_proj[step - 1]) ** 2)
        )
    return out
