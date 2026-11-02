"""Bounded mathematical sensitivity in a fixed PCA representation."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.decomposition import PCA

from neurobranch.kato_interactions.models import FrozenLinearModel


def neuron_delta(pca: PCA, neuron_index: int, amplitude: float) -> np.ndarray:
    """Latent displacement produced by adding ``amplitude`` SD to one neuron."""
    j = int(neuron_index)
    return float(amplitude) * np.asarray(pca.components_[:, j], dtype=np.float64)


def unit_neuron_delta(neuron_index: int, amplitude: float, n_neurons: int) -> np.ndarray:
    delta = np.zeros(int(n_neurons), dtype=np.float64)
    delta[int(neuron_index)] = float(amplitude)
    return delta


def finite_difference_error(model: FrozenLinearModel, z0: np.ndarray, delta: np.ndarray) -> float:
    """Relative disagreement between analytic Jacobian and one-step finite difference."""
    analytic = model.A @ np.asarray(delta, dtype=np.float64)
    fd = model.forecast_mean(np.asarray(z0) + np.asarray(delta), 1)[0] - model.forecast_mean(z0, 1)[0]
    denom = max(float(np.linalg.norm(analytic)), 1e-12)
    return float(np.linalg.norm(fd - analytic) / denom)


def response_metrics(
    model: FrozenLinearModel,
    pca: PCA,
    x0: np.ndarray,
    neuron_index: int,
    *,
    amplitude: float,
    horizon: int = 20,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Perturb one standardized neuron, then propagate in the fitted neuron space."""
    delta = unit_neuron_delta(neuron_index, amplitude, len(x0))
    base = model.forecast_mean(x0, horizon)
    pert = model.forecast_mean(np.asarray(x0) + delta, horizon)
    neuron = pert - base
    latent = pca.transform(pert) - pca.transform(base)
    h = min(int(horizon), len(neuron))
    v = neuron[h - 1]
    nv = float(np.linalg.norm(v))
    nv_base = float(np.linalg.norm(base[h - 1]))
    peak = float(np.linalg.norm(neuron, axis=1).max())
    target_component = float(neuron[h - 1, int(neuron_index)])
    latent_h = latent[h - 1]
    metrics = {
        "amplitude_sd": float(amplitude),
        "neuron_response_h20": nv,
        "neuron_response_peak": peak,
        "neuron_relative_h20": nv / max(nv_base, 1e-12),
        "latent_response_h20": float(np.linalg.norm(latent_h)),
        "latent_response_peak": float(np.linalg.norm(latent, axis=1).max()),
        "latent_relative_h20": float(np.linalg.norm(latent_h) / max(np.linalg.norm(pca.transform(base[h - 1][None, :])[0]), 1e-12)),
        "target_component_h20": target_component,
        "response_decay_frames": float(np.argmax(np.linalg.norm(neuron, axis=1) < 0.1 * max(peak, 1e-12))),
        "finite_difference_error": finite_difference_error(model, x0, delta),
        "target_loading_norm": float(np.linalg.norm(pca.components_[:, int(neuron_index)])),
    }
    return metrics, neuron, latent


def response_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom > 1e-12 else float("nan")


def response_magnitude_ratio(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(b)), 1e-12)
    return float(np.linalg.norm(a) / denom)


def support_fraction(
    paths: np.ndarray,
    *,
    train_reference: np.ndarray,
    lower: float = 0.01,
    upper: float = 0.99,
) -> float:
    arr = np.asarray(paths, dtype=np.float64)
    lo = np.quantile(np.asarray(train_reference, dtype=np.float64), lower, axis=0)
    hi = np.quantile(np.asarray(train_reference, dtype=np.float64), upper, axis=0)
    return float(((arr < lo) | (arr > hi)).mean())


def matched_control_indices(
    train_variance: np.ndarray,
    target_index: int,
    *,
    exclude: set[int] | None = None,
    n_controls: int = 2,
) -> list[int]:
    """Prespecified nearest-training-variance controls, excluding all targets."""
    train_variance = np.asarray(train_variance, dtype=np.float64)
    blocked = set(exclude or set()) | {int(target_index)}
    candidates = [j for j in range(len(train_variance)) if j not in blocked]
    candidates.sort(key=lambda j: abs(float(train_variance[j]) - float(train_variance[target_index])))
    return candidates[: int(n_controls)]


def pairwise_disagreement(
    baselines: dict[str, np.ndarray],
    responses: dict[str, np.ndarray],
) -> dict[str, float]:
    keys = sorted(set(baselines) & set(responses))
    pairs = list(combinations(keys, 2))
    if not pairs:
        return {
            "n_fits": len(keys),
            "mean_response_cosine": float("nan"),
            "min_response_cosine": float("nan"),
            "mean_magnitude_ratio_cv": float("nan"),
            "prediction_relative_spread": float("nan"),
            "sign_agreement": float("nan"),
        }
    cosines = []
    ratios = []
    sign_hits = []
    spreads = []
    for a, b in pairs:
        ca = response_cosine(responses[a], responses[b])
        cosines.append(ca)
        ratios.append(response_magnitude_ratio(responses[a], responses[b]))
        sign_hits.append(float(np.sign(responses[a][0]) == np.sign(responses[b][0])))
        va = np.asarray(baselines[a], dtype=np.float64)
        vb = np.asarray(baselines[b], dtype=np.float64)
        spreads.append(float(np.linalg.norm(va - vb) / max(np.linalg.norm(va), np.linalg.norm(vb), 1e-12)))
    mags = np.asarray([np.linalg.norm(v) for v in responses.values()], dtype=np.float64)
    return {
        "n_fits": len(keys),
        "mean_response_cosine": float(np.nanmean(cosines)),
        "min_response_cosine": float(np.nanmin(cosines)),
        "mean_magnitude_ratio_cv": float(np.nanstd(mags) / max(np.nanmean(mags), 1e-12)),
        "prediction_relative_spread": float(np.nanmean(spreads)),
        "sign_agreement": float(np.nanmean(sign_hits)),
    }
