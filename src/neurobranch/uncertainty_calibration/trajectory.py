"""Proper trajectory-level scoring for probabilistic neural simulations."""

from __future__ import annotations

import numpy as np

from neurobranch.probabilistic.metrics import energy_score


def trajectory_energy_score(
    ensemble: np.ndarray,
    observation: np.ndarray,
    *,
    normalize_horizon: bool = True,
) -> float:
    """Energy score over flattened, prespecified trajectory representations.

    ``ensemble`` has shape ``(members, horizon, features)`` and ``observation``
    has shape ``(horizon, features)``.  Flattening keeps temporal ordering and
    scores the joint distribution rather than each horizon separately.
    """
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observation = np.asarray(observation, dtype=np.float64)
    if ensemble.ndim != 3:
        raise ValueError("ensemble must be (members, horizon, features)")
    if observation.shape != ensemble.shape[1:]:
        raise ValueError("observation shape mismatch")
    scale = np.sqrt(ensemble.shape[1]) if normalize_horizon else 1.0
    flat = ensemble.reshape(ensemble.shape[0], -1) / scale
    obs = observation.reshape(-1) / scale
    return float(energy_score(flat, obs))


def variogram_score(
    ensemble: np.ndarray, observation: np.ndarray, *, pairs: int = 40, p: float = 0.5
) -> float:
    """Variogram score for a fixed set of feature pairs.

    Lower is better. The pair set is deterministic and shared across models.
    """
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observation = np.asarray(observation, dtype=np.float64)
    if ensemble.ndim == 2:
        ensemble = ensemble[:, None, :]
    d = ensemble.shape[-1]
    if pairs >= d * (d - 1) // 2:
        i, j = np.triu_indices(d, k=1)
    else:
        rng = np.random.default_rng(20261102)
        i = rng.integers(0, d, size=pairs)
        j = rng.integers(0, d, size=pairs)
        keep = i != j
        i, j = i[keep], j[keep]
    diff_obs = np.abs(observation[..., i] - observation[..., j]) ** p
    diff_ens = np.abs(ensemble[..., i] - ensemble[..., j]) ** p
    return float(np.mean((diff_ens.mean(axis=0) - diff_obs) ** 2))
