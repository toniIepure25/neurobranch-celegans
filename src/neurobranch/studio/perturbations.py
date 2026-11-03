"""Model-dependent virtual perturbation calculations.

This is not an optogenetic, electrical, or behavioural intervention.  It edits
one coordinate of a fitted linear state-space model and rolls the model forward
under zero innovation.  The result is a mathematical counterfactual, not a
prediction of what a living animal would do.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sklearn.decomposition import PCA

from neurobranch.kato_common_state.models import (
    fit_window_common,
    rollout_common_factor,
)
from neurobranch.kato_interactions.models import FrozenLinearModel, fit_window
from neurobranch.studio.activity import _all_recordings


@dataclass(frozen=True)
class PerturbationResult:
    session: int
    landmark: int
    neuron: str
    delta_z: float
    horizon: int
    pca_baseline: np.ndarray
    pca_perturbed: np.ndarray
    pca_difference: np.ndarray
    neuron_response: np.ndarray
    population_response: np.ndarray
    disagreement_h20: float
    ood_fraction: float
    ood: bool
    b3_radius: float
    b5_radius: float
    explained_variance_ratio: np.ndarray
    origin_time: float
    context_time: np.ndarray
    future_time: np.ndarray
    observed_context: np.ndarray
    observed_origin: np.ndarray


def _pca(data) -> PCA:
    train = data.X[data.masks["train"]]
    return PCA(n_components=min(10, train.shape[1]), random_state=17).fit(train)


@lru_cache(maxsize=24)
def perturb(
    session: int,
    landmark: int,
    neuron: str,
    delta_z: float = 1.0,
    horizon: int = 20,
) -> PerturbationResult:
    data = _all_recordings()[int(session)]
    sid = int(session)
    t = int(landmark)
    names = [str(name) for name in data.predictor_names]
    if neuron not in names:
        raise KeyError(f"Neuron {neuron!r} is not a predictor in recording {sid}")
    if not 0 <= t < data.n_frames - horizon:
        raise ValueError("Perturbation origin is outside the observation")
    pca = _pca(data)
    scores = pca.transform(data.X)
    factors = scores
    b3 = fit_window(data.X, t, window=500, ridge=1.0, spectral_cap=0.995)
    b5 = fit_window_common(data.X, factors, t, k=10, window=500, ridge=1.0)
    if b3 is None or b5 is None:
        raise ValueError("The selected model could not be fitted at this origin")
    n = int(data.X.shape[1])
    zeros = np.zeros((1, int(horizon), n), dtype=np.float64)
    x0 = np.asarray(data.X[t], dtype=np.float64).copy()
    xp = x0.copy()
    j = names.index(neuron)
    xp[j] += float(delta_z)

    b3_frozen = FrozenLinearModel(A=b3.A, b=b3.b)
    base_ens = _roll(b3_frozen, x0, zeros)
    pert_ens = _roll(b3_frozen, xp, zeros)
    base_b5 = rollout_common_factor(b5, x0, factors[t, :10], zeros)[0]
    pert_b5 = rollout_common_factor(b5, xp, factors[t, :10], zeros)[0]

    pca_base = pca.transform(base_ens[0])
    pca_pert = pca.transform(pert_ens[0])
    pca_base_b5 = pca.transform(base_b5)
    pca_pert_b5 = pca.transform(pert_b5)
    diff = pca_pert - pca_base

    # Descriptive OOD check: where does the edited coordinate sit relative to
    # the *training* marginal, not the test tail used for evaluation?
    train = data.X[data.masks["train"], j]
    lo, hi = np.percentile(train, [0.5, 99.5])
    ood_fraction = float(np.mean((xp[j] < lo) | (xp[j] > hi)))
    disagreement = float(
        np.linalg.norm((pca_pert - pca_base) - (pca_pert_b5 - pca_base_b5), axis=1)[-1]
    )
    tv = np.asarray(data.recording.time_seconds, dtype=np.float64)
    return PerturbationResult(
        session=sid,
        landmark=t,
        neuron=neuron,
        delta_z=float(delta_z),
        horizon=int(horizon),
        pca_baseline=pca_base,
        pca_perturbed=pca_pert,
        pca_difference=diff,
        neuron_response=np.asarray(pert_ens - base_ens)[0, :, j],
        population_response=np.linalg.norm(pert_ens - base_ens, axis=2)[0],
        disagreement_h20=disagreement,
        ood_fraction=ood_fraction,
        ood=bool(ood_fraction > 0),
        b3_radius=float(np.max(np.abs(np.linalg.eigvals(b3.A)))),
        b5_radius=float(np.max(np.abs(np.linalg.eigvals(b5.factor_A)))),
        explained_variance_ratio=np.asarray(pca.explained_variance_ratio_),
        origin_time=float(tv[t]),
        context_time=tv[max(0, t - 120) : t + 1],
        future_time=tv[t + 1 : t + 1 + int(horizon)],
        observed_context=scores[max(0, t - 120) : t + 1],
        observed_origin=scores[t],
    )


def _roll(model, x0: np.ndarray, eps: np.ndarray) -> np.ndarray:
    x = np.repeat(np.asarray(x0, dtype=np.float64)[None, :], eps.shape[0], axis=0)
    out = np.empty((eps.shape[0], eps.shape[1], x.shape[1]), dtype=np.float64)
    for step in range(eps.shape[1]):
        x = x @ model.A.T + model.b + eps[:, step]
        out[:, step] = x
    return out
