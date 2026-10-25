"""Deployable baselines for the Phase 6 response benchmark.

All baselines are fitted on the training rows of a fold and return one point
prediction per test row.  The target-mean and oracle-pair-mean baselines are
intentionally strong: a model that only beats a global mean has not shown useful
pair structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from neurobranch.causal.features import FeatureBundle


@dataclass
class BaselinePrediction:
    name: str
    y_pred: np.ndarray
    y_lower: np.ndarray | None = None
    y_upper: np.ndarray | None = None


def _context_matrix(bundle: FeatureBundle, idx: np.ndarray) -> np.ndarray:
    raw = bundle.context_raw[idx].astype(float)
    medians = np.nanmedian(bundle.context_raw, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    return np.where(np.isfinite(raw), raw, medians)


def _fit_context_ridge(bundle: FeatureBundle, train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    x_train = _context_matrix(bundle, train_idx)
    scaler = StandardScaler().fit(x_train)
    model = Ridge(alpha=1.0).fit(scaler.transform(x_train), bundle.y[train_idx])
    return model.predict(scaler.transform(_context_matrix(bundle, test_idx)))


def _fit_pair_ridge(bundle: FeatureBundle, train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    scaler = StandardScaler().fit(bundle.pair_features[train_idx])
    model = Ridge(alpha=10.0).fit(scaler.transform(bundle.pair_features[train_idx]), bundle.y[train_idx])
    return model.predict(scaler.transform(bundle.pair_features[test_idx]))


def _group_mean(
    bundle: FeatureBundle,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    column: str,
) -> np.ndarray:
    idx = bundle.target_idx if column == "target" else bundle.responder_idx
    global_mean = float(np.mean(bundle.y[train_idx]))
    frame = pd.DataFrame({"g": idx[train_idx], "y": bundle.y[train_idx]})
    means = frame.groupby("g")["y"].mean()
    return np.array([means.get(int(i), global_mean) for i in idx[test_idx]], dtype=float)


def _pair_mean(bundle: FeatureBundle, train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    global_mean = float(np.mean(bundle.y[train_idx]))
    frame = pd.DataFrame(
        {
            "t": bundle.target_idx[train_idx],
            "r": bundle.responder_idx[train_idx],
            "y": bundle.y[train_idx],
        }
    )
    means = frame.groupby(["t", "r"])["y"].mean()
    out = []
    for t, r in zip(bundle.target_idx[test_idx], bundle.responder_idx[test_idx]):
        out.append(float(means.get((int(t), int(r)), global_mean)))
    return np.asarray(out, dtype=float)


def _svd_matrix_factorisation(
    bundle: FeatureBundle, train_idx: np.ndarray, test_idx: np.ndarray, *, rank: int = 16
) -> np.ndarray:
    n_t = len(bundle.target_names)
    n_r = len(bundle.responder_names)
    global_mean = float(np.mean(bundle.y[train_idx]))
    matrix = np.full((n_t, n_r), np.nan, dtype=float)
    frame = pd.DataFrame(
        {
            "t": bundle.target_idx[train_idx],
            "r": bundle.responder_idx[train_idx],
            "y": bundle.y[train_idx],
        }
    )
    for (t, r), value in frame.groupby(["t", "r"])["y"].mean().items():
        matrix[int(t), int(r)] = float(value)
    observed = np.isfinite(matrix)
    if observed.sum() < max(20, rank * 2):
        return np.full(len(test_idx), global_mean, dtype=float)
    centred = np.where(observed, matrix - global_mean, 0.0)
    rank = min(rank, min(centred.shape) - 1)
    svd = TruncatedSVD(n_components=rank, random_state=17)
    factors = svd.fit_transform(centred)
    recon = factors @ svd.components_ + global_mean
    out = recon[bundle.target_idx[test_idx], bundle.responder_idx[test_idx]]
    # For a target with no training pair at all, the SVD row is an artefact of
    # zero filling; fall back to the observed target mean or global mean.
    seen_targets = np.unique(bundle.target_idx[train_idx])
    target_means = frame.groupby("t")["y"].mean()
    for j, t in enumerate(bundle.target_idx[test_idx]):
        if int(t) not in set(seen_targets.tolist()):
            out[j] = float(target_means.get(int(t), global_mean))
    return np.asarray(out, dtype=float)


def predict_baselines(
    bundle: FeatureBundle,
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, BaselinePrediction]:
    global_mean = float(np.mean(bundle.y[train_idx]))
    target = _group_mean(bundle, train_idx, test_idx, column="target")
    responder = _group_mean(bundle, train_idx, test_idx, column="responder")
    return {
        "global_mean": BaselinePrediction("global_mean", np.full(len(test_idx), global_mean)),
        "target_mean": BaselinePrediction("target_mean", target),
        "responder_mean": BaselinePrediction("responder_mean", responder),
        "additive": BaselinePrediction("additive", target + responder - global_mean),
        "context_only": BaselinePrediction("context_only", _fit_context_ridge(bundle, train_idx, test_idx)),
        "connectome_ridge": BaselinePrediction(
            "connectome_ridge", _fit_pair_ridge(bundle, train_idx, test_idx)
        ),
        "svd_mf": BaselinePrediction("svd_mf", _svd_matrix_factorisation(bundle, train_idx, test_idx)),
        "oracle_pair_mean": BaselinePrediction("oracle_pair_mean", _pair_mean(bundle, train_idx, test_idx)),
    }
