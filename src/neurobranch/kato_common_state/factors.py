"""Train-only common population factors for the Kato control audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class FactorModel:
    pca: PCA
    n_features: int
    n_components: int
    fit_record: dict[str, Any]

    def scores(self, X: np.ndarray, k: int) -> np.ndarray:
        k = int(k)
        if k < 1 or k > self.n_components:
            raise ValueError(f"k must be in [1, {self.n_components}]")
        arr = np.asarray(X, dtype=np.float64)
        return (arr - self.pca.mean_) @ self.pca.components_[:k].T

    def common(self, X: np.ndarray, k: int) -> np.ndarray:
        f = self.scores(X, k)
        return f @ self.pca.components_[:k]

    def residual(self, X: np.ndarray, k: int) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        return arr - self.pca.mean_ - self.common(arr, k)


def fit_factor_model(X_train: np.ndarray, *, n_components: int = 10, seed: int = 17) -> FactorModel:
    train = np.asarray(X_train, dtype=np.float64)
    if train.ndim != 2 or len(train) < 10:
        raise ValueError("training matrix must be two-dimensional with at least 10 frames")
    n_components = int(min(n_components, train.shape[1]))
    pca = PCA(n_components=n_components, random_state=int(seed)).fit(train)
    return FactorModel(
        pca=pca,
        n_features=int(train.shape[1]),
        n_components=n_components,
        fit_record={
            "fit_split": "train",
            "n_train_frames": int(len(train)),
            "seed": int(seed),
            "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        },
    )


def variance_table(model: FactorModel, ks: tuple[int, ...]) -> list[dict[str, float]]:
    cumulative = np.cumsum(model.pca.explained_variance_ratio_)
    rows = []
    for k in ks:
        k = int(k)
        rows.append(
            {
                "k": k,
                "explained_variance_fraction": float(cumulative[k - 1]),
                "incremental_variance_fraction": float(model.pca.explained_variance_ratio_[k - 1]),
            }
        )
    return rows
