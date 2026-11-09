"""Save/load fitted simulator models as plain NumPy arrays.

No pickle and no TensorFlow dependency: the artifact is inspectable and portable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dataset import BEHAVIOR_NAMES
from .dynamics import BehaviorReadout, FittedModels, PolynomialRidgeDynamics


def save_models(models: FittedModels, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dynamics = models.dynamics
    behavior = models.behavior
    if any(
        value is None
        for value in (
            dynamics.feature_mean_,
            dynamics.feature_scale_,
            dynamics.target_mean_,
            dynamics.target_scale_,
            dynamics.coef_,
            dynamics.clip_min_,
            dynamics.clip_max_,
            behavior.coef_,
            behavior.intercept_,
            behavior.classes_,
            behavior.class_counts_,
        )
    ):
        raise RuntimeError("cannot save an unfitted model")
    metadata = {
        "format": "neurobranch-bundlenet-sim-v1",
        "train_worm_ids": list(models.train_worm_ids),
        "behavior_names": list(BEHAVIOR_NAMES),
        "alpha": dynamics.alpha,
        "scope": (
            "Quadratic latent dynamics + logistic behaviour read-out on published "
            "BunDLe-Net embeddings. Not a biophysical or causal model."
        ),
    }
    np.savez_compressed(
        path,
        feature_mean=dynamics.feature_mean_,
        feature_scale=dynamics.feature_scale_,
        target_mean=dynamics.target_mean_,
        target_scale=dynamics.target_scale_,
        dynamics_coef=dynamics.coef_,
        clip_min=dynamics.clip_min_,
        clip_max=dynamics.clip_max_,
        behavior_coef=behavior.coef_,
        behavior_intercept=behavior.intercept_,
        behavior_classes=behavior.classes_,
        behavior_class_counts=behavior.class_counts_,
        metadata=np.asarray(json.dumps(metadata)),
    )
    return path


def load_models(path: Path | str) -> FittedModels:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"]))
        dynamics = PolynomialRidgeDynamics(alpha=float(metadata["alpha"]))
        dynamics.feature_mean_ = payload["feature_mean"]
        dynamics.feature_scale_ = payload["feature_scale"]
        dynamics.target_mean_ = payload["target_mean"]
        dynamics.target_scale_ = payload["target_scale"]
        dynamics.coef_ = payload["dynamics_coef"]
        dynamics.clip_min_ = payload["clip_min"]
        dynamics.clip_max_ = payload["clip_max"]

        behavior = BehaviorReadout()
        behavior.coef_ = payload["behavior_coef"]
        behavior.intercept_ = payload["behavior_intercept"]
        behavior.classes_ = payload["behavior_classes"]
        behavior.class_counts_ = payload["behavior_class_counts"]

    return FittedModels(
        dynamics=dynamics,
        behavior=behavior,
        train_worm_ids=tuple(int(w) for w in metadata["train_worm_ids"]),
    )
