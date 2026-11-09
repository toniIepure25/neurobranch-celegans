"""Held-out-worm evaluation for the BunDLe-Net latent simulator."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from .dataset import BEHAVIOR_NAMES, WormTrajectory
from .dynamics import FittedModels, fit_models


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean(axis=0)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _score_worm(
    models: FittedModels,
    test: WormTrajectory,
    horizon: int | None = None,
) -> dict[str, float | int]:
    x = test.latent[:-1]
    y = test.latent[1:]
    one_step = models.dynamics.predict(x)

    n = len(test.latent) if horizon is None else min(len(test.latent), horizon)
    free = models.dynamics.rollout(test.latent[0], n_steps=n - 1)
    truth = test.latent[1:n]
    # Free-run no-dynamics baseline: stay at the initial state.  Comparing a
    # rollout with one-step persistence would be an invalid, teacher-forced
    # baseline and would make every honest rollout look catastrophic.
    persistence = np.repeat(test.latent[0][None, :], n - 1, axis=0)

    true_behavior = test.behavior[1:n]
    decoded_true = models.behavior.predict(test.latent[1:n])
    decoded_sim = models.behavior.predict(free)

    majority = models.behavior.majority_class_
    majority_pred = np.full_like(true_behavior, majority)

    return {
        "worm_id": test.worm_id,
        "n_frames": int(len(test.latent)),
        "one_step_r2": r2_score_np(y, one_step),
        "free_run_mse": float(np.mean((free - truth) ** 2)),
        "persistence_mse": float(np.mean((persistence - truth) ** 2)),
        "free_run_skill_vs_persistence": float(
            1.0 - np.mean((free - truth) ** 2) / max(np.mean((persistence - truth) ** 2), 1e-12)
        ),
        "behavior_accuracy_on_true_latent": float(accuracy_score(true_behavior, decoded_true)),
        "behavior_balanced_accuracy_on_true_latent": float(
            balanced_accuracy_score(true_behavior, decoded_true)
        ),
        "behavior_macro_f1_on_true_latent": float(
            f1_score(true_behavior, decoded_true, average="macro", zero_division=0)
        ),
        "behavior_accuracy_on_simulated_latent": float(accuracy_score(true_behavior, decoded_sim)),
        "behavior_majority_accuracy": float(accuracy_score(true_behavior, majority_pred)),
    }


@dataclass(frozen=True)
class EvaluationSummary:
    alpha: float
    n_worms: int
    n_behaviors: int
    folds: tuple[dict[str, float | int], ...]
    mean_one_step_r2: float
    mean_free_run_skill: float
    mean_behavior_accuracy_true: float
    mean_behavior_accuracy_simulated: float
    mean_behavior_majority_accuracy: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["behavior_names"] = list(BEHAVIOR_NAMES)
        payload["scope"] = (
            "Held-out-worm evaluation of a quadratic latent dynamics model and a "
            "logistic behaviour read-out fitted to published BunDLe-Net embeddings. "
            "This validates latent predictability, not mechanism or causality."
        )
        return payload


def evaluate_leave_one_worm_out(
    trajectories: list[WormTrajectory],
    alpha: float = 1.0,
) -> EvaluationSummary:
    """Train on four worms, test on the fifth; repeat for all five."""
    folds: list[dict[str, float | int]] = []
    for test in trajectories:
        train_ids = tuple(t.worm_id for t in trajectories if t.worm_id != test.worm_id)
        models = fit_models(trajectories, alpha=alpha, train_worm_ids=train_ids)
        folds.append(_score_worm(models, test))
    return EvaluationSummary(
        alpha=alpha,
        n_worms=len(trajectories),
        n_behaviors=len(BEHAVIOR_NAMES),
        folds=tuple(folds),
        mean_one_step_r2=float(np.mean([f["one_step_r2"] for f in folds])),
        mean_free_run_skill=float(np.mean([f["free_run_skill_vs_persistence"] for f in folds])),
        mean_behavior_accuracy_true=float(
            np.mean([f["behavior_accuracy_on_true_latent"] for f in folds])
        ),
        mean_behavior_accuracy_simulated=float(
            np.mean([f["behavior_accuracy_on_simulated_latent"] for f in folds])
        ),
        mean_behavior_majority_accuracy=float(np.mean([f["behavior_majority_accuracy"] for f in folds])),
    )


def save_summary(summary: EvaluationSummary, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    return path
