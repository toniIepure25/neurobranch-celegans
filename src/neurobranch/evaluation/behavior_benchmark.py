"""Incremental behaviour prediction benchmark on held-out Kato time blocks.

The question is not "can a classifier label the current behaviour?" but whether
neural activity adds information beyond the current behavioural state.  A
state-only Markov/logistic model is therefore the primary baseline, and
transition-only metrics are reported separately from the dominant persistence
class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler

from neurobranch.preprocessing import WormData, split_starts


@dataclass
class BehaviorConfig:
    pca_dim: int = 20
    max_iter: int = 2000
    C: float = 1.0
    seed: int = 0
    n_bootstrap: int = 200


def _onehot(y: np.ndarray, n_classes: int = 4) -> np.ndarray:
    return np.eye(n_classes, dtype=float)[y]


def _pca_features(
    train_X: np.ndarray, test_X: np.ndarray, dim: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(train_X)
    pca = PCA(n_components=min(dim, train_X.shape[1], train_X.shape[0] - 1), random_state=seed, svd_solver="randomized")
    pca.fit(scaler.transform(train_X))
    return pca.transform(scaler.transform(train_X)), pca.transform(scaler.transform(test_X))


def _multiclass_brier(y: np.ndarray, proba: np.ndarray) -> float:
    onehot = _onehot(y)
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def _metrics(y: np.ndarray, pred: np.ndarray, proba: np.ndarray, current: np.ndarray) -> dict[str, Any]:
    labels = np.arange(4)
    changed = y != current
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        "log_loss": float(log_loss(y, np.clip(proba, 1e-9, 1), labels=labels)),
        "brier": _multiclass_brier(y, proba),
        "persistence_accuracy": float(np.mean(current == y)),
        "n": int(len(y)),
        "n_changed": int(changed.sum()),
        "change_accuracy": float(np.mean(pred[changed] == y[changed])) if changed.any() else float("nan"),
        "change_recall": float(np.mean(pred[changed] == y[changed])) if changed.any() else float("nan"),
    }


def _fit_predict(model: LogisticRegression, Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = clone(model)
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)
    # Some folds can miss a class; expand to the fixed four-class order.
    if proba.shape[1] != 4:
        full = np.zeros((len(Xte), 4))
        full[:, model.classes_.astype(int)] = proba
        proba = full
    return proba.argmax(axis=1), proba


def _markov_predict(train_cur: np.ndarray, train_next: np.ndarray, test_cur: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = np.full((4, 4), 0.5)
    for a, b in zip(train_cur, train_next):
        counts[int(a), int(b)] += 1
    probs = counts / counts.sum(axis=1, keepdims=True)
    proba = probs[test_cur.astype(int)]
    return proba.argmax(axis=1), proba


def evaluate_worm_behavior(
    data: WormData,
    *,
    cfg: BehaviorConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or BehaviorConfig()
    train_starts = split_starts(data, "train", history=1, horizon=1)
    test_starts = split_starts(data, "test", history=1, horizon=1)
    if len(train_starts) < 50 or len(test_starts) < 20:
        raise ValueError(f"worm {data.worm_id}: too few starts")
    tr_cur = data.y_macro[train_starts]
    tr_next = data.y_macro[train_starts + 1]
    te_cur = data.y_macro[test_starts]
    te_next = data.y_macro[test_starts + 1]

    Xtr_neural, Xte_neural = _pca_features(
        data.X[train_starts].astype(float), data.X[test_starts].astype(float), cfg.pca_dim, cfg.seed
    )
    Xtr_state = _onehot(tr_cur)
    Xte_state = _onehot(te_cur)
    Xtr_both = np.hstack([Xtr_state, Xtr_neural])
    Xte_both = np.hstack([Xte_state, Xte_neural])

    rows = []
    # Persistence has no fitted parameters.
    for model_name, pred, proba in [
        ("persistence", te_cur, _onehot(te_cur)),
    ]:
        m = _metrics(te_next, pred, proba, te_cur)
        m.update({"worm_id": data.worm_id, "model": model_name, "split": "test"})
        rows.append(m)

    pred, proba = _markov_predict(tr_cur, tr_next, te_cur)
    m = _metrics(te_next, pred, proba, te_cur)
    m.update({"worm_id": data.worm_id, "model": "markov_state", "split": "test"})
    rows.append(m)

    logit = LogisticRegression(max_iter=cfg.max_iter, C=cfg.C, solver="lbfgs")
    for model_name, Xtr, Xte in [
        ("logistic_state", Xtr_state, Xte_state),
        ("logistic_neural", Xtr_neural, Xte_neural),
        ("logistic_state_neural", Xtr_both, Xte_both),
    ]:
        pred, proba = _fit_predict(logit, Xtr, tr_next, Xte)
        m = _metrics(te_next, pred, proba, te_cur)
        m.update({"worm_id": data.worm_id, "model": model_name, "split": "test"})
        rows.append(m)
    return pd.DataFrame(rows)


def summarize_behavior(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "log_loss",
        "brier",
        "persistence_accuracy",
        "change_accuracy",
        "n",
        "n_changed",
    ]
    out = []
    for model, g in results.groupby("model"):
        row: dict[str, Any] = {"model": model, "n_worms": int(g["worm_id"].nunique())}
        for m in metrics:
            row[f"{m}_mean"] = float(g[m].mean())
            row[f"{m}_sd"] = float(g[m].std(ddof=1))
        out.append(row)
    return pd.DataFrame(out).sort_values("model").reset_index(drop=True)


def incremental_contrasts(results: pd.DataFrame) -> pd.DataFrame:
    """Paired per-worm differences for neural increment over state-only models."""
    base = results[results.model == "logistic_state"].set_index("worm_id")
    rows = []
    for model in ["logistic_neural", "logistic_state_neural"]:
        other = results[results.model == model].set_index("worm_id")
        common = base.index.intersection(other.index)
        if len(common) == 0:
            continue
        for metric in ["balanced_accuracy", "macro_f1", "log_loss", "brier", "change_accuracy"]:
            d = other.loc[common, metric] - base.loc[common, metric]
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean_delta": float(d.mean()),
                    "sd_delta": float(d.std(ddof=1)),
                    "wins": int((d > 0).sum()) if metric not in {"log_loss", "brier"} else int((d < 0).sum()),
                    "n_worms": int(len(common)),
                    "per_worm": {str(k): float(v) for k, v in d.items()},
                }
            )
    return pd.DataFrame(rows)


def bootstrap_change_accuracy(
    data: WormData, results: pd.DataFrame, *, n_bootstrap: int = 200, seed: int = 0
) -> dict[str, Any]:
    """Block bootstrap over contiguous test frames for the change subset."""
    # Reconstructing per-frame predictions is intentionally avoided here; the
    # per-worm contrast is the primary uncertainty statement.  This helper
    # records the test-set change prevalence, which drives the metric's noise.
    test_starts = split_starts(data, "test", history=1, horizon=1)
    changed = data.y_macro[test_starts + 1] != data.y_macro[test_starts]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(changed), len(changed))
        vals.append(float(changed[idx].mean()))
    return {
        "change_prevalence": float(changed.mean()),
        "change_prevalence_ci_low": float(np.quantile(vals, 0.025)),
        "change_prevalence_ci_high": float(np.quantile(vals, 0.975)),
        "n_test": int(len(changed)),
    }
