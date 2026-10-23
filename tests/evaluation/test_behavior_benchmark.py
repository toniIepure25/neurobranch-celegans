from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurobranch.evaluation.behavior_benchmark import (
    _markov_predict,
    _metrics,
    incremental_contrasts,
)


def test_markov_probabilities_are_rows_and_argmax() -> None:
    cur = np.array([0, 0, 1, 1])
    nxt = np.array([0, 1, 1, 1])
    pred, proba = _markov_predict(cur, nxt, np.array([0, 1]))
    assert proba.shape == (2, 4)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.array_equal(pred, proba.argmax(axis=1))


def test_metrics_reward_perfect_predictions() -> None:
    y = np.array([0, 1, 2, 3])
    current = np.array([0, 0, 1, 1])
    proba = np.eye(4)[y]
    m = _metrics(y, y, proba, current)
    assert m["accuracy"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["log_loss"] < 1e-6
    assert m["n_changed"] == 3


def test_incremental_contrast_marks_wins() -> None:
    rows = [
        {"worm_id": 0, "model": "logistic_state", "balanced_accuracy": 0.5, "macro_f1": 0.5, "log_loss": 1.0, "brier": 0.5, "change_accuracy": 0.2},
        {"worm_id": 1, "model": "logistic_state", "balanced_accuracy": 0.5, "macro_f1": 0.5, "log_loss": 1.0, "brier": 0.5, "change_accuracy": 0.2},
        {"worm_id": 0, "model": "logistic_state_neural", "balanced_accuracy": 0.6, "macro_f1": 0.6, "log_loss": 0.9, "brier": 0.4, "change_accuracy": 0.3},
        {"worm_id": 1, "model": "logistic_state_neural", "balanced_accuracy": 0.4, "macro_f1": 0.4, "log_loss": 1.1, "brier": 0.6, "change_accuracy": 0.1},
    ]
    d = incremental_contrasts(pd.DataFrame(rows))
    bal = d[d.metric == "balanced_accuracy"].iloc[0]
    assert bal["mean_delta"] == pytest.approx(0.0)
    assert bal["wins"] == 1
    ll = d[d.metric == "log_loss"].iloc[0]
    assert ll["mean_delta"] == pytest.approx(0.0)
    assert ll["wins"] == 1
