"""Tests for the architecture-compatible small GRU."""

from __future__ import annotations

import numpy as np

from neurobranch.models.small_rnn import RNNConfig, SmallGRUDecoder, predict_rnn, train_rnn


def test_rnn_output_shape_and_probabilities() -> None:
    model = SmallGRUDecoder(2, hidden_size=3, dropout=0.5)
    x = np.random.default_rng(0).normal(size=(7, 4, 2)).astype(np.float32)
    import torch

    with torch.no_grad():
        p = model(torch.as_tensor(x))
    assert p.shape == (7, 4)
    assert np.all((p.numpy() >= 0) & (p.numpy() <= 1))


def test_rnn_rejects_wrong_input_schema() -> None:
    model = SmallGRUDecoder(2)
    import torch

    try:
        model(torch.zeros(3, 2))
    except ValueError:
        return
    raise AssertionError("wrong schema was accepted")


def test_rnn_training_and_event_aggregation() -> None:
    rng = np.random.default_rng(2)
    n_events = 24
    x = rng.normal(size=(n_events, 4, 2)).astype(np.float32)
    y = np.tile([0, 1], n_events // 2)
    groups = np.arange(n_events)
    model, info = train_rnn(
        x,
        y,
        groups,
        x,
        y,
        groups,
        config=RNNConfig(epochs=4, batch_size=8, patience=2, seed=2),
    )
    assert info["epochs_run"] >= 1
    p, out_groups = predict_rnn(model, x, groups)
    assert p.shape == (n_events,)
    assert np.array_equal(out_groups, groups)
