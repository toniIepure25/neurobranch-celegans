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


def test_rnn_class_balancing_uses_training_labels_only() -> None:
    rng = np.random.default_rng(3)
    x_train = rng.normal(size=(8, 4, 2)).astype(np.float32)
    y_train = np.zeros(8, dtype=int)
    groups_train = np.arange(8)
    x_valid = rng.normal(size=(4, 4, 2)).astype(np.float32)
    y_valid = np.array([0, 1, 0, 1])
    groups_valid = np.arange(4)
    try:
        train_rnn(
            x_train,
            y_train,
            groups_train,
            x_valid,
            y_valid,
            groups_valid,
            config=RNNConfig(epochs=2, batch_size=4, patience=1, seed=3),
        )
    except ValueError as error:
        assert "class-balanced" in str(error)
        return
    raise AssertionError("training accepted a single-class training set")
