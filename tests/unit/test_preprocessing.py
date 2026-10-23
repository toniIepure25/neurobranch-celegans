from __future__ import annotations

import numpy as np

from neurobranch.preprocessing import causal_highpass_segment, causal_moving_average, split_starts


def test_causal_moving_average_does_not_use_future():
    x = np.arange(40, dtype=float)[:, None]
    a = causal_moving_average(x, 5)
    x2 = x.copy()
    x2[30:] += 1000
    b = causal_moving_average(x2, 5)
    assert np.allclose(a[:30], b[:30])


def test_highpass_removes_linear_drift():
    t = np.arange(400, dtype=float)[:, None]
    x = 0.01 * t + np.sin(t / 10) * 0.2
    y = causal_highpass_segment(x, 80)
    assert abs(y[-100:].mean()) < abs(x[-100:].mean())
    assert np.isfinite(y).all()


def test_blocked_masks_are_disjoint(toy_data):
    m = toy_data.masks
    assert not np.any(m["train"] & m["val"])
    assert not np.any(m["train"] & m["test"])
    assert not np.any(m["val"] & m["test"])
    assert m["test"].sum() > 0 and m["train"].sum() > 0


def test_windows_do_not_cross_masks(toy_data):
    for split in ("train", "val", "test"):
        starts = split_starts(toy_data, split, history=4, horizon=2)
        for t in starts[:20]:
            assert toy_data.masks[split][t - 3 : t + 2].all()


def test_scaler_fit_on_train_only(toy_data):
    train = toy_data.masks["train"]
    assert np.allclose(toy_data.X[train].mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(toy_data.X[train].std(axis=0), 1.0, atol=1e-5)
