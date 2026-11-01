"""Support calibration keeps raw distances and q90/q95/q99 thresholds."""

from __future__ import annotations

import numpy as np

from neurobranch.simulator.adapters import SupportModel


def test_support_model_records_raw_scores_and_three_thresholds():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(200, 3))
    cal = rng.normal(size=(80, 3))
    support = SupportModel(train, calibration_X=cal, quantile=0.99)
    table = support.support_table(cal[:5])
    assert set(support.thresholds) == {"q90", "q95", "q99"}
    assert table["distance"].shape == (5,)
    for name in ("q90", "q95", "q99"):
        assert table[name].dtype == bool
    assert support.thresholds["q90"] <= support.thresholds["q95"] <= support.thresholds["q99"]
    assert support.threshold_at(0.95) == support.thresholds["q95"]


def test_threshold_audit_is_stricter_at_lower_quantiles():
    rng = np.random.default_rng(1)
    train = rng.normal(size=(300, 2))
    cal = rng.normal(size=(100, 2))
    support = SupportModel(train, calibration_X=cal, quantile=0.99)
    far = np.array([[10.0, 10.0]])
    table = support.support_table(far)
    assert not table["q99"][0]
    assert not table["q95"][0]
