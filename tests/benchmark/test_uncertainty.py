"""Tests for the two-way bootstrap and its coverage parameter."""

from __future__ import annotations

import numpy as np
import pandas as pd

from neurobranch.design.calibration import simulate_atlas
from neurobranch.uncertainty import two_way_bootstrap


def _frame() -> pd.DataFrame:
    return simulate_atlas(
        n_animals=10,
        n_targets=2,
        n_responders=6,
        n_events=2,
        signal_sd=1.0,
        noise_sd=0.8,
        interaction_sd=0.4,
        seed=7,
    )


def test_two_way_bootstrap_reproducible_and_bounded() -> None:
    frame = _frame()
    first = two_way_bootstrap(frame, fold_size=10, n_boot=40, seed=3, min_animals=6)
    second = two_way_bootstrap(frame, fold_size=10, n_boot=40, seed=3, min_animals=6)
    assert first.summary["n_targets"] == 2
    assert first.summary["n_boot"] == 40
    assert first.summary["pooled_ci_low"] <= first.summary["pooled_ci_high"]
    assert 0.0 <= first.summary["pooled_ci_low"] <= 1.0
    assert 0.0 <= first.summary["pooled_ci_high"] <= 1.0
    assert np.allclose(first.draws, second.draws, equal_nan=True)
    assert first.by_target.equals(second.by_target)
    assert (first.by_target["n_bootstrap_valid"] == 40).all()


def test_two_way_bootstrap_coverage_filter_reduces_responders() -> None:
    frame = _frame()
    full = two_way_bootstrap(frame, n_boot=30, min_animals=6, min_responder_coverage=0.0)
    strict = two_way_bootstrap(frame, n_boot=30, min_animals=6, min_responder_coverage=1.0)
    assert strict.summary["min_responder_coverage"] == 1.0
    assert strict.by_target["n_responders"].max() <= full.by_target["n_responders"].max()
    assert strict.by_target["n_responders"].min() >= 3


def test_two_way_bootstrap_requires_enough_replicates() -> None:
    frame = _frame()
    try:
        two_way_bootstrap(frame, n_boot=5)
    except ValueError as error:
        assert "n_boot" in str(error)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected ValueError")
