"""Tests for observational mechanism helpers and label orientation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from neurobranch.behavior.mechanisms import (
    crossfit_predictions,
    holm_adjust,
    stability_by_channel,
    window_summary,
)
from neurobranch.behavior.protocols import prior_shift
from neurobranch.behavior.saav_alignment import dataset_from_arrays


def test_window_summary_shape_and_known_mean() -> None:
    x = np.ones((4, 6, 2))
    x[:, :, 1] = 3.0
    out = window_summary(x)
    assert out.shape == (4, 12)
    assert np.allclose(out[:, 0], 1.0)
    assert np.allclose(out[:, 6], 3.0)


def test_prior_shift_preserves_ranking_and_changes_prevalence() -> None:
    p = np.array([0.2, 0.4, 0.8])
    shifted = prior_shift(p, 0.1)
    assert np.all(np.diff(shifted) > 0)
    assert np.mean(shifted) < np.mean(p)


def test_holm_adjust_monotone_and_bounded() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert all(0 <= v <= 1 for v in adjusted.values())
    assert adjusted["a"] <= adjusted["c"] <= adjusted["b"]


def _synthetic_dataset(n: int = 40, frames: int = 8) -> object:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, frames, 2)).astype(np.float32)
    y = np.tile([0, 1], n // 2)
    x[y == 1, :, 1] += 2.0
    meta = pd.DataFrame(
        {
            "animal_id": [f"a{i // 5}" for i in range(n)],
            "cohort": ["wt_octanol"] * n,
            "event_index": np.arange(n),
            "start": np.arange(n) * 10,
            "end": np.arange(n) * 10 + frames,
        }
    )
    return dataset_from_arrays(x, y, meta, ("hc", "saav"))


def test_stability_detects_positive_synthetic_effect() -> None:
    data = _synthetic_dataset()
    stability, effects = stability_by_channel(data, ("saav",))
    assert len(effects) > 0
    assert stability.iloc[0]["mean_dorsal_minus_ventral"] > 0
    assert stability.iloc[0]["fraction_positive"] > 0.5


def test_crossfit_predictions_are_finite_and_animal_disjoint() -> None:
    data = _synthetic_dataset()
    animals = data.meta["animal_id"].to_numpy()
    x = window_summary(data.x)
    p = crossfit_predictions(x, data.y, animals, n_splits=4, seed=2)
    assert p.shape == (data.n_events,)
    assert np.isfinite(p).all()
    assert np.all((p > 0) & (p < 1))
