"""Tests for Phase 5 split integrity and protocol metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurobranch.behavior.protocols import (
    event_metrics,
    make_folds,
    make_sections,
    paired_cluster_bootstrap,
    section_balanced_accuracy,
)
from neurobranch.behavior.saav_alignment import dataset_from_arrays


def _dataset(n: int = 60, frames: int = 12) -> object:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(n, frames, 2)).astype(np.float32)
    y = np.tile([0, 1], n // 2)
    meta = pd.DataFrame(
        {
            "animal_id": [f"a{i % 10:02d}" for i in range(n)],
            "cohort": ["wt_octanol"] * (n // 2) + ["wt_diacetyl"] * (n - n // 2),
            "event_index": np.arange(n),
            "start": np.arange(n) * frames,
            "end": np.arange(n) * frames + frames,
            "direction": np.where(y == 1, "dorsal", "ventral"),
        }
    )
    return dataset_from_arrays(x, y, meta, ("hc", "saav"))


def test_original_folds_have_overlap_but_no_event_overlap() -> None:
    data = _dataset()
    folds = make_folds(data, "original", n_outer=5, n_inner=4, seed=3)
    assert len(folds) == 20
    for fold in folds:
        assert not (set(fold.train) & set(fold.valid))
        assert not (set(fold.train) & set(fold.test))
        assert not (set(fold.valid) & set(fold.test))
    assert any(fold.note == "ungrouped event split" for fold in folds)


def test_grouped_folds_never_share_animals() -> None:
    data = _dataset()
    animals = data.meta["animal_id"].to_numpy()
    for fold in make_folds(data, "grouped", n_outer=5, n_inner=2, seed=3):
        assert not set(animals[fold.train]) & set(animals[fold.test])
        assert not set(animals[fold.valid]) & set(animals[fold.test])


def test_chronological_test_is_latest_block_per_animal() -> None:
    data = _dataset()
    folds = make_folds(data, "chronological", n_outer=5, n_inner=2, seed=3)
    assert folds
    test = folds[0].test
    for animal in np.unique(data.meta["animal_id"]):
        idx = np.flatnonzero(data.meta["animal_id"].to_numpy() == animal)
        idx = idx[np.argsort(data.meta.iloc[idx]["start"].to_numpy())]
        expected = idx[-max(1, int(round(0.2 * len(idx)))) :]
        assert set(expected).issubset(set(test))


def test_external_train_and_test_cohorts_are_disjoint() -> None:
    data = _dataset()
    folds = make_folds(data, "external", seed=3)
    assert folds
    for fold in folds:
        assert set(data.meta.iloc[fold.train]["cohort"]) == {"wt_octanol"}
        assert set(data.meta.iloc[fold.test]["cohort"]) == {"wt_diacetyl"}


def test_sections_are_contiguous_and_cover_frames() -> None:
    x = np.arange(5 * 8 * 2, dtype=float).reshape(5, 8, 2)
    sections = make_sections(x, section_frames=4)
    assert sections.shape == (5, 5, 4, 2)
    assert np.allclose(sections[0, 0], x[0, :4])
    assert np.allclose(sections[0, -1], x[0, -4:])


def test_event_metrics_smoke() -> None:
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.3, 0.6])
    m = event_metrics(y, p)
    assert m["n"] == 4
    assert 0.0 <= m["auroc"] <= 1.0
    assert m["log_loss"] > 0


def test_section_balanced_accuracy_compares_predictions_with_labels() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.2, 0.4, 0.6, 0.8])
    assert section_balanced_accuracy(y, p, groups=np.arange(4)) == pytest.approx(1.0)


def test_paired_cluster_bootstrap_is_paired_and_clustered() -> None:
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 20)
    p_a = rng.uniform(size=40)
    p_b = np.clip(p_a + 0.05, 0, 1)
    groups = np.repeat(np.arange(10), 4)
    result = paired_cluster_bootstrap(y, p_a, p_b, groups, metric="brier", n_boot=100, seed=1)
    assert np.isfinite(result["delta"])
    assert result["ci_low"] <= result["ci_high"]
