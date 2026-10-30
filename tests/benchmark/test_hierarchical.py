"""Tests for the hierarchical reliability model.

The model is an additive measurement model on standardized cell means:
``y_ar = mu_r + u_ar + e_ar`` with ``mu_r ~ N(0, sigma^2)``,
``u_ar ~ N(0, tau^2)`` and known sampling variance ``v_ar``.  These tests use
the existing synthetic atlas generator so the generating model and the fitted
model share the same additive structure; they do not claim robustness to
misspecification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurobranch.benchmark.hierarchical import (
    TargetData,
    fit_hierarchical_reliability,
    fit_hierarchical_target,
    posterior_ceiling,
)
from neurobranch.design.calibration import simulate_atlas


def _fit(frame: pd.DataFrame, **kwargs: object) -> object:
    defaults = {
        "coverage": 0.0,
        "n_eval": (1, 22),
        "chains": 2,
        "warmup": 250,
        "draws": 350,
        "seed": 7,
    }
    defaults.update(kwargs)
    return fit_hierarchical_reliability(frame, **defaults)


def test_strong_signal_recovers_high_pooled_reliability() -> None:
    frame = simulate_atlas(
        n_animals=16,
        n_targets=2,
        n_responders=10,
        n_events=4,
        signal_sd=2.0,
        noise_sd=0.5,
        interaction_sd=0.2,
        seed=3,
    )
    result = _fit(frame)
    assert result.status == "ok"
    assert result.by_target["status"].eq("ok").all()
    # A strong shared profile should survive pooling far better than one animal.
    assert result.pooled_reliability_median > 0.7
    assert result.single_animal_reliability_median < result.pooled_reliability_median
    assert np.isfinite(result.pooled_ceiling_median)
    assert 0.0 <= result.pooled_ceiling_median <= 1.0


def test_weak_signal_is_lower_than_strong_signal() -> None:
    strong = simulate_atlas(
        n_animals=14,
        n_targets=2,
        n_responders=10,
        signal_sd=2.0,
        noise_sd=0.5,
        interaction_sd=0.2,
        seed=4,
    )
    weak = simulate_atlas(
        n_animals=14,
        n_targets=2,
        n_responders=10,
        signal_sd=0.3,
        noise_sd=1.5,
        interaction_sd=1.0,
        seed=4,
    )
    strong_result = _fit(strong)
    weak_result = _fit(weak)
    assert strong_result.pooled_reliability_median > weak_result.pooled_reliability_median


def test_reliability_is_monotone_in_animal_count() -> None:
    frame = simulate_atlas(
        n_animals=12,
        n_targets=2,
        n_responders=8,
        signal_sd=1.0,
        noise_sd=1.0,
        interaction_sd=0.5,
        seed=5,
    )
    result = _fit(frame, n_eval=(1, 2, 4, 8, 22))
    for target in result.by_target["target"]:
        row = result.by_target.set_index("target").loc[target]
        values = [row[f"reliability_n{n}"] for n in (1, 2, 4, 8, 22)]
        assert values == sorted(values)
        assert 0.0 <= values[0] <= values[-1] <= 1.0


def test_seed_reproducibility() -> None:
    frame = simulate_atlas(n_animals=12, n_targets=2, n_responders=8, seed=6)
    first = _fit(frame, seed=11)
    second = _fit(frame, seed=11)
    assert first.pooled_reliability_median == pytest.approx(second.pooled_reliability_median)
    assert first.by_target["reliability_n22"].tolist() == pytest.approx(
        second.by_target["reliability_n22"].tolist()
    )


def test_too_few_responders_is_reported_not_silently_zero() -> None:
    frame = simulate_atlas(n_animals=10, n_targets=1, n_responders=2, n_events=3, seed=8)
    result = _fit(frame)
    assert result.status in {"no_eligible_targets", "ok_with_failures"}
    assert result.by_target.empty or not result.by_target["status"].eq("ok").any()


def test_ceiling_is_sqrt_of_reliability() -> None:
    frame = simulate_atlas(n_animals=12, n_targets=2, n_responders=8, seed=9)
    result = _fit(frame)
    assert posterior_ceiling(0.49) == pytest.approx(0.7)
    row = result.by_target.iloc[0]
    assert row["ceiling_n22"] == pytest.approx(np.sqrt(row["reliability_n22"]))


def _target_data(v: np.ndarray | None = None) -> TargetData:
    rng = np.random.default_rng(101)
    y = rng.normal(size=(10, 8))
    if v is None:
        v = np.full_like(y, 0.25)
    mask = np.ones_like(y, dtype=bool)
    return TargetData(
        target="synthetic",
        y=y,
        v=v,
        mask=mask,
        n_animals=10,
        n_responders=8,
        coverage=1.0,
        nu=float(np.mean(v[mask])),
        responders=tuple(f"n{i}" for i in range(8)),
    )


def test_known_sampling_variance_is_used_by_the_fit() -> None:
    data = _target_data()
    fit = fit_hierarchical_target(data, draws=300, grid_resolution=30, seed=3)
    assert fit.status == "ok"
    assert fit.row["mean_sampling_var"] == pytest.approx(0.25)
    assert fit.row["n_variance_floored"] == 0


def test_zero_sampling_variance_is_floored_and_reported() -> None:
    data = _target_data()
    v = data.v.copy()
    v[0, 0] = 0.0
    data = TargetData(
        target=data.target,
        y=data.y,
        v=v,
        mask=data.mask,
        n_animals=data.n_animals,
        n_responders=data.n_responders,
        coverage=data.coverage,
        nu=float(np.mean(v[data.mask])),
        responders=data.responders,
    )
    fit = fit_hierarchical_target(data, draws=300, grid_resolution=30, seed=3)
    assert fit.row["n_variance_floored"] == 1
    assert fit.row["mean_sampling_var"] > 0.0


def test_invalid_sampling_variance_fails_closed() -> None:
    data = _target_data()
    v = data.v.copy()
    v[0, 0] = np.nan
    bad = TargetData(
        target=data.target,
        y=data.y,
        v=v,
        mask=data.mask,
        n_animals=data.n_animals,
        n_responders=data.n_responders,
        coverage=data.coverage,
        nu=float(np.nanmean(v[data.mask])),
        responders=data.responders,
    )
    with pytest.raises(ValueError, match="sampling variances"):
        fit_hierarchical_target(bad, draws=300, grid_resolution=30, seed=3)
