"""Tests for the Phase 13 measurement models."""

from __future__ import annotations

import numpy as np
import pytest

from neurobranch.benchmark.hierarchical import TargetData
from neurobranch.benchmark.measurement import (
    MeasurementConfig,
    estimate_event_variance,
    fit_measurement_target,
)


def _target(
    *,
    n_animals: int = 14,
    n_responders: int = 8,
    n_events: float = 3.0,
    v: float = 0.25,
    seed: int = 7,
) -> TargetData:
    rng = np.random.default_rng(seed)
    y = rng.normal(size=(n_animals, n_responders))
    mask = np.ones_like(y, dtype=bool)
    counts = np.full_like(y, n_events)
    cell_var = np.full_like(y, v * n_events)
    return TargetData(
        target="synthetic",
        y=y,
        v=np.full_like(y, v),
        mask=mask,
        n_animals=n_animals,
        n_responders=n_responders,
        coverage=1.0,
        nu=v,
        responders=tuple(f"n{i}" for i in range(n_responders)),
        n_events=counts,
        cell_var=cell_var,
    )


def test_projection_scaling_uses_within_animal_responder_count() -> None:
    data = _target()
    fit = fit_measurement_target(
        data,
        config=MeasurementConfig(model="m1_projection_scaled", draws=200, grid_resolution=30),
    )
    assert fit.status == "ok"
    assert fit.row["mean_sampling_var"] == pytest.approx(0.25 * (1.0 - 1.0 / 8.0))


def test_event_variance_recovers_a_known_scale() -> None:
    data = _target(n_events=4.0, v=0.2)
    # cell_var = s^2 = 0.8; the event variance is 0.8 / 3?  The input is set so
    # omega^2 = 0.8 and v = omega^2 / n = 0.2.
    posterior = estimate_event_variance(data, prior_scale=1.0, grid_size=300)
    assert posterior.n_variance_cells == data.mask.sum()
    assert posterior.n_one_event_cells == 0
    assert 0.5 < posterior.omega2_median < 1.2
    assert posterior.omega2_ci_low <= posterior.omega2_median <= posterior.omega2_ci_high


def test_event_model_removes_the_variance_floor() -> None:
    data = _target(n_events=1.0, v=0.0)
    data = TargetData(
        target=data.target,
        y=data.y,
        v=data.v,
        mask=data.mask,
        n_animals=data.n_animals,
        n_responders=data.n_responders,
        coverage=data.coverage,
        nu=1e-8,
        responders=data.responders,
        n_events=data.n_events,
        cell_var=np.zeros_like(data.y),
        n_variance_floored=int(data.mask.sum()),
    )
    fit = fit_measurement_target(
        data,
        config=MeasurementConfig(model="m2_event_variance", draws=200, grid_resolution=30),
    )
    assert fit.status == "ok"
    assert fit.row["n_variance_floored"] == 0
    assert fit.row["mean_sampling_var"] > 1e-6


def test_single_responder_animals_are_dropped() -> None:
    data = _target(n_animals=8, n_responders=5)
    mask = data.mask.copy()
    mask[0, :] = False
    mask[0, 0] = True
    data = TargetData(
        target=data.target,
        y=data.y,
        v=data.v,
        mask=mask,
        n_animals=data.n_animals,
        n_responders=data.n_responders,
        coverage=data.coverage,
        nu=float(data.v[mask].mean()),
        responders=data.responders,
        n_events=data.n_events,
        cell_var=data.cell_var,
    )
    fit = fit_measurement_target(
        data,
        config=MeasurementConfig(model="m1_projection_scaled", draws=200, grid_resolution=30),
    )
    assert fit.row["n_single_responder_animals_dropped"] == 1
    assert fit.row["n_animals"] == 7


def test_unknown_model_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown measurement model"):
        MeasurementConfig(model="not-a-model")
