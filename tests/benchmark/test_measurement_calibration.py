"""Smoke tests for Phase 13 calibration and null controls."""

from __future__ import annotations

import numpy as np

from neurobranch.benchmark.hierarchical import TargetData
from neurobranch.benchmark.measurement_calibration import (
    _simulate_template,
    permutation_control,
    simulation_based_calibration,
)


def _template(index: int) -> TargetData:
    rng = np.random.default_rng(index)
    animals, responders = 8, 5
    n = rng.integers(1, 4, size=(animals, responders)).astype(float)
    v = 0.25 / np.maximum(n, 1.0)
    return TargetData(
        target=f"t{index}",
        y=rng.normal(size=(animals, responders)),
        v=v,
        mask=np.ones((animals, responders), dtype=bool),
        n_animals=animals,
        n_responders=responders,
        coverage=1.0,
        nu=float(v.mean()),
        responders=tuple(f"n{i}" for i in range(responders)),
        n_events=n,
        cell_var=np.where(n >= 2.0, 0.5, 0.0),
    )


def test_simulate_template_preserves_shapes_and_event_variance() -> None:
    template = _template(1)
    simulated = _simulate_template(
        template,
        rng=np.random.default_rng(2),
        sigma=0.5,
        tau=0.5,
        omega=0.7,
        model="m2_event_variance",
        target_id="sim",
    )
    assert simulated.y.shape == template.y.shape
    assert simulated.mask.sum() == template.mask.sum()
    assert simulated.n_events is not None
    assert simulated.cell_var is not None
    assert np.all(simulated.cell_var[simulated.mask] >= 0.0)
    assert simulated.nu > 0.0


def test_small_sbc_returns_parameter_summaries() -> None:
    templates = [_template(index) for index in range(12)]
    output = simulation_based_calibration(
        templates,
        model="m1_projection_scaled",
        n_reps=10,
        targets_per_rep=1,
        draws=100,
        grid_resolution=20,
        event_grid=40,
        seed=5,
    )
    assert output["n_fits"] == 10
    assert output["parameters"]["n22"]["status"] == "ok"
    assert 0.0 <= output["parameters"]["n22"]["coverage"] <= 1.0


def test_permutation_control_reports_a_null_distribution() -> None:
    templates = [_template(index) for index in range(12)]
    output = permutation_control(
        templates,
        model="m1_projection_scaled",
        n_targets=3,
        draws=100,
        grid_resolution=20,
        event_grid=40,
        seed=6,
    )
    assert output["n_requested"] == 3
    assert output["n_eligible"] >= 1
    assert np.isfinite(output["median_reliability_n22"])
