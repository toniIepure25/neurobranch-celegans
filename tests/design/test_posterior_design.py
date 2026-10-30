"""Tests for the posterior-predictive design table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from neurobranch.benchmark.hierarchical import HierarchicalResult
from neurobranch.design.posterior_design import design_summary, design_table, reliability_curve


def _result() -> HierarchicalResult:
    rng = np.random.default_rng(1)
    draws = np.column_stack([rng.uniform(0.4, 0.6, 500), rng.uniform(0.05, 0.15, 500)])
    by_target = pd.DataFrame(
        [
            {"target": "T1", "status": "ok", "mean_sampling_var": 0.1},
            {"target": "T2", "status": "ok", "mean_sampling_var": 0.1},
        ]
    )
    return HierarchicalResult(
        by_target=by_target,
        summary={"status": "ok"},
        draws_by_target={"T1": draws, "T2": draws},
        status="ok",
        pooled_reliability_median=0.8,
        single_animal_reliability_median=0.4,
        pooled_ceiling_median=0.89,
    )


def test_reliability_curve_is_monotone() -> None:
    curve = reliability_curve(np.array([0.5]), np.array([0.1]), 0.1, max_animals=12)
    assert curve["n_animals"].tolist() == list(range(1, 13))
    assert curve["reliability_median"].is_monotonic_increasing
    assert (curve["reliability_low"] <= curve["reliability_median"]).all()
    assert (curve["reliability_median"] <= curve["reliability_high"]).all()


def test_design_table_reaches_moderate_target() -> None:
    frame = design_table(_result(), target_reliability=0.5, probability=0.6, max_animals=40)
    assert len(frame) == 2
    assert frame["status"].eq("ok").all()
    assert (frame["required_animals"] >= 1).all()
    summary = design_summary(frame)
    assert summary["status"] == "ok"
    assert summary["n_reached"] == 2
