from __future__ import annotations

import numpy as np
import pandas as pd

from neurobranch.evaluation.randi_benchmark import (
    BenchmarkConfig,
    _cluster_bootstrap,
    response_reliability,
)


def test_cluster_bootstrap_returns_ordered_intervals() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=100)
    yhat = y + rng.normal(scale=0.5, size=100)
    groups = np.repeat(np.arange(20), 5)
    ci = _cluster_bootstrap(y, yhat, groups, n_bootstrap=50, seed=0)
    assert ci["mae_ci_low"] <= ci["mae_ci_high"]
    assert ci["corr_ci_low"] <= ci["corr_ci_high"]


def test_response_reliability_detects_identical_profiles() -> None:
    rows = []
    for event in [0, 1]:
        for neuron, val in [("A", 1.0), ("B", 2.0), ("C", -1.0)]:
            rows.append(
                {
                    "dataset_id": 0,
                    "event_id": f"0:{event}",
                    "target": "X",
                    "neuron": neuron,
                    "response": val,
                }
            )
    out = response_reliability(pd.DataFrame(rows), response_column="response")
    assert out["n_pairs"] == 1
    assert abs(out["median_pairwise_r"] - 1.0) < 1e-12


def test_benchmark_config_defaults_are_explicit() -> None:
    cfg = BenchmarkConfig()
    assert cfg.response_column == "response"
    assert cfg.n_bootstrap > 0
    assert cfg.alpha > 0
