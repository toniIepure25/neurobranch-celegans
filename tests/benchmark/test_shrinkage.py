"""Tests for empirical-Bayes reliability shrinkage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from neurobranch.benchmark.reliability import shrink_reliability


def test_shrink_preserves_raw_and_bounds() -> None:
    table = pd.DataFrame(
        {
            "target": [f"T{i}" for i in range(8)],
            "single_animal_reliability": np.linspace(0.05, 0.75, 8),
            "n_animals": np.full(8, 10.0),
        }
    )
    out = shrink_reliability(table)
    assert "single_animal_reliability" in out.columns
    assert "single_animal_reliability_shrunk" in out.columns
    assert out["single_animal_reliability"].equals(table["single_animal_reliability"])
    shrunk = out["single_animal_reliability_shrunk"].to_numpy(dtype=float)
    assert np.isfinite(shrunk).all()
    assert ((shrunk > 0) & (shrunk < 1)).all()
    assert (out["shrink_weight"].to_numpy(dtype=float) >= 0).all()


def test_shrink_without_between_variance_collapses_to_mean() -> None:
    table = pd.DataFrame(
        {
            "single_animal_reliability": [0.2, 0.2, 0.2],
            "n_animals": [8.0, 8.0, 8.0],
        }
    )
    out = shrink_reliability(table)
    values = out["single_animal_reliability_shrunk"].to_numpy(dtype=float)
    assert np.allclose(values, values[0])


def test_shrink_missing_column_raises() -> None:
    table = pd.DataFrame({"single_animal_reliability": [0.2]})
    try:
        shrink_reliability(table)
    except KeyError as error:
        assert "n_animals" in str(error)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected KeyError")
