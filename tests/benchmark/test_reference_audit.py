"""Tests for the Phase 14 reference-audit variance options."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurobranch.benchmark.variance import (
    cell_summaries,
    profile_components_by_target,
    profile_components_from_cells,
)


def _trials() -> pd.DataFrame:
    rows = []
    for animal in ("a0", "a1"):
        for neuron in ("n0", "n1", "n2"):
            n_events = 1 if (animal == "a0" and neuron == "n0") else 2
            for event in range(n_events):
                rows.append(
                    {
                        "dataset_id": animal,
                        "target": "t0",
                        "neuron": neuron,
                        "event_id": f"{animal}-{neuron}-{event}",
                        "response_mean": float(event + len(neuron) + (animal == "a1")),
                    }
                )
    return pd.DataFrame(rows)


def test_one_event_variance_override_is_applied_only_to_single_events() -> None:
    frame = _trials()
    cells = cell_summaries(frame)
    single = cells[cells["n"] == 1]
    assert not single.empty
    assert np.allclose(single["var"], 0.0)

    overridden = cell_summaries(frame, one_event_variance=1.75)
    assert np.allclose(overridden.loc[overridden["n"] == 1, "var"], 1.75)
    assert np.allclose(
        overridden.loc[overridden["n"] >= 2, "var"],
        cells.loc[cells["n"] >= 2, "var"],
    )


def test_homogeneous_event_variance_replaces_every_cell() -> None:
    cells = cell_summaries(_trials(), event_variance=2.5)
    assert np.allclose(cells["var"], 2.5)


def test_variance_options_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError):
        cell_summaries(_trials(), one_event_variance=1.0, event_variance=2.0)


def test_components_from_cells_match_frame_estimator() -> None:
    frame = _trials()
    expected = profile_components_by_target(frame, min_animals=2, min_responder_coverage=0.0)
    cells = cell_summaries(frame)
    actual = profile_components_from_cells(cells, min_animals=2, min_responder_coverage=0.0)
    pd.testing.assert_frame_equal(actual, expected)


def test_components_from_cells_fail_closed_on_missing_columns() -> None:
    with pytest.raises(ValueError):
        profile_components_from_cells(pd.DataFrame({"target": ["t0"]}))
