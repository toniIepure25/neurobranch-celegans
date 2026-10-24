from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neurobranch.behavior.turn_direction import build_designs, segment_bounds
from neurobranch.data.kramer import DEFAULT_RAW, load_all

DATA = Path(DEFAULT_RAW)


def test_segment_windows_end_before_offset():
    for length in range(12, 30):
        start, end = 10, 10 + length
        for window in ("early", "mid", "late"):
            bounds = segment_bounds(start, end, window)
            assert bounds is not None
            s, e = bounds
            assert e - s == 4
            assert s >= start
            assert e <= end - 2  # at least two outcome-adjacent frames are held out


def test_segment_rejects_short_events_and_unknown_window():
    assert segment_bounds(0, 11, "mid") is None
    with pytest.raises(ValueError):
        segment_bounds(0, 12, "not-a-window")


@pytest.mark.skipif(not (DATA / "wt.wbData.mat").exists(), reason="Kramer archive not downloaded")
def test_orientation_and_event_conventions():
    recs, audit = load_all(cohorts=("wt_octanol",), min_events=5)
    assert recs
    assert len(audit) == len(recs)
    for rec in recs[:5]:
        assert np.allclose(np.abs(rec.dv), np.abs(rec.head_angle), atol=1e-6)
        assert rec.quality["orientation_sign"] in {-1, 1}
        table = rec.event_table(min_length=12)
        assert (table["end"] > table["start"]).all()
        assert table["duration_frames"].tolist() == (table["end"] - table["start"]).tolist()
        both = table[table["valid_post"]]
        assert set(both["direction"]).issubset({"dorsal", "ventral"})
        # Sign convention: direction is the sign of the post-offset mean dv.
        assert np.all((both["post_dv"] > 0) == (both["direction"] == "dorsal"))
        assert np.all((both["post_dv"] < 0) == (both["direction"] == "ventral"))


@pytest.mark.skipif(not (DATA / "wt.wbData.mat").exists(), reason="Kramer archive not downloaded")
def test_design_segments_are_prospective():
    recs, _ = load_all(cohorts=("tdc1_octanol",), min_events=5)
    designs, labels = build_designs(recs[:2], window="mid")
    assert designs
    assert labels
    for d in designs:
        assert len(d.y) == len(d.events)
        assert np.all(d.segment_ends <= d.events["end"].to_numpy() - 2)
        assert np.all(d.segment_starts >= d.events["start"].to_numpy())
        assert d.population.shape[1] == 32
        assert d.behavior.shape[1] == 23
        assert d.saav.shape[1] == 1
