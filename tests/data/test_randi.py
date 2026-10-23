from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neurobranch.data.randi import (
    canonical_neuron_name,
    extract_trials,
    load_recording,
)


def _write_dataset(
    root: Path,
    *,
    trailing_blank_labels: int = 0,
    stim_frames: list[int] | None = None,
    stim_targets: list[int] | None = None,
) -> Path:
    ds = root / "0"
    t = np.arange(30) * 0.5
    traces = np.zeros((30, 3))
    traces[11:, 1] = 5.0
    traces[11:, 2] = -2.0
    np.savetxt(f"{ds}_gcamp.txt", traces)
    np.savetxt(f"{ds}_t.txt", t)
    labels = ["A", "B", "C"] + [""] * trailing_blank_labels
    Path(f"{ds}_labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    frames = stim_frames or [10]
    targets = stim_targets if stim_targets is not None else [0] * len(frames)
    assert len(frames) == len(targets)
    Path(f"{ds}_stim_neurons.txt").write_text("\n".join(str(x) for x in targets) + "\n", encoding="utf-8")
    Path(f"{ds}_stim_volume_i.txt").write_text("\n".join(str(x) for x in frames) + "\n", encoding="utf-8")
    Path(f"{ds}_ds_name.txt").write_text("synthetic/0", encoding="utf-8")
    return root


def test_canonical_names_do_not_invent_identity() -> None:
    assert canonical_neuron_name(" DB1 ") == "DB01"
    assert canonical_neuron_name("VB2") == "VB02"
    assert canonical_neuron_name("AS1") == "AS01"
    assert canonical_neuron_name("RIGL") == "RIGL"
    assert canonical_neuron_name("RIG") is None
    assert canonical_neuron_name("12") is None
    assert canonical_neuron_name("") is None
    assert canonical_neuron_name("smthng else") is None


def test_load_recording_trims_only_trailing_blank_labels(tmp_path: Path) -> None:
    _write_dataset(tmp_path, trailing_blank_labels=4)
    rec = load_recording(tmp_path, 0)
    assert rec.n_traces == 3
    assert rec.labels == ("A", "B", "C")
    assert rec.n_frames == 30
    assert rec.dt == 0.5


def test_extract_trials_excludes_stimulated_trace_and_keeps_sign(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    rec = load_recording(tmp_path, 0)
    table = extract_trials(rec)
    assert set(table["neuron"]) == {"B", "C"}
    assert "A" not in set(table["neuron"])
    b = table[table.neuron == "B"].iloc[0]
    c = table[table.neuron == "C"].iloc[0]
    assert b.response == 5.0
    assert c.response == -2.0
    assert b.response_sign == 1
    assert c.response_sign == -1
    assert b.event_id == "0:0"
    for col in ("response_mean", "response_integral", "response_peak_time", "pre_sd"):
        assert col in table.columns
    assert np.isfinite(b.response_mean)
    assert np.isfinite(b.response_integral)
    assert b.prev_stim_interval != b.prev_stim_interval  # NaN for the first event


def test_previous_stimulus_interval_uses_immediately_preceding_event(tmp_path: Path) -> None:
    _write_dataset(tmp_path, stim_frames=[10, 11, 16], stim_targets=[0, 0, 0])
    rec = load_recording(tmp_path, 0)
    table = extract_trials(rec)
    b = table[(table.neuron == "B") & (table.event_order == 2)].iloc[0]
    # Immediately preceding event is frame 11, not the closest pair (10->11).
    assert np.isclose(b.prev_stim_interval, (16 - 11) * 0.5)
    assert bool(b.prev_same_target) is True


def test_extract_trials_does_not_impute_nan(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    arr = np.loadtxt(tmp_path / "0_gcamp.txt")
    arr[:, 2] = np.nan
    np.savetxt(tmp_path / "0_gcamp.txt", arr)
    rec = load_recording(tmp_path, 0)
    table = extract_trials(rec)
    assert set(table["neuron"]) == {"B"}


def test_coverage_summary_reports_coverage(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    rec = load_recording(tmp_path, 0)
    table = extract_trials(rec)
    from neurobranch.data.randi import coverage_summary

    s = coverage_summary(table, tmp_path)
    assert s["n_source_datasets"] == 1
    assert s["n_events"] == 1
    assert s["n_animals"] == 1
    assert s["n_stimulated_neurons"] == 1
    assert isinstance(pd.DataFrame(table), pd.DataFrame)

