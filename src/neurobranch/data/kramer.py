"""Adapter for the Kramer et al. (2026) directed-turning dataset.

The Dryad archive contains one MATLAB ``wbData`` structure per cohort and one
HDF5 file per recording.  The MATLAB structures carry the neuron identities,
odor annotations, behaviour and reversal events; the HDF5 files carry the
**original** calcium traces.  The MATLAB ``gcamp_trace`` field is a
whole-recording z-score, so it is never used for the primary analyses: raw
``trace_array_original`` values are read from the matched HDF5 file and
normalised inside each training/evaluation fold (or causally within an animal).

Reversal events use **0-based start and exclusive end**, verified against
``reversal_vec``.  A recording may begin in the middle of a reversal; that
initial partial segment is not present in the event table.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio

from neurobranch.config import sha256_file

DATASET_DOI = "doi:10.5061/dryad.8sf7m0cz2"
DATASET_LICENSE = "CC0-1.0"
DEFAULT_RAW = "data/raw/kramer2026/files"
DT_FALLBACK_S = 0.6000188
POST_TURN_FRAMES = 2
COHORT_FILES = {
    "wt_octanol": "wt.wbData.mat",
    "wt_diacetyl": "diacetyl.wbData.mat",
    "tdc1_octanol": "tdc1.wbData.mat",
}


@dataclass
class KramerRecording:
    animal_id: str
    cohort: str
    source_mat: str
    source_h5: str
    source_sha256: str
    traces_raw: np.ndarray  # (time, neuron), original GCaMP
    traces_published_z: np.ndarray  # (time, neuron), whole-recording z-score
    time_seconds: np.ndarray
    velocity: np.ndarray
    reversal_vec: np.ndarray
    reversal_onsets: np.ndarray  # 0-based inclusive start
    reversal_offsets: np.ndarray  # 0-based exclusive end
    head_angle: np.ndarray
    dv: np.ndarray
    worm_angle: np.ndarray
    angular_velocity: np.ndarray
    pumping: np.ndarray
    distance_from_center: np.ndarray
    odor_idx: np.ndarray
    labels: tuple[str | None, ...] = ()
    label_ambiguous: tuple[bool, ...] = ()
    quality: dict[str, Any] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.traces_raw.shape[0])

    @property
    def n_neurons(self) -> int:
        return int(self.traces_raw.shape[1])

    @property
    def dt(self) -> float:
        if len(self.time_seconds) > 1:
            return float(np.median(np.diff(self.time_seconds)))
        return DT_FALLBACK_S

    @property
    def duration_s(self) -> float:
        return float(self.time_seconds[-1] - self.time_seconds[0])

    @property
    def identified_indices(self) -> dict[str, int]:
        return {
            name: i
            for i, (name, amb) in enumerate(zip(self.labels, self.label_ambiguous))
            if name is not None and not amb
        }

    @property
    def n_labelled(self) -> int:
        return int(sum(name is not None for name in self.labels))

    def event_table(self, *, min_length: int = 0) -> pd.DataFrame:
        """Return one row per reversal with a usable post-turn window.

        The direction is taken from the mean ``dv`` in the first two frames
        after the reversal end.  Positive ``dv`` is called dorsal and negative
        ``dv`` ventral.  This mapping is the archive's D/V sign convention; a
        marker-neuron sanity check is recorded in the Phase 4 audit.  The raw
        signed value is retained so the mapping can be audited or reversed.
        """
        rows: list[dict[str, Any]] = []
        dv = np.asarray(self.dv, dtype=float).ravel()
        for event_index, (start, end) in enumerate(zip(self.reversal_onsets, self.reversal_offsets)):
            start = int(start)
            end = int(end)
            length = end - start
            if length < min_length:
                continue
            valid_post = end + POST_TURN_FRAMES <= self.n_frames
            post_dv = float(np.mean(dv[end : end + POST_TURN_FRAMES])) if valid_post else float("nan")
            direction = ""
            if valid_post and np.isfinite(post_dv):
                direction = "dorsal" if post_dv > 0 else "ventral"
            rows.append(
                {
                    "animal_id": self.animal_id,
                    "cohort": self.cohort,
                    "event_index": event_index,
                    "start": start,
                    "end": end,
                    "duration_frames": length,
                    "duration_s": length * self.dt,
                    "post_dv": post_dv,
                    "direction": direction,
                    "near_zero": bool(valid_post and abs(post_dv) < 0.05),
                    "odor_present": bool(np.any(self.odor_idx[start:end] > 0)),
                    "valid_post": bool(valid_post),
                }
            )
        return pd.DataFrame(rows)

    def as_phase3_recording(self):
        """Compatibility wrapper for the Phase 3 prospective benchmark.

        Only fields used by the Phase 3 feature builder are populated.  The
        archive does not contain the copper-centre distance, so that feature is
        constant zero; this is recorded as a deviation rather than imputed.
        """
        from neurobranch.data.baskoylu import FreelyMovingRecording

        return FreelyMovingRecording(
            animal_id=self.animal_id,
            source_file=self.source_h5,
            source_sha256=self.source_sha256,
            traces_raw=self.traces_raw,
            traces_published_z=self.traces_published_z,
            time_seconds=self.time_seconds,
            velocity=self.velocity,
            reversal_vec=self.reversal_vec,
            reversal_onsets=self.reversal_onsets,
            reversal_offsets=self.reversal_offsets,
            head_angle=self.dv,
            angular_velocity=self.angular_velocity,
            distance_from_center=self.distance_from_center,
            labels=self.labels,
            label_confidence=tuple(None for _ in self.labels),
            label_ambiguous=self.label_ambiguous,
            quality=self.quality,
        )


def _signature(velocity: np.ndarray, head_angle: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(np.asarray(velocity, dtype=np.float64).round(8).tobytes())
    h.update(np.asarray(head_angle, dtype=np.float64).round(8).tobytes())
    return h.hexdigest()


def _h5_index(data_dir: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for path in sorted(data_dir.glob("*-data.h5")):
        with h5py.File(path, "r") as f:
            velocity = np.asarray(f["behavior/velocity"][:], dtype=np.float64).ravel()
            head = np.asarray(f["behavior/head_angle"][:], dtype=np.float64).ravel()
        out.setdefault(_signature(velocity, head), []).append(path)
    return out


def _parse_ids(wb_row: Any, n_neurons: int) -> tuple[tuple[str | None, ...], tuple[bool, ...]]:
    names: list[str | None] = [None] * n_neurons
    for i in range(min(n_neurons, wb_row.gcamp_wIDs.shape[0])):
        cell = wb_row.gcamp_wIDs[i, 0]
        value = str(cell.ravel()[0]).strip() if getattr(cell, "size", 0) else ""
        names[i] = value or None
    counts: dict[str, int] = {}
    for name in names:
        if name is not None:
            counts[name] = counts.get(name, 0) + 1
    ambiguous = tuple(name is not None and counts[name] > 1 for name in names)
    return tuple(names), ambiguous


def _validate_events(rec: KramerRecording) -> dict[str, Any]:
    starts = np.asarray(rec.reversal_onsets, dtype=int)
    ends = np.asarray(rec.reversal_offsets, dtype=int)
    if starts.ndim != 1 or ends.ndim != 1 or len(starts) != len(ends):
        raise ValueError(f"{rec.animal_id}: malformed reversal event table")
    if np.any(starts < 0) or np.any(ends > rec.n_frames) or np.any(ends <= starts):
        raise ValueError(f"{rec.animal_id}: reversal event bounds are invalid")
    if np.any(np.diff(starts) < 0):
        raise ValueError(f"{rec.animal_id}: reversal events are not sorted")
    if len(starts) > 1 and np.any(starts[1:] < ends[:-1]):
        raise ValueError(f"{rec.animal_id}: overlapping reversal events")
    occupied = np.zeros(rec.n_frames, dtype=bool)
    for s, e in zip(starts, ends):
        occupied[s:e] = True
        if not np.all(rec.reversal_vec[s:e] == 1):
            raise ValueError(f"{rec.animal_id}: reversal table disagrees with reversal_vec")
    mismatch = np.flatnonzero(occupied.astype(np.int8) != rec.reversal_vec.astype(np.int8))
    if len(mismatch) and not (len(mismatch) == 1 and mismatch[0] == 0):
        raise ValueError(
            f"{rec.animal_id}: reversal_events disagree with reversal_vec in {len(mismatch)} frames"
        )
    return {
        "event_vec_mismatch_frames": int(len(mismatch)),
        "initial_partial_reversal": bool(len(mismatch) == 1),
        "events_validated": int(len(starts)),
    }


def _read_h5(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        out = {
            "raw": np.asarray(f["gcamp/trace_array_original"][:], dtype=np.float64),
            "z": np.asarray(f["gcamp/trace_array"][:], dtype=np.float64),
            "velocity": np.asarray(f["behavior/velocity"][:], dtype=np.float64).ravel(),
            "rv": np.asarray(f["behavior/reversal_vec"][:], dtype=np.int64).ravel(),
            "head": np.asarray(f["behavior/head_angle"][:], dtype=np.float64).ravel(),
            "worm_angle": np.asarray(f["behavior/worm_angle"][:], dtype=np.float64).ravel(),
        }
        for key, name in [
            ("angular", "behavior/angular_velocity"),
            ("pumping", "behavior/pumping"),
            ("x", "behavior/zeroed_x_confocal"),
            ("y", "behavior/zeroed_y_confocal"),
        ]:
            out[key] = np.asarray(f[name][:], dtype=np.float64).ravel() if name in f else np.zeros(len(out["velocity"]))
        for key, name in [("time", "timing/timestamp_confocal")]:
            out[key] = np.asarray(f[name][:], dtype=np.float64).ravel() if name in f else np.arange(len(out["velocity"])) * DT_FALLBACK_S
    return out


def _build_recording(
    cohort: str,
    row_index: int,
    wb_row: Any,
    h5_path: Path,
    *,
    mat_path: Path,
) -> KramerRecording:
    h5 = _read_h5(h5_path)
    velocity = np.asarray(wb_row.velocity, dtype=np.float64).ravel()
    rv = np.asarray(wb_row.reversal_vec, dtype=np.int64).ravel()
    head = np.asarray(wb_row.head_angle, dtype=np.float64).ravel()
    dv = np.asarray(wb_row.dv, dtype=np.float64).ravel()
    worm_angle = np.asarray(wb_row.worm_angle, dtype=np.float64).ravel()
    events = np.asarray(wb_row.reversal_events, dtype=np.int64)
    if events.ndim == 2 and events.shape[0] == 2 and events.shape[1] != 2:
        events = events.T
    if events.ndim != 2 or events.shape[1] != 2:
        raise ValueError(f"{cohort}[{row_index}]: reversal_events has shape {events.shape}")
    n = len(velocity)
    if h5["raw"].shape != (n, wb_row.gcamp_trace.shape[0]):
        raise ValueError(
            f"{cohort}[{row_index}]: HDF5 raw shape {h5['raw'].shape} does not match "
            f"MAT ({n}, {wb_row.gcamp_trace.shape[0]})"
        )
    if not np.allclose(velocity, h5["velocity"], atol=1e-7):
        raise ValueError(f"{cohort}[{row_index}]: MAT/HDF5 velocity mismatch")
    if not np.allclose(head, h5["head"], atol=1e-7):
        raise ValueError(f"{cohort}[{row_index}]: MAT/HDF5 head-angle mismatch")
    # ``dv`` is the orientation-corrected dorsal/ventral head angle; ``head_angle``
    # is the raw camera sign.  The two are equal or opposite for an entire
    # recording, depending on which side the animal lies on.  Using raw
    # ``head_angle`` across animals would mix dorsal and ventral arbitrarily.
    if not np.allclose(np.abs(dv), np.abs(head), atol=1e-7):
        raise ValueError(f"{cohort}[{row_index}]: |dv| does not equal |head_angle|")
    orientation_sign = int(np.sign(np.dot(dv, head)))
    labels, ambiguous = _parse_ids(wb_row, h5["raw"].shape[1])
    time = h5["time"] - h5["time"][0] if len(h5["time"]) == n else np.arange(n) * DT_FALLBACK_S
    odor = np.asarray(wb_row.octanol_idx, dtype=np.int64).ravel() if hasattr(wb_row, "octanol_idx") else np.zeros(n)
    if len(odor) != n:
        odor = np.zeros(n, dtype=np.int64)
    x, y = h5["x"], h5["y"]
    dist = np.sqrt((x - np.nanmean(x)) ** 2 + (y - np.nanmean(y)) ** 2) if len(x) else np.zeros(n)
    rec = KramerRecording(
        animal_id=f"{cohort}_{row_index:02d}",
        cohort=cohort,
        source_mat=str(mat_path),
        source_h5=str(h5_path),
        source_sha256=sha256_file(h5_path),
        traces_raw=h5["raw"],
        traces_published_z=h5["z"],
        time_seconds=time,
        velocity=velocity,
        reversal_vec=rv,
        reversal_onsets=events[:, 0],
        reversal_offsets=events[:, 1],
        head_angle=head,
        dv=dv,
        worm_angle=worm_angle,
        angular_velocity=h5["angular"],
        pumping=h5["pumping"],
        distance_from_center=dist,
        odor_idx=odor,
        labels=labels,
        label_ambiguous=ambiguous,
        quality={},
    )
    qc = _validate_events(rec)
    rec.quality = {
        "n_frames": rec.n_frames,
        "n_neurons": rec.n_neurons,
        "n_labelled_traces": rec.n_labelled,
        "n_identified_traces": len(rec.identified_indices),
        "n_ambiguous_traces": int(sum(ambiguous)),
        "duration_s": rec.duration_s,
        "median_dt_s": rec.dt,
        "n_reversal_events": int(len(rec.reversal_onsets)),
        "reversal_fraction": float(np.mean(rv)),
        "odor_frames": int(np.sum(odor > 0)),
        "orientation_sign": orientation_sign,
        "trace_source": "h5_trace_array_original",
        **qc,
    }
    return rec


def load_all(
    data_dir: str | Path = DEFAULT_RAW,
    *,
    cohorts: tuple[str, ...] = tuple(COHORT_FILES),
    min_events: int = 5,
) -> tuple[list[KramerRecording], list[dict[str, Any]]]:
    """Load matched MAT/HDF5 recordings, returning usable records and an audit.

    Missing HDF5 files are recorded as structural exclusions, not silently
    replaced by the published whole-recording z-score.
    """
    data_dir = Path(data_dir)
    index = _h5_index(data_dir)
    recs: list[KramerRecording] = []
    audit: list[dict[str, Any]] = []
    for cohort in cohorts:
        mat_path = data_dir / COHORT_FILES[cohort]
        if not mat_path.exists():
            raise FileNotFoundError(mat_path)
        payload = sio.loadmat(
            mat_path,
            variable_names=["wbData"],
            squeeze_me=False,
            struct_as_record=False,
            chars_as_strings=True,
        )
        wb = payload["wbData"]
        for row_index in range(wb.shape[1]):
            row = wb[0, row_index]
            velocity = np.asarray(row.velocity, dtype=np.float64).ravel()
            head = np.asarray(row.head_angle, dtype=np.float64).ravel()
            candidates = index.get(_signature(velocity, head), [])
            if not candidates:
                audit.append(
                    {
                        "animal_id": f"{cohort}_{row_index:02d}",
                        "cohort": cohort,
                        "usable": False,
                        "reason": "no matching HDF5 raw-trace file",
                    }
                )
                continue
            h5_path = candidates[0]
            try:
                rec = _build_recording(cohort, row_index, row, h5_path, mat_path=mat_path)
            except Exception as exc:  # noqa: BLE001 - audit must record structural failures
                audit.append(
                    {
                        "animal_id": f"{cohort}_{row_index:02d}",
                        "cohort": cohort,
                        "usable": False,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            usable = len(rec.reversal_onsets) >= min_events
            audit.append({**rec.quality, "animal_id": rec.animal_id, "cohort": cohort, "usable": usable, "reason": "" if usable else "too few reversals"})
            if usable:
                recs.append(rec)
    return recs, audit
