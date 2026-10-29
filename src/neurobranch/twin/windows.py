"""Event-aligned post-stimulus windows for the Phase 8 response twin.

The Phase 6/7 tables store a scalar response per target-responder pair.  Phase 8
needs the actual pre/post trajectory, but it must not invent response values for
traces that are absent from the source export.  This module therefore aligns the
already-audited Phase 3 trial table with the raw Randi exports, fills a dense
event x responder tensor, and records a mask for every observed cell.

No transformation is fitted here.  Normalisation, imputation and scaling belong
to the training split and are fitted in :mod:`neurobranch.twin.train`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT, resolve_path, sha256_file
from neurobranch.data.randi import RandiRecording, load_recording

DEFAULT_TRIALS = "reports/phase3/randi_trial_table_phase3.csv.gz"
DEFAULT_RAW_DIR = "data/raw/randi2023/exported_data"


@dataclass(frozen=True)
class WindowConfig:
    """Window geometry.  ``post_start=1`` matches the Phase 3 extraction."""

    pre_frames: int = 10
    post_frames: int = 20
    post_start: int = 1
    min_responders: int = 2
    response_column: str = "response_mean"

    def validate(self) -> None:
        if self.pre_frames < 1 or self.post_frames < 1:
            raise ValueError("pre_frames and post_frames must be positive")
        if self.post_start < 0:
            raise ValueError("post_start must be non-negative")
        if self.min_responders < 1:
            raise ValueError("min_responders must be positive")


@dataclass
class EventWindowCache:
    """Dense, masked event-level windows plus event metadata.

    Attributes are stored in arrays so the cache can be persisted without
    pickles: ``pre`` and ``post`` are ``[n_events, n_responders, frames]``,
    ``mask`` is ``[n_events, n_responders]`` and ``scalar`` is the Phase 3
    ``response_mean`` target for the same cell.
    """

    pre: np.ndarray
    post: np.ndarray
    mask: np.ndarray
    scalar: np.ndarray
    history: np.ndarray
    history_names: tuple[str, ...]
    event_ids: np.ndarray
    dataset_ids: np.ndarray
    target_names: tuple[str, ...]
    responder_names: tuple[str, ...]
    animal_names: tuple[str, ...]
    target_idx: np.ndarray
    event_target: np.ndarray
    event_index: np.ndarray
    stim_frame: np.ndarray
    config: WindowConfig
    source_trials: str
    source_raw_dir: str
    trials_sha256: str

    @property
    def n_events(self) -> int:
        return int(len(self.event_ids))

    @property
    def n_responders(self) -> int:
        return int(len(self.responder_names))

    def event_rows(self, event_id: str) -> np.ndarray:
        """Return responder rows for one event, in responder-name order."""
        hits = np.flatnonzero(self.event_ids == str(event_id))
        if hits.size != 1:
            raise KeyError(f"event {event_id!r} not uniquely present")
        return np.flatnonzero(self.mask[hits[0]])

    def manifest(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "n_events": self.n_events,
            "n_responders": self.n_responders,
            "n_animals": len(self.animal_names),
            "n_targets": len(self.target_names),
            "n_masked_cells": int(self.mask.sum()),
            "pre_frames": int(self.pre.shape[-1]),
            "post_frames": int(self.post.shape[-1]),
            "history_names": list(self.history_names),
            "config": asdict(self.config),
            "source_trials": self.source_trials,
            "source_raw_dir": self.source_raw_dir,
            "trials_sha256": self.trials_sha256,
            "array_shapes": {
                "pre": list(self.pre.shape),
                "post": list(self.post.shape),
                "mask": list(self.mask.shape),
                "scalar": list(self.scalar.shape),
            },
        }
        payload["manifest_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return payload

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            pre=self.pre.astype(np.float32),
            post=self.post.astype(np.float32),
            mask=self.mask.astype(bool),
            scalar=self.scalar.astype(np.float32),
            history=self.history.astype(np.float32),
            history_names=np.asarray(self.history_names, dtype=object),
            event_ids=self.event_ids.astype(object),
            dataset_ids=self.dataset_ids.astype(object),
            target_names=np.asarray(self.target_names, dtype=object),
            responder_names=np.asarray(self.responder_names, dtype=object),
            animal_names=np.asarray(self.animal_names, dtype=object),
            target_idx=self.target_idx.astype(np.int64),
            event_target=self.event_target.astype(np.int64),
            event_index=self.event_index.astype(np.int64),
            stim_frame=self.stim_frame.astype(np.int64),
            config_json=np.asarray(json.dumps(asdict(self.config), sort_keys=True)),
            source_trials=np.asarray(self.source_trials),
            source_raw_dir=np.asarray(self.source_raw_dir),
            trials_sha256=np.asarray(self.trials_sha256),
        )
        return p


def _event_history(frame: pd.DataFrame, *, dt: float) -> tuple[np.ndarray, tuple[str, ...]]:
    interval = pd.to_numeric(frame.get("prev_stim_interval"), errors="coerce").to_numpy(dtype=float)
    same = pd.to_numeric(frame.get("prev_same_target"), errors="coerce").to_numpy(dtype=float)
    order = pd.to_numeric(frame.get("event_order"), errors="coerce").to_numpy(dtype=float)
    pre_mean = pd.to_numeric(frame.get("pre_activity"), errors="coerce").to_numpy(dtype=float)
    pre_sd = pd.to_numeric(frame.get("pre_sd"), errors="coerce").to_numpy(dtype=float)
    interval_s = np.where(np.isfinite(interval), interval, 0.0)
    n = len(frame)
    pre_mean_scalar = float(np.nanmedian(pre_mean)) if np.isfinite(pre_mean).any() else 0.0
    pre_sd_scalar = float(np.nanmedian(pre_sd)) if np.isfinite(pre_sd).any() else 0.0
    names = (
        "log_prev_interval",
        "prev_same_target",
        "event_order_scaled",
        "pre_activity_median",
        "pre_sd_median",
        "inverse_interval",
        "interval_missing",
        "event_order_missing",
    )
    out = np.column_stack(
        [
            np.log1p(np.maximum(interval_s, 0.0)),
            np.where(np.isfinite(same), same, 0.0),
            order / 100.0,
            np.full(n, pre_mean_scalar),
            np.full(n, pre_sd_scalar),
            np.full(n, 1.0 / (float(np.median(interval_s)) + dt)) if dt > 0 else np.zeros(n),
            (~np.isfinite(interval)).astype(float),
            (~np.isfinite(order)).astype(float),
        ]
    ).astype(np.float32)
    # Event metadata is repeated for every responder row; keep one row.
    return out[0], names


def _load_recording_cached(
    raw_dir: Path, dataset_id: int, cache: dict[int, RandiRecording]
) -> RandiRecording:
    if dataset_id not in cache:
        cache[dataset_id] = load_recording(raw_dir, dataset_id)
    return cache[dataset_id]


def build_event_windows(
    trials_path: str | Path = DEFAULT_TRIALS,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    *,
    config: WindowConfig | None = None,
    output_path: str | Path | None = None,
    root: str | Path | None = None,
) -> EventWindowCache:
    """Build the dense event-window cache from the audited Phase 3 table.

    Only cells present in the trial table are filled.  If a source trace is not
    finite over the entire pre/post window, that cell is left masked rather than
    imputed.  Events with fewer than ``min_responders`` finite cells are dropped
    and counted in the returned manifest.
    """
    cfg = config or WindowConfig()
    cfg.validate()
    base = Path(root) if root is not None else PROJECT_ROOT
    trials_file = resolve_path(trials_path, base)
    raw_path = resolve_path(raw_dir, base)
    if not trials_file.exists():
        raise FileNotFoundError(f"trial table not found: {trials_file}")
    if not raw_path.exists():
        raise FileNotFoundError(f"raw Randi directory not found: {raw_path}")
    table = pd.read_csv(trials_file)
    required = {
        "dataset_id",
        "event_id",
        "event_index",
        "target",
        "neuron",
        "stim_frame",
        "target_trace_index",
        "neuron_trace_index",
        cfg.response_column,
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"trial table missing {sorted(missing)}")
    table = table.dropna(subset=["target", "neuron", cfg.response_column]).reset_index(drop=True)
    if table.empty:
        raise ValueError("no finite trial rows")

    target_names = tuple(sorted(table["target"].astype(str).unique()))
    responder_names = tuple(sorted(table["neuron"].astype(str).unique()))
    animal_names = tuple(sorted(table["dataset_id"].astype(str).unique()))
    target_map = {n: i for i, n in enumerate(target_names)}
    responder_map = {n: i for i, n in enumerate(responder_names)}
    animal_set = set(animal_names)
    del animal_set

    # One row per event; the trial table repeats event metadata for responders.
    events = (
        table.sort_values(["dataset_id", "event_index"])
        .drop_duplicates("event_id")
        .reset_index(drop=True)
    )
    event_ids: list[str] = []
    dataset_ids: list[str] = []
    event_index: list[int] = []
    stim_frames: list[int] = []
    event_target: list[int] = []
    histories: list[np.ndarray] = []
    history_names: tuple[str, ...] = ()
    pre_rows: list[np.ndarray] = []
    post_rows: list[np.ndarray] = []
    scalar_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    recording_cache: dict[int, RandiRecording] = {}
    dropped_events = 0
    dropped_cells = 0

    grouped = {str(k): g for k, g in table.groupby("event_id", sort=False)}
    for row in events.itertuples(index=False):
        event_id = str(row.event_id)
        rows = grouped.get(event_id)
        if rows is None or rows.empty:
            dropped_events += 1
            continue
        ds = int(row.dataset_id)
        i0 = int(row.stim_frame)
        rec = _load_recording_cached(raw_path, ds, recording_cache)
        if i0 < cfg.pre_frames or i0 + cfg.post_start + cfg.post_frames > rec.n_frames:
            dropped_events += 1
            continue
        pre = np.full((len(responder_names), cfg.pre_frames), np.nan, dtype=np.float32)
        post = np.full((len(responder_names), cfg.post_frames), np.nan, dtype=np.float32)
        scalar = np.full(len(responder_names), np.nan, dtype=np.float32)
        mask = np.zeros(len(responder_names), dtype=bool)
        for cell in rows.itertuples(index=False):
            j = int(cell.neuron_trace_index)
            if j < 0 or j >= rec.n_traces:
                dropped_cells += 1
                continue
            pre_j = rec.traces[i0 - cfg.pre_frames : i0, j]
            post_j = rec.traces[i0 + cfg.post_start : i0 + cfg.post_start + cfg.post_frames, j]
            if pre_j.shape[0] != cfg.pre_frames or post_j.shape[0] != cfg.post_frames:
                dropped_cells += 1
                continue
            if not (np.isfinite(pre_j).all() and np.isfinite(post_j).all()):
                dropped_cells += 1
                continue
            baseline = float(np.median(pre_j))
            delta = post_j - baseline
            name = str(cell.neuron)
            if name not in responder_map:
                dropped_cells += 1
                continue
            ridx = responder_map[name]
            pre[ridx] = pre_j.astype(np.float32)
            post[ridx] = delta.astype(np.float32)
            scalar[ridx] = float(np.mean(delta))
            mask[ridx] = True
        if int(mask.sum()) < cfg.min_responders:
            dropped_events += 1
            continue
        hist, history_names = _event_history(rows, dt=rec.dt)
        event_ids.append(event_id)
        dataset_ids.append(str(ds))
        event_index.append(int(row.event_index))
        stim_frames.append(i0)
        event_target.append(target_map[str(row.target)])
        histories.append(hist)
        pre_rows.append(pre)
        post_rows.append(post)
        scalar_rows.append(scalar)
        mask_rows.append(mask)

    if not event_ids:
        raise ValueError("no usable events after windowing")
    cache = EventWindowCache(
        pre=np.stack(pre_rows, axis=0),
        post=np.stack(post_rows, axis=0),
        mask=np.stack(mask_rows, axis=0),
        scalar=np.stack(scalar_rows, axis=0),
        history=np.stack(histories, axis=0),
        history_names=history_names,
        event_ids=np.asarray(event_ids, dtype=object),
        dataset_ids=np.asarray(dataset_ids, dtype=object),
        target_names=target_names,
        responder_names=responder_names,
        animal_names=animal_names,
        target_idx=np.asarray(event_target, dtype=np.int64),
        event_target=np.asarray(event_target, dtype=np.int64),
        event_index=np.asarray(event_index, dtype=np.int64),
        stim_frame=np.asarray(stim_frames, dtype=np.int64),
        config=cfg,
        source_trials=str(trials_file),
        source_raw_dir=str(raw_path),
        trials_sha256=sha256_file(trials_file),
    )
    if output_path is not None:
        cache.save(output_path)
        manifest_path = Path(output_path).with_suffix(".manifest.json")
        manifest = cache.manifest()
        manifest["dropped_events"] = dropped_events
        manifest["dropped_cells"] = dropped_cells
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return cache


def load_window_cache(path: str | Path) -> EventWindowCache:
    """Load a cache written by :meth:`EventWindowCache.save`."""
    p = Path(path)
    data = np.load(p, allow_pickle=True)
    cfg = WindowConfig(**json.loads(str(data["config_json"].item())))
    return EventWindowCache(
        pre=np.asarray(data["pre"], dtype=np.float32),
        post=np.asarray(data["post"], dtype=np.float32),
        mask=np.asarray(data["mask"], dtype=bool),
        scalar=np.asarray(data["scalar"], dtype=np.float32),
        history=np.asarray(data["history"], dtype=np.float32),
        history_names=tuple(str(x) for x in data["history_names"].tolist()),
        event_ids=np.asarray(data["event_ids"], dtype=object),
        dataset_ids=np.asarray(data["dataset_ids"], dtype=object),
        target_names=tuple(str(x) for x in data["target_names"].tolist()),
        responder_names=tuple(str(x) for x in data["responder_names"].tolist()),
        animal_names=tuple(str(x) for x in data["animal_names"].tolist()),
        target_idx=np.asarray(data["target_idx"], dtype=np.int64),
        event_target=np.asarray(data["event_target"], dtype=np.int64),
        event_index=np.asarray(data["event_index"], dtype=np.int64),
        stim_frame=np.asarray(data["stim_frame"], dtype=np.int64),
        config=cfg,
        source_trials=str(data["source_trials"].item()),
        source_raw_dir=str(data["source_raw_dir"].item()),
        trials_sha256=str(data["trials_sha256"].item()),
    )
