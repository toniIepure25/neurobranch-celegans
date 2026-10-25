"""Pre-stimulus population-state features for Phase 7.

The state cache stores only the median pre-stimulus fluorescence per recorded
neuron and event.  Post-stimulus frames are never read.  The PCA projector is
fitted inside the outer-training split, so the outer test cannot influence the
state representation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from neurobranch.causal.features import FeatureBundle
from neurobranch.data.randi import load_recording


@dataclass(frozen=True)
class StateCache:
    event_ids: tuple[str, ...]
    node_names: tuple[str, ...]
    values: np.ndarray
    mask: np.ndarray

    @property
    def n_events(self) -> int:
        return int(len(self.event_ids))


@dataclass
class StateProjector:
    scaler: StandardScaler
    pca: PCA
    node_names: tuple[str, ...]
    train_event_ids: frozenset[str]
    fill_values: np.ndarray

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(f"state_pc_{i:02d}" for i in range(self.pca.n_components_))

    def transform(self, cache: StateCache, event_ids: Iterable[str]) -> pd.DataFrame:
        ids = [str(e) for e in event_ids]
        index = {e: i for i, e in enumerate(cache.event_ids)}
        rows = np.full((len(ids), len(cache.node_names)), np.nan, dtype=float)
        for j, event in enumerate(ids):
            if event in index:
                row = cache.values[index[event]]
                mask = cache.mask[index[event]]
                rows[j, mask] = row[mask]
        # Training-only imputation and projection.
        means = self.fill_values
        rows = np.where(np.isfinite(rows), rows, means)
        projected = self.pca.transform(self.scaler.transform(rows))
        return pd.DataFrame(projected, columns=list(self.feature_names), index=ids)


def build_state_cache(
    trials: pd.DataFrame,
    *,
    data_dir: str | Path,
    pre_frames: int = 8,
) -> StateCache:
    """Extract median pre-stimulus activity for every event in ``trials``."""
    work = trials[["event_id", "dataset_id", "event_index"]].drop_duplicates("event_id")
    node_names = tuple(sorted(trials["neuron"].astype(str).unique()))
    node_index = {name: i for i, name in enumerate(node_names)}
    values = np.full((len(work), len(node_names)), np.nan, dtype=np.float32)
    mask = np.zeros_like(values, dtype=bool)
    event_ids = tuple(work["event_id"].astype(str))
    for ds, group in work.groupby("dataset_id"):
        rec = load_recording(data_dir, int(ds))
        label_to_node = {}
        for trace_i, label in enumerate(rec.labels):
            if label is not None and label in node_index and label not in label_to_node:
                label_to_node[label] = trace_i
        for row_pos, event_index in zip(
            np.flatnonzero(work["dataset_id"].to_numpy() == ds),
            group["event_index"].astype(int),
        ):
            i0 = int(rec.stim_frames[event_index])
            if i0 < pre_frames:
                continue
            pre = rec.traces[i0 - pre_frames : i0]
            for label, trace_i in label_to_node.items():
                column = node_index[label]
                column_values = pre[:, trace_i]
                if np.isfinite(column_values).all():
                    values[row_pos, column] = float(np.median(column_values))
                    mask[row_pos, column] = True
    return StateCache(event_ids=event_ids, node_names=node_names, values=values, mask=mask)


def fit_state_projector(
    cache: StateCache,
    *,
    train_event_ids: Iterable[str],
    n_components: int = 16,
    seed: int = 17,
) -> StateProjector:
    train = frozenset(str(e) for e in train_event_ids)
    positions = [i for i, e in enumerate(cache.event_ids) if e in train]
    if len(positions) < max(10, n_components + 2):
        raise ValueError("too few training events for state PCA")
    x = cache.values[positions].astype(float)
    means = np.nanmean(x, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    x = np.where(np.isfinite(x), x, means)
    scaler = StandardScaler().fit(x)
    n_components = min(n_components, x.shape[1] - 1, x.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=seed).fit(scaler.transform(x))
    return StateProjector(
        scaler=scaler,
        pca=pca,
        node_names=cache.node_names,
        train_event_ids=train,
        fill_values=means,
    )


def attach_state_features(
    bundle: FeatureBundle,
    projector: StateProjector,
    cache: StateCache,
) -> FeatureBundle:
    """Append train-projected state PCs to the CRO context."""
    state = projector.transform(cache, bundle.event_ids).to_numpy(dtype=np.float32)
    context = np.concatenate([bundle.context_raw.astype(np.float32), state], axis=1)
    return replace(bundle, context_raw=context, state_features=state)


def save_state_cache(cache: StateCache, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        event_ids=np.asarray(cache.event_ids, dtype=str),
        node_names=np.asarray(cache.node_names, dtype=str),
        values=cache.values,
        mask=cache.mask,
    )


def load_state_cache(path: str | Path) -> StateCache:
    data = np.load(Path(path), allow_pickle=False)
    return StateCache(
        event_ids=tuple(str(x) for x in data["event_ids"]),
        node_names=tuple(str(x) for x in data["node_names"]),
        values=np.asarray(data["values"], dtype=np.float32),
        mask=np.asarray(data["mask"], dtype=bool),
    )
