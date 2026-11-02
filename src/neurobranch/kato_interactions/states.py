"""Training-only neural-state geometry and transition diagnostics."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def fit_states(projected_train: np.ndarray, *, n_states: int = 6, seed: int = 23) -> KMeans:
    return KMeans(n_clusters=int(n_states), random_state=int(seed), n_init=10).fit(
        np.asarray(projected_train, dtype=np.float64)
    )


def assign_states(paths: np.ndarray, centers: np.ndarray) -> np.ndarray:
    arr = np.asarray(paths, dtype=np.float64)
    flat = arr.reshape(-1, arr.shape[-1])
    dist = ((flat[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    return dist.argmin(axis=-1).reshape(arr.shape[:-1])


def transition_counts(labels: np.ndarray, n_states: int) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr[None, :]
    counts = np.zeros((int(n_states), int(n_states)), dtype=np.float64)
    for row in arr:
        if len(row) < 2:
            continue
        np.add.at(counts, (row[:-1], row[1:]), 1.0)
    return counts


def row_normalize(counts: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    denom = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, denom, out=np.zeros_like(counts), where=denom > eps)


def occupancy(labels: np.ndarray, n_states: int) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int64).ravel()
    return np.bincount(arr, minlength=int(n_states)).astype(np.float64) / max(len(arr), 1)


def dwell_times(labels: np.ndarray, n_states: int) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr[None, :]
    values: list[list[int]] = [[] for _ in range(int(n_states))]
    for row in arr:
        if len(row) == 0:
            continue
        start = 0
        for i in range(1, len(row) + 1):
            if i == len(row) or row[i] != row[start]:
                values[int(row[start])].append(i - start)
                start = i
    return np.asarray([np.mean(v) if v else np.nan for v in values], dtype=np.float64)


def weighted_transition_tv(model: np.ndarray, observed: np.ndarray, counts: np.ndarray) -> float:
    weights = np.asarray(counts, dtype=np.float64).sum(axis=1)
    if weights.sum() <= 0:
        return float("nan")
    rows = 0.5 * np.abs(np.asarray(model) - np.asarray(observed)).sum(axis=1)
    return float(np.sum(rows * weights) / weights.sum())


def simulate_markov(
    transition: np.ndarray,
    *,
    n_members: int,
    horizon: int,
    initial: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n_states = transition.shape[0]
    out = np.empty((int(n_members), int(horizon)), dtype=np.int64)
    out[:, 0] = np.asarray(initial, dtype=np.int64)[: int(n_members)]
    cumulative = np.cumsum(transition, axis=1)
    cumulative[:, -1] = 1.0
    for step in range(1, int(horizon)):
        u = rng.random(int(n_members))
        out[:, step] = (u[:, None] > cumulative[out[:, step - 1]]).sum(axis=1)
        out[:, step] = np.clip(out[:, step], 0, n_states - 1)
    return out


def state_metrics(
    model_labels: np.ndarray,
    observed_labels: np.ndarray,
    *,
    train_labels: np.ndarray,
    n_states: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Compare a model state path ensemble with one observed continuation."""
    model_labels = np.asarray(model_labels, dtype=np.int64)
    observed_labels = np.asarray(observed_labels, dtype=np.int64)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    model_counts = transition_counts(model_labels, n_states)
    obs_counts = transition_counts(observed_labels, n_states)
    train_counts = transition_counts(train_labels, n_states)
    model_T = row_normalize(model_counts)
    obs_T = row_normalize(obs_counts)
    train_T = row_normalize(train_counts)
    model_occ = occupancy(model_labels, n_states)
    obs_occ = occupancy(observed_labels, n_states)
    train_occ = occupancy(train_labels, n_states)
    model_dwell = dwell_times(model_labels, n_states)
    obs_dwell = dwell_times(observed_labels, n_states)
    m = model_labels.reshape(-1, model_labels.shape[-1])
    o = observed_labels.reshape(-1, observed_labels.shape[-1])
    metrics = {
        "occupancy_tv_observed": float(0.5 * np.abs(model_occ - obs_occ).sum()),
        "occupancy_tv_train": float(0.5 * np.abs(model_occ - train_occ).sum()),
        "transition_tv_observed": weighted_transition_tv(model_T, obs_T, obs_counts),
        "transition_tv_train": weighted_transition_tv(model_T, train_T, train_counts),
        "dwell_l1_observed": float(np.nanmean(np.abs(model_dwell - obs_dwell))),
        "recurrence_model": float((m[:, 1:] == m[:, :-1]).mean()) if m.shape[1] > 1 else float("nan"),
        "recurrence_observed": float((o[:, 1:] == o[:, :-1]).mean()) if o.shape[1] > 1 else float("nan"),
        "occupancy_entropy_model": float(-(model_occ * np.log(np.clip(model_occ, 1e-12, None))).sum()),
        "occupancy_entropy_observed": float(-(obs_occ * np.log(np.clip(obs_occ, 1e-12, None))).sum()),
    }
    return metrics, {
        "model_transition": model_T,
        "observed_transition": obs_T,
        "train_transition": train_T,
        "model_occupancy": model_occ,
        "observed_occupancy": obs_occ,
        "train_occupancy": train_occ,
        "model_dwell": model_dwell,
        "observed_dwell": obs_dwell,
    }
