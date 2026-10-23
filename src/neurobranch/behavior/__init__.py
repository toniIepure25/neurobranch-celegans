"""Behavioural sequence utilities: runs, transitions and future outcomes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from neurobranch.data.kato import MACRO_DISPLAY, MACRO_ORDER


@dataclass(frozen=True)
class Run:
    state: str
    start: int
    end: int  # exclusive

    @property
    def duration(self) -> int:
        return self.end - self.start


def runs(labels: Sequence[str]) -> list[Run]:
    if len(labels) == 0:
        return []
    out: list[Run] = []
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[start]:
            out.append(Run(str(labels[start]), start, i))
            start = i
    out.append(Run(str(labels[start]), start, len(labels)))
    return out


def transition_counts(labels: Sequence[str], order: Sequence[str] = MACRO_ORDER) -> np.ndarray:
    idx = {s: i for i, s in enumerate(order)}
    C = np.zeros((len(order), len(order)), dtype=np.int64)
    for a, b in zip(labels[:-1], labels[1:]):
        if a in idx and b in idx:
            C[idx[a], idx[b]] += 1
    return C


def transition_matrix(counts: np.ndarray, *, smoothing: float = 0.0) -> np.ndarray:
    C = counts.astype(np.float64) + smoothing
    row = C.sum(axis=1, keepdims=True)
    return np.divide(C, row, out=np.zeros_like(C), where=row > 0)


def next_state_after_leaving(labels: Sequence[str], t: int, horizon: int) -> tuple[str, int] | None:
    """First state different from ``labels[t]`` within ``(t, t+horizon]``.

    Returns ``(outcome, latency)`` or ``None`` if no transition is observed in
    the horizon.  Censored outcomes are *not* silently mapped to the current
    state; downstream code decides how to handle them.
    """
    current = labels[t]
    for j in range(t + 1, min(len(labels), t + horizon + 1)):
        if labels[j] != current:
            return str(labels[j]), j - t
    return None


def future_distribution(
    labels: Sequence[str],
    indices: Sequence[int],
    horizon: int,
    *,
    order: Sequence[str] = MACRO_ORDER,
    include_censored_as_current: bool = True,
) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for t in indices:
        outcome = next_state_after_leaving(labels, int(t), horizon)
        if outcome is None:
            if include_censored_as_current:
                counts[str(labels[int(t)])] += 1
        else:
            counts[outcome[0]] += 1
    total = sum(counts.values())
    return {s: counts.get(s, 0) / total for s in order} if total else {s: 0.0 for s in order}


def state_at_horizon(labels: Sequence[str], indices: Sequence[int], horizon: int) -> np.ndarray:
    return np.array([labels[min(len(labels) - 1, int(t) + horizon)] for t in indices])


def reversal_to_turn_events(labels: Sequence[str], horizon: int = 20) -> list[dict]:
    """All frames in a reverse run whose future *leaving* state is a turn."""
    events = []
    for t in range(len(labels) - 1):
        if labels[t] != "REV":
            continue
        outcome = next_state_after_leaving(labels, t, horizon)
        if outcome and outcome[0] == "TURN":
            events.append({"frame": t, "latency": outcome[1], "from": "REV", "to": "TURN"})
    return events


def describe_distribution(p: dict[str, float]) -> str:
    return ", ".join(f"{MACRO_DISPLAY.get(k, k)}={v:.2f}" for k, v in p.items() if v > 0)


def conditional_entropy(p: dict[str, float]) -> float:
    vals = np.array([v for v in p.values() if v > 0], dtype=np.float64)
    return float(-(vals * np.log2(vals)).sum())


def behaviour_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def majority_baseline(y_train: np.ndarray, y_test: np.ndarray) -> float:
    vals, counts = np.unique(y_train, return_counts=True)
    return float(np.mean(y_test == vals[np.argmax(counts)]))
