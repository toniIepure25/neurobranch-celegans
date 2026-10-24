"""Evaluation protocols and clustered uncertainty for Phase 5.

The official SAAV decoder uses an ungrouped event-level split.  The same events
contain repeated animal identity and repeated four-frame sections.  This module
makes the distinction between that protocol, animal-disjoint CV, chronological
within-animal evaluation and external-cohort transfer explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from neurobranch.behavior.saav_alignment import AlignedDataset

Regime = Literal["original", "grouped", "chronological", "external"]
OFFLINE_MODES: tuple[str, ...] = ("behavior", "saav", "both")


@dataclass(frozen=True)
class Fold:
    outer: int
    inner: int
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    regime: str
    note: str = ""


def _stratified_indices(y: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """Return class-wise shuffled event indices."""
    groups = []
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        groups.append(idx[rng.permutation(len(idx))])
    return groups


def _assign_outer_folds(y: np.ndarray, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    """Assign folds separately within each class to preserve class ratio."""
    folds = np.full(len(y), -1, dtype=int)
    for group in _stratified_indices(y, rng):
        for k, idx in enumerate(group):
            folds[idx] = k % n_folds
    if np.any(folds < 0):
        raise ValueError("not every event received an outer fold")
    return folds


def _inner_split(
    train_idx: np.ndarray,
    y: np.ndarray,
    *,
    n_inner: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create class-stratified inner train/validation splits."""
    folds = _assign_outer_folds(y[train_idx], n_inner, rng)
    out = []
    for j in range(n_inner):
        valid = train_idx[folds == j]
        train = train_idx[folds != j]
        if len(valid) == 0 or len(train) == 0:
            continue
        out.append((np.sort(train), np.sort(valid)))
    return out


def make_folds(
    data: AlignedDataset,
    regime: Regime,
    *,
    n_outer: int = 5,
    n_inner: int = 4,
    seed: int = 17,
) -> list[Fold]:
    """Build exactly labelled folds for one evaluation regime."""
    rng = np.random.default_rng(seed)
    y = np.asarray(data.y, dtype=int)
    animals = data.meta["animal_id"].astype(str).to_numpy()
    folds: list[Fold] = []

    if regime in {"original", "grouped", "chronological"}:
        if regime == "original":
            outer = _assign_outer_folds(y, n_outer, rng)
            for k in range(n_outer):
                test = np.flatnonzero(outer == k)
                train_pool = np.flatnonzero(outer != k)
                for j, (train, valid) in enumerate(
                    _inner_split(train_pool, y, n_inner=n_inner, rng=rng)
                ):
                    folds.append(
                        Fold(k, j, train, valid, test, regime, "ungrouped event split")
                    )
        elif regime == "grouped":
            unique = np.array(sorted(set(animals)))
            shuffled = unique[rng.permutation(len(unique))]
            for k, held_out in enumerate(np.array_split(shuffled, n_outer)):
                test = np.flatnonzero(np.isin(animals, held_out))
                train_pool = np.flatnonzero(~np.isin(animals, held_out))
                for j, (train, valid) in enumerate(
                    _inner_split(train_pool, y, n_inner=n_inner, rng=rng)
                ):
                    folds.append(
                        Fold(k, j, train, valid, test, regime, "held-out animals")
                    )
        else:
            # Chronological within each animal: latest 20% of that animal's
            # events are test; training and validation come from earlier events.
            test_parts: list[np.ndarray] = []
            train_parts: list[np.ndarray] = []
            for animal in sorted(set(animals)):
                idx = np.flatnonzero(animals == animal)
                idx = idx[np.argsort(data.meta.iloc[idx]["start"].to_numpy())]
                n_test = max(1, int(round(0.2 * len(idx))))
                if len(idx) - n_test < 1:
                    continue
                train_parts.append(idx[:-n_test])
                test_parts.append(idx[-n_test:])
            train_pool = np.sort(np.concatenate(train_parts)) if train_parts else np.empty(0, int)
            test = np.sort(np.concatenate(test_parts)) if test_parts else np.empty(0, int)
            if len(test) and len(train_pool):
                for j, (train, valid) in enumerate(
                    _inner_split(train_pool, y, n_inner=n_inner, rng=rng)
                ):
                    folds.append(
                        Fold(j, j, train, valid, test, regime, "future block of each animal")
                    )
    elif regime == "external":
        train_pool = np.flatnonzero(data.meta["cohort"].to_numpy() == "wt_octanol")
        test = np.flatnonzero(data.meta["cohort"].to_numpy() == "wt_diacetyl")
        if len(train_pool) and len(test):
            for j, (train, valid) in enumerate(
                _inner_split(train_pool, y, n_inner=n_inner, rng=rng)
            ):
                folds.append(Fold(j, j, train, valid, test, regime, "octanol to diacetyl"))
    else:
        raise ValueError(f"unknown regime {regime!r}")

    for fold in folds:
        if set(fold.train) & set(fold.valid) or set(fold.train) & set(fold.test):
            raise AssertionError("train/validation/test overlap")
        if set(fold.valid) & set(fold.test):
            raise AssertionError("validation/test overlap")
    return folds


def make_sections(x: np.ndarray, *, section_frames: int = 4) -> np.ndarray:
    """All contiguous four-frame sections, shape (events, sections, frames, channels)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError("x must be (events, frames, channels)")
    if x.shape[1] < section_frames:
        raise ValueError("event is shorter than the section")
    return np.stack(
        [x[:, start : start + section_frames, :] for start in range(x.shape[1] - section_frames + 1)],
        axis=1,
    )


def flatten_sections(
    sections: np.ndarray,
    y: np.ndarray,
    event_ids: np.ndarray,
    event_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select event sections and return flat sections, labels and event groups."""
    selected = sections[event_mask]
    n_events, n_sections = selected.shape[:2]
    flat = selected.reshape(n_events * n_sections, selected.shape[2], selected.shape[3])
    labels = np.repeat(np.asarray(y)[event_mask], n_sections)
    groups = np.repeat(np.asarray(event_ids)[event_mask], n_sections)
    return flat, labels, groups


def calibration_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Logistic calibration slope/intercept; undefined for one class."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    logit = np.log(p / (1 - p))
    design = np.column_stack([np.ones_like(logit), logit])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[1]), float(coef[0])


def prior_shift(p: np.ndarray, prevalence: float) -> np.ndarray:
    """Shift class-balanced logits to a target prevalence without changing ranking."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    prevalence = float(np.clip(prevalence, 1e-4, 1 - 1e-4))
    logit = np.log(p / (1 - p)) + np.log(prevalence / (1 - prevalence))
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))


def event_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    """Proper scoring rules and threshold metrics for binary events."""
    y = np.asarray(y, dtype=int).ravel()
    p = np.clip(np.asarray(p, dtype=float).ravel(), 1e-6, 1 - 1e-6)
    if len(y) != len(p):
        raise ValueError("y and p length mismatch")
    out: dict[str, Any] = {"n": int(len(y)), "prevalence": float(np.mean(y)) if len(y) else float("nan")}
    if len(y) == 0:
        return out
    out["log_loss"] = float(log_loss(y, p, labels=[0, 1]))
    out["brier"] = float(brier_score_loss(y, p))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y, p >= 0.5))
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
        slope, intercept = calibration_slope(y, p)
        out["calibration_slope"] = slope
        out["calibration_intercept"] = intercept
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
        out["calibration_slope"] = float("nan")
        out["calibration_intercept"] = float("nan")
    return out


def section_balanced_accuracy(
    y: np.ndarray,
    p: np.ndarray,
    *,
    groups: np.ndarray,
) -> float:
    """Official-style section accuracy: average class means, not event means."""
    frame = pd.DataFrame({"y": np.asarray(y, dtype=int), "p": np.asarray(p, dtype=float), "g": groups})
    # One score per section, averaging any per-timestep predictions already done.
    per_section = frame.groupby("g", sort=False).agg({"y": "first", "p": "mean"})
    if per_section["y"].nunique() < 2:
        return float("nan")
    correct = (per_section["p"] >= 0.5).astype(int) == per_section["y"].astype(int)
    return float(correct.groupby(per_section["y"]).mean().mean())


def _resample_groups(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique = np.unique(groups)
    chosen = rng.choice(unique, size=len(unique), replace=True)
    pieces = [np.flatnonzero(groups == g) for g in chosen]
    idx = np.concatenate(pieces)
    return y[idx], p_a[idx], p_b[idx]


def paired_cluster_bootstrap(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    groups: np.ndarray,
    *,
    metric: str = "auroc",
    n_boot: int = 2000,
    seed: int = 17,
) -> dict[str, Any]:
    """Paired animal-cluster bootstrap for a metric difference.

    ``delta`` is model B minus model A.  For AUROC, positive is better.  For
    log_loss/brier, negative is better.  The returned interval is on ``delta``.
    """
    y = np.asarray(y, dtype=int)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    groups = np.asarray(groups)
    if len(y) != len(p_a) or len(y) != len(p_b) or len(y) != len(groups):
        raise ValueError("all inputs must have the same length")
    if len(np.unique(y)) < 2:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p": float("nan")}
    rng = np.random.default_rng(seed)

    def score(yy: np.ndarray, pp: np.ndarray) -> float:
        if metric == "auroc":
            return float(roc_auc_score(yy, pp))
        if metric == "auprc":
            return float(average_precision_score(yy, pp))
        if metric == "log_loss":
            return float(log_loss(yy, np.clip(pp, 1e-6, 1 - 1e-6), labels=[0, 1]))
        if metric == "brier":
            return float(brier_score_loss(yy, pp))
        if metric == "balanced_accuracy":
            return float(balanced_accuracy_score(yy, pp >= 0.5))
        raise ValueError(f"unknown metric {metric!r}")

    observed = score(y, p_b) - score(y, p_a)
    deltas: list[float] = []
    for _ in range(n_boot):
        yy, aa, bb = _resample_groups(y, p_a, p_b, groups, rng)
        if len(np.unique(yy)) < 2:
            continue
        deltas.append(score(yy, bb) - score(yy, aa))
    if not deltas:
        return {"delta": observed, "ci_low": float("nan"), "ci_high": float("nan"), "p": float("nan")}
    deltas_arr = np.asarray(deltas)
    if metric in {"log_loss", "brier"}:
        p_value = float(np.clip(np.mean(deltas_arr >= 0), 0.0, 1.0))  # probability of no improvement
    else:
        p_value = float(np.clip(np.mean(deltas_arr <= 0), 0.0, 1.0))
    p_two_sided = float(np.clip(2 * min(np.mean(deltas_arr <= 0), np.mean(deltas_arr >= 0)), 0.0, 1.0))
    return {
        "delta": float(observed),
        "ci_low": float(np.quantile(deltas_arr, 0.025)),
        "ci_high": float(np.quantile(deltas_arr, 0.975)),
        "p": p_value,
        "p_two_sided": p_two_sided,
        "n_boot": int(len(deltas_arr)),
    }


def summarize_by_animal(
    meta: pd.DataFrame,
    y: np.ndarray,
    p: np.ndarray,
    *,
    mode: str,
) -> pd.DataFrame:
    """Per-animal metrics, with AUC only where both classes exist."""
    frame = meta[["animal_id", "cohort"]].copy()
    frame["y"] = np.asarray(y, dtype=int)
    frame["p"] = np.asarray(p, dtype=float)
    rows = []
    for (animal, cohort), group in frame.groupby(["animal_id", "cohort"]):
        rows.append({"animal_id": animal, "cohort": cohort, "mode": mode, **event_metrics(group.y.to_numpy(), group.p.to_numpy())})
    return pd.DataFrame(rows)


def animal_overlap(meta: pd.DataFrame, train: np.ndarray, test: np.ndarray) -> dict[str, Any]:
    train_animals = set(meta.iloc[train]["animal_id"].astype(str))
    test_animals = set(meta.iloc[test]["animal_id"].astype(str))
    return {
        "n_train_animals": len(train_animals),
        "n_test_animals": len(test_animals),
        "n_overlap_animals": len(train_animals & test_animals),
        "fraction_test_events_with_train_animal": float(
            meta.iloc[test]["animal_id"].astype(str).isin(train_animals).mean()
        ),
    }
