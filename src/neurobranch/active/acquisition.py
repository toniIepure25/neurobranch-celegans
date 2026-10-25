"""Active target acquisition for the Phase 7 response-operator benchmark.

The policies here select *which target to stimulate next* under a fixed
candidate pool.  They are intentionally small, deterministic heuristics, not a
claim of optimal Bayesian design.  The hybrid policy is preregistered; every
other policy is a comparator or ablation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STRATEGIES = ("random", "uncertainty", "effect", "diversity", "discrepancy", "hybrid")


def robust_z(values: np.ndarray) -> np.ndarray:
    """Robust standardisation that is safe for constant score vectors."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    median = float(np.nanmedian(values))
    iqr = float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25))
    scale = iqr if iqr > 1e-9 else float(np.nanstd(values))
    if scale <= 1e-9:
        return np.zeros_like(values)
    return np.clip((values - median) / scale, -5.0, 5.0)


def _pairwise_min_distance(features: np.ndarray, chosen: np.ndarray) -> np.ndarray:
    """Minimum Euclidean distance from every candidate to the chosen set."""
    if chosen.size == 0:
        return np.zeros(features.shape[0], dtype=float)
    diff = features[:, None, :] - features[None, chosen, :]
    return np.sqrt(np.sum(diff**2, axis=2)).min(axis=1)


def _min_distance_to_reference(features: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if reference.size == 0:
        return np.zeros(features.shape[0], dtype=float)
    diff = features[:, None, :] - reference[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=2)).min(axis=1)


@dataclass(frozen=True)
class AcquisitionContext:
    candidates: tuple[str, ...]
    features: np.ndarray
    uncertainty: dict[str, float]
    effect: dict[str, float]
    discrepancy: dict[str, float]
    acquired_features: np.ndarray
    rng: np.random.Generator


def _score_vector(ctx: AcquisitionContext, strategy: str) -> np.ndarray:
    uncertainty = np.array([ctx.uncertainty.get(t, 0.0) for t in ctx.candidates], dtype=float)
    effect = np.array([ctx.effect.get(t, 0.0) for t in ctx.candidates], dtype=float)
    discrepancy = np.array([ctx.discrepancy.get(t, 0.0) for t in ctx.candidates], dtype=float)
    if strategy == "uncertainty":
        return robust_z(uncertainty)
    if strategy == "effect":
        return robust_z(effect)
    if strategy == "discrepancy":
        return robust_z(discrepancy)
    if strategy == "diversity":
        return robust_z(_pairwise_min_distance(ctx.features, np.empty(0, dtype=int)))
    if strategy == "hybrid":
        return robust_z(uncertainty) + 0.5 * robust_z(discrepancy)
    if strategy == "random":
        return np.zeros(len(ctx.candidates), dtype=float)
    raise ValueError(f"unknown acquisition strategy {strategy!r}")


def select_batch(ctx: AcquisitionContext, strategy: str, batch_size: int) -> list[str]:
    """Select a batch without replacement.

    For ``diversity`` and ``hybrid`` the diversity term is recomputed after each
    pick, which makes the batch greedy and avoids selecting near-duplicates.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown acquisition strategy {strategy!r}")
    if batch_size < 1 or not ctx.candidates:
        return []
    if ctx.features.shape[0] != len(ctx.candidates):
        raise ValueError("feature rows and candidate names disagree")

    base = _score_vector(ctx, strategy)
    selected: list[int] = []
    chosen = np.empty(0, dtype=int)
    for _ in range(min(batch_size, len(ctx.candidates))):
        available = np.setdiff1d(np.arange(len(ctx.candidates)), np.asarray(selected, dtype=int))
        if available.size == 0:
            break
        if strategy == "random":
            pick = int(ctx.rng.choice(available))
        else:
            score = base[available].copy()
            if strategy in {"diversity", "hybrid"}:
                reference = ctx.acquired_features
                if chosen.size:
                    reference = np.vstack([reference, ctx.features[chosen]])
                min_dist = _min_distance_to_reference(ctx.features[available], reference)
                diversity = robust_z(min_dist)
                score = (
                    diversity
                    if strategy == "diversity"
                    else score + 0.25 * diversity
                )
            pick = int(available[int(np.argmax(score))])
        selected.append(pick)
        chosen = np.append(chosen, pick)
    return [ctx.candidates[i] for i in selected]
