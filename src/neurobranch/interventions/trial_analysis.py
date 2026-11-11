"""Animal-level analysis for randomized intervention trials.

This is the conservative fallback when a mixed-effects logistic model is not
available: reduce each animal to a proportion, compare arms at the animal level,
bootstrap animals within arm, and use an animal-label permutation test.  It never
treats trials or frames as independent observations.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np


def _animal_proportions(
    rows: Iterable[dict[str, Any]],
    arm: str,
) -> tuple[dict[str, float], int]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if str(row.get("arm", "")).strip() != arm:
            continue
        animal = str(row.get("animal_id", "")).strip()
        if not animal:
            raise ValueError("animal_id is empty")
        try:
            outcome = float(row.get("outcome", ""))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"outcome must be binary, got {row.get('outcome')!r}") from exc
        if outcome not in (0.0, 1.0):
            raise ValueError(f"outcome must be binary, got {outcome!r}")
        grouped[animal].append(outcome)
    proportions = {animal: float(np.mean(values)) for animal, values in grouped.items()}
    return proportions, sum(len(values) for values in grouped.values())


def analyze_trial_outcomes(
    rows: Iterable[dict[str, Any]],
    *,
    n_bootstrap: int = 2000,
    n_permutations: int = 2000,
    seed: int = 20261111,
) -> dict[str, Any]:
    """Estimate the animal-level target-minus-sham risk difference."""
    materialized = list(rows)
    target, n_target_trials = _animal_proportions(materialized, "target")
    sham, n_sham_trials = _animal_proportions(materialized, "sham")
    if not target or not sham:
        raise ValueError("both target and sham arms are required")
    if len(target) < 2 or len(sham) < 2:
        raise ValueError("at least two animals per arm are required for a group comparison")
    if n_bootstrap < 1 or n_permutations < 1:
        raise ValueError("bootstrap and permutation counts must be positive")

    target_values = np.array(list(target.values()), dtype=float)
    sham_values = np.array(list(sham.values()), dtype=float)
    effect = float(target_values.mean() - sham_values.mean())
    rng = np.random.default_rng(seed)

    boot = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        boot[index] = (
            rng.choice(target_values, size=len(target_values), replace=True).mean()
            - rng.choice(sham_values, size=len(sham_values), replace=True).mean()
        )
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    pooled = np.concatenate([target_values, sham_values])
    n_target = len(target_values)
    perm = np.empty(n_permutations, dtype=float)
    for index in range(n_permutations):
        shuffled = rng.permutation(pooled)
        perm[index] = shuffled[:n_target].mean() - shuffled[n_target:].mean()
    p_value = float((1 + np.sum(np.abs(perm) >= abs(effect))) / (n_permutations + 1))

    return {
        "format": "vienna-intervention-analysis-v1",
        "effect": effect,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": p_value,
        "n_target_animals": len(target_values),
        "n_sham_animals": len(sham_values),
        "n_target_trials": n_target_trials,
        "n_sham_trials": n_sham_trials,
        "n_bootstrap": n_bootstrap,
        "n_permutations": n_permutations,
        "seed": seed,
        "method": "animal-level proportions; percentile bootstrap; label permutation",
        "caveat": (
            "One randomized experiment is hypothesis-generating; a causal claim "
            "requires replication and a working positive control."
        ),
    }
