"""Leakage-safe grouped splits for the Phase 6 response benchmark.

Splits are built from the long-form table, not from a model's feature matrix.
Every split holds out whole groups; a row can appear in exactly one of train,
validation and test within a fold.  The functions are deterministic for a given
seed and are covered by explicit overlap tests.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from neurobranch.data.schemas import SchemaError

SPLIT_NAMES = (
    "unseen_targets",
    "unseen_pairs",
    "unseen_animals",
    "unseen_targets_and_animals",
)


@dataclass(frozen=True)
class Fold:
    split: str
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    held_out: tuple[str, ...]
    seed: int

    def manifest(self) -> dict[str, object]:
        payload = {
            "split": self.split,
            "fold": self.fold,
            "seed": self.seed,
            "n_train": int(len(self.train_idx)),
            "n_val": int(len(self.val_idx)),
            "n_test": int(len(self.test_idx)),
            "held_out": list(self.held_out),
            "train_hash": _hash_indices(self.train_idx),
            "val_hash": _hash_indices(self.val_idx),
            "test_hash": _hash_indices(self.test_idx),
        }
        payload["manifest_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return payload


def _hash_indices(idx: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(idx, dtype="<i8").tobytes()).hexdigest()


def _balanced_group_folds(groups: pd.Series, n_folds: int, *, seed: int) -> dict[str, int]:
    """Assign whole groups to folds, balancing total row counts.

    Greedy largest-first assignment is deterministic apart from ties, which are
    broken by the seeded RNG.  It avoids the common failure mode where a rare
    target is placed in a fold that then has almost no test events.
    """
    counts = groups.value_counts()
    keys = list(counts.index)
    rng = np.random.default_rng(seed)
    order = sorted(
        range(len(keys)),
        key=lambda i: (-int(counts.iloc[i]), rng.random(), str(keys[i])),
    )
    totals = np.zeros(n_folds, dtype=np.int64)
    assignment: dict[str, int] = {}
    for i in order:
        fold = int(np.argmin(totals))
        key = str(keys[i])
        assignment[key] = fold
        totals[fold] += int(counts.iloc[i])
    if any(totals == 0):
        raise SchemaError(f"cannot populate {n_folds} folds from {len(keys)} groups")
    return assignment


def _require_columns(table: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise SchemaError(f"table missing columns {missing}")


def _fold_from_masks(
    *,
    split: str,
    fold: int,
    seed: int,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    held_out: list[str],
) -> Fold:
    if (train_mask & val_mask).any() or (train_mask & test_mask).any() or (val_mask & test_mask).any():
        raise SchemaError(f"{split}: masks overlap")
    if not test_mask.any():
        raise SchemaError(f"{split}: empty test mask")
    if not val_mask.any():
        raise SchemaError(f"{split}: empty validation mask")
    return Fold(
        split=split,
        fold=fold,
        train_idx=np.flatnonzero(train_mask),
        val_idx=np.flatnonzero(val_mask),
        test_idx=np.flatnonzero(test_mask),
        held_out=tuple(held_out),
        seed=seed,
    )


def build_folds(
    table: pd.DataFrame,
    split: str,
    *,
    n_folds: int = 5,
    seed: int = 17,
    val_offset: int = 1,
) -> list[Fold]:
    """Build all folds for a named split.

    ``unseen_targets`` holds out entire stimulated neurons; ``unseen_pairs``
    holds out target-responder pairs; ``unseen_animals`` holds out animals.
    The combined split holds out both target and animal groups, which is the
    hardest and is reported separately.
    """
    _require_columns(table, ("dataset_id", "target", "neuron"))
    if split not in SPLIT_NAMES:
        raise SchemaError(f"unknown split {split!r}")
    if n_folds < 2:
        raise SchemaError("n_folds must be >= 2")

    n = len(table)
    if n == 0:
        raise SchemaError("empty response table")

    if split == "unseen_pairs":
        group = table["target"].astype(str) + "|||" + table["neuron"].astype(str)
    elif split == "unseen_targets":
        group = table["target"].astype(str)
    elif split == "unseen_animals":
        group = table["dataset_id"].astype(str)
    else:
        group = None

    if group is not None:
        assignment = _balanced_group_folds(group, n_folds, seed=seed)
        group_fold = group.map(assignment).to_numpy()
        folds = []
        for test_fold in range(n_folds):
            val_fold = (test_fold + val_offset) % n_folds
            train_mask = (group_fold != test_fold) & (group_fold != val_fold)
            val_mask = group_fold == val_fold
            test_mask = group_fold == test_fold
            held = sorted(k for k, v in assignment.items() if v == test_fold)
            folds.append(
                _fold_from_masks(
                    split=split,
                    fold=test_fold,
                    seed=seed,
                    train_mask=train_mask,
                    val_mask=val_mask,
                    test_mask=test_mask,
                    held_out=held,
                )
            )
        return folds

    # Combined target + animal holdout.  Both group families are assigned
    # independently; test/validation rows are the union, train is the complement.
    target_group = table["target"].astype(str)
    animal_group = table["dataset_id"].astype(str)
    target_assignment = _balanced_group_folds(target_group, n_folds, seed=seed)
    animal_assignment = _balanced_group_folds(animal_group, n_folds, seed=seed + 101)
    target_fold = target_group.map(target_assignment).to_numpy()
    animal_fold = animal_group.map(animal_assignment).to_numpy()
    folds = []
    for test_fold in range(n_folds):
        val_fold = (test_fold + val_offset) % n_folds
        test_mask = (target_fold == test_fold) | (animal_fold == test_fold)
        val_mask = ((target_fold == val_fold) | (animal_fold == val_fold)) & ~test_mask
        train_mask = ~(test_mask | val_mask)
        held = sorted(
            {f"target:{k}" for k, v in target_assignment.items() if v == test_fold}
            | {f"animal:{k}" for k, v in animal_assignment.items() if v == test_fold}
        )
        folds.append(
            _fold_from_masks(
                split=split,
                fold=test_fold,
                seed=seed,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                held_out=held,
            )
        )
    return folds
