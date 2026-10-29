"""Leakage-safe animal tasks with target-disjoint support/test sets.

Phase 7 proposed targets at the level of the whole atlas.  Phase 8 asks a
different question: given a *new animal*, can a small support set of observed
targets calibrate predictions for a disjoint set of held-out targets?  The
support and test targets are therefore disjoint by construction, and animals
are never shared between train/validation/test within a fold.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from neurobranch.benchmark.splits import build_folds
from neurobranch.twin.windows import EventWindowCache


def _hash_ids(values: tuple[str, ...]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_seed(seed: int, animal: str) -> int:
    digest = hashlib.sha256(f"{seed}:{animal}".encode()).hexdigest()
    return int(digest[:8], 16)


@dataclass(frozen=True)
class AnimalTask:
    """One animal's support/test split, with target-disjoint event sets."""

    animal: str
    support_events: tuple[str, ...]
    test_events: tuple[str, ...]
    support_targets: tuple[str, ...]
    test_targets: tuple[str, ...]

    def manifest(self) -> dict[str, Any]:
        payload = {
            "animal": self.animal,
            "n_support_events": len(self.support_events),
            "n_test_events": len(self.test_events),
            "n_support_targets": len(self.support_targets),
            "n_test_targets": len(self.test_targets),
            "support_hash": _hash_ids(self.support_events),
            "test_hash": _hash_ids(self.test_events),
            "support_target_hash": _hash_ids(self.support_targets),
            "test_target_hash": _hash_ids(self.test_targets),
        }
        payload["manifest_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return payload


@dataclass
class AnimalTaskSplit:
    split: str
    fold: int
    seed: int
    support_fraction: float
    train_animals: tuple[str, ...]
    val_animals: tuple[str, ...]
    test_animals: tuple[str, ...]
    tasks: dict[str, AnimalTask]

    @property
    def train_tasks(self) -> dict[str, AnimalTask]:
        return {a: self.tasks[a] for a in self.train_animals if a in self.tasks}

    @property
    def val_tasks(self) -> dict[str, AnimalTask]:
        return {a: self.tasks[a] for a in self.val_animals if a in self.tasks}

    @property
    def test_tasks(self) -> dict[str, AnimalTask]:
        return {a: self.tasks[a] for a in self.test_animals if a in self.tasks}

    def manifest(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "split": self.split,
            "fold": self.fold,
            "seed": self.seed,
            "support_fraction": self.support_fraction,
            "train_animals": list(self.train_animals),
            "val_animals": list(self.val_animals),
            "test_animals": list(self.test_animals),
            "tasks": {a: self.tasks[a].manifest() for a in sorted(self.tasks)},
        }
        payload["manifest_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return payload


def _event_frame(cache: EventWindowCache) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_id": cache.dataset_ids.astype(str),
            "target": [cache.target_names[i] for i in cache.target_idx],
            "neuron": "__event__",
            "event_id": cache.event_ids.astype(str),
        }
    )


def _split_targets(
    targets: list[str],
    *,
    fraction: float,
    rng: np.random.Generator,
    min_support: int,
    min_test: int,
) -> tuple[list[str], list[str]]:
    unique = sorted(set(targets))
    if len(unique) < min_support + min_test:
        return [], []
    order = list(rng.permutation(unique))
    n_support = int(round(len(unique) * fraction))
    n_support = max(min_support, min(len(unique) - min_test, n_support))
    support = sorted(order[:n_support])
    test = sorted(order[n_support:])
    if len(test) < min_test:
        return [], []
    return support, test


def _make_task(
    cache: EventWindowCache,
    animal: str,
    events: list[str],
    *,
    fixed_test: list[str] | None,
    support_fraction: float,
    rng: np.random.Generator,
    min_support_targets: int,
    min_test_targets: int,
) -> AnimalTask | None:
    if not events:
        return None
    event_idx = {e: i for i, e in enumerate(cache.event_ids.astype(str))}
    by_target: dict[str, list[str]] = {}
    for e in events:
        i = event_idx.get(e)
        if i is None:
            continue
        target = cache.target_names[int(cache.target_idx[i])]
        by_target.setdefault(target, []).append(e)
    if fixed_test is None:
        support_targets, test_targets = _split_targets(
            list(by_target),
            fraction=support_fraction,
            rng=rng,
            min_support=min_support_targets,
            min_test=min_test_targets,
        )
        test_events = [e for t in test_targets for e in by_target[t]]
    else:
        test_events = sorted(set(fixed_test))
        test_targets_set = set()
        for e in test_events:
            i = event_idx.get(e)
            if i is not None:
                test_targets_set.add(cache.target_names[int(cache.target_idx[i])])
        test_targets = sorted(test_targets_set)
        candidate_targets = [t for t in by_target if t not in test_targets_set]
        if len(candidate_targets) < min_support_targets:
            return None
        # For a fixed test set, use all target-disjoint candidates as support.
        support_targets = sorted(candidate_targets)
    if len(support_targets) < min_support_targets or len(test_targets) < min_test_targets:
        return None
    support_events_t = tuple(sorted(e for t in support_targets for e in by_target.get(t, [])))
    test_events_t = tuple(sorted(test_events))
    if not support_events_t or not test_events_t:
        return None
    support_targets_t = tuple(sorted(support_targets))
    test_targets_t = tuple(sorted(test_targets))
    if set(support_targets_t) & set(test_targets_t):
        raise ValueError(f"support/test target overlap for animal {animal}")
    if set(support_events_t) & set(test_events_t):
        raise ValueError(f"support/test event overlap for animal {animal}")
    return AnimalTask(
        animal=animal,
        support_events=support_events_t,
        test_events=test_events_t,
        support_targets=support_targets_t,
        test_targets=test_targets_t,
    )


def build_animal_tasks(
    cache: EventWindowCache,
    *,
    split: str = "unseen_animals",
    n_folds: int = 5,
    seed: int = 17,
    support_fraction: float = 0.5,
    min_support_targets: int = 2,
    min_test_targets: int = 2,
) -> list[AnimalTaskSplit]:
    """Build folds of animal tasks.

    ``unseen_animals`` is the primary split.  For each held-out animal, its own
    events are split into target-disjoint support/test sets.  A task's support
    labels are observed at inference; its test labels are never used for model
    fitting.  Other split names use the Phase 6 grouped split machinery and then
    enforce target-disjoint support/test sets as well.
    """
    if not 0.0 < support_fraction < 1.0:
        raise ValueError("support_fraction must be in (0, 1)")
    events = _event_frame(cache)
    folds = build_folds(events, split, n_folds=n_folds, seed=seed)
    all_events: dict[str, list[str]] = {}
    for animal, frame in events.groupby("dataset_id", sort=False):
        all_events[str(animal)] = sorted(frame["event_id"].astype(str).tolist())

    out: list[AnimalTaskSplit] = []
    for fold in folds:
        fold_train = set(events.iloc[fold.train_idx]["event_id"].astype(str))
        fold_val = set(events.iloc[fold.val_idx]["event_id"].astype(str))
        fold_test = set(events.iloc[fold.test_idx]["event_id"].astype(str))
        train_animals = tuple(sorted(events.iloc[fold.train_idx]["dataset_id"].astype(str).unique()))
        val_animals = tuple(sorted(events.iloc[fold.val_idx]["dataset_id"].astype(str).unique()))
        test_animals = tuple(sorted(events.iloc[fold.test_idx]["dataset_id"].astype(str).unique()))
        tasks: dict[str, AnimalTask] = {}

        for animal in train_animals:
            role_events = sorted(set(all_events[animal]) & fold_train)
            task = _make_task(
                cache,
                animal,
                role_events,
                fixed_test=None,
                support_fraction=support_fraction,
                rng=np.random.default_rng(_stable_seed(seed, animal)),
                min_support_targets=min_support_targets,
                min_test_targets=min_test_targets,
            )
            if task is not None:
                tasks[animal] = task
        for animal in val_animals:
            role_events = sorted(set(all_events[animal]) & fold_val)
            task = _make_task(
                cache,
                animal,
                role_events,
                fixed_test=None,
                support_fraction=support_fraction,
                rng=np.random.default_rng(_stable_seed(seed, animal)),
                min_support_targets=min_support_targets,
                min_test_targets=min_test_targets,
            )
            if task is not None:
                tasks[animal] = task
        for animal in test_animals:
            if split == "unseen_animals":
                role_events = sorted(all_events[animal])
                fixed_test = None
            else:
                role_events = sorted(all_events[animal])
                fixed_test = sorted(set(all_events[animal]) & fold_test)
            task = _make_task(
                cache,
                animal,
                role_events,
                fixed_test=fixed_test,
                support_fraction=support_fraction,
                rng=np.random.default_rng(_stable_seed(seed, animal)),
                min_support_targets=min_support_targets,
                min_test_targets=min_test_targets,
            )
            if task is not None:
                tasks[animal] = task

        # Test animals without enough target-disjoint events are excluded from
        # the fitted task list; they remain visible in the manifest via
        # ``test_animals`` and are reported as exclusions by the runner.
        out.append(
            AnimalTaskSplit(
                split=split,
                fold=fold.fold,
                seed=seed,
                support_fraction=support_fraction,
                train_animals=tuple(a for a in train_animals if a in tasks),
                val_animals=tuple(a for a in val_animals if a in tasks),
                test_animals=tuple(a for a in test_animals if a in tasks),
                tasks=tasks,
            )
        )
    return out
