"""Fast Phase 8 unit tests that do not read the raw atlas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from neurobranch.twin.evaluate import frame_metrics, paired_metric_delta, trajectory_pearson
from neurobranch.twin.model import ResponseTwin, TwinConfig
from neurobranch.twin.tasks import AnimalTask, AnimalTaskSplit, _split_targets
from neurobranch.twin.train import support_profile_raw
from neurobranch.twin.windows import EventWindowCache, WindowConfig


def synthetic_cache(n_events: int = 12, n_resp: int = 4) -> EventWindowCache:
    rng = np.random.default_rng(7)
    animals = ["A", "B", "C"]
    targets = ("T0", "T1", "T2", "T3")
    responders = tuple(f"R{i}" for i in range(n_resp))
    target_idx = np.arange(n_events) % len(targets)
    dataset_ids = np.array([animals[i % len(animals)] for i in range(n_events)])
    mask = np.ones((n_events, n_resp), dtype=bool)
    pre = rng.normal(size=(n_events, n_resp, 3)).astype(np.float32)
    post = rng.normal(size=(n_events, n_resp, 4)).astype(np.float32)
    scalar = post.mean(axis=-1).astype(np.float32)
    history = rng.normal(size=(n_events, 8)).astype(np.float32)
    return EventWindowCache(
        pre=pre,
        post=post,
        mask=mask,
        scalar=scalar,
        history=history,
        history_names=(
            "log_prev_interval",
            "prev_same_target",
            "event_order_scaled",
            "pre_activity_median",
            "pre_sd_median",
            "inverse_interval",
            "interval_missing",
            "event_order_missing",
        ),
        event_ids=np.array([f"{d}:{i}" for i, d in enumerate(dataset_ids)]),
        dataset_ids=dataset_ids,
        target_names=targets,
        responder_names=responders,
        animal_names=tuple(animals),
        target_idx=target_idx,
        event_target=np.array([targets[i] for i in target_idx]),
        event_index=np.arange(n_events, dtype=float),
        stim_frame=np.arange(n_events, dtype=float) * 10,
        config=WindowConfig(),
        source_trials="synthetic",
        source_raw_dir="synthetic",
        trials_sha256="0" * 64,
    )


def test_support_profile_is_permutation_invariant_and_empty_safe() -> None:
    cache = synthetic_cache()
    a = support_profile_raw(cache, [str(e) for e in cache.event_ids[:3]])
    b = support_profile_raw(cache, [str(e) for e in cache.event_ids[2::-1]])
    assert a.shape == (16,)
    assert np.allclose(a, b, atol=1e-6)
    assert support_profile_raw(cache, []).tolist() == [0.0] * 16


def test_target_split_is_disjoint() -> None:
    support, test = _split_targets(
        ["a", "b", "c", "d", "e", "f"],
        fraction=0.5,
        rng=np.random.default_rng(3),
        min_support=2,
        min_test=2,
    )
    assert set(support).isdisjoint(test)
    assert len(support) >= 2 and len(test) >= 2


def test_response_twin_forward_shapes() -> None:
    cfg = TwinConfig(rank=4, hidden=8, temporal_basis=3, support_dim=16, dropout=0.0)
    model = ResponseTwin(
        d_target=5,
        d_responder=4,
        d_pair=2,
        d_context=8,
        pre_frames=3,
        post_frames=4,
        config=cfg,
        support_dim=16,
    )
    b = 7
    out = model(
        torch.zeros(b, 5),
        torch.zeros(b, 4),
        torch.zeros(b, 2),
        torch.zeros(b, 8),
        torch.zeros(b, 3),
        torch.zeros(b, 3),
        torch.zeros(b, 16),
        torch.zeros(b, 4),
        support_active=torch.zeros(b),
    )
    smu, ssig, tmu, tsig = out
    assert smu.shape == (b,)
    assert ssig.shape == (b,)
    assert tmu.shape == (b, 4)
    assert tsig.shape == (b, 4)
    assert torch.all(ssig > 0) and torch.all(tsig > 0)


def test_trajectory_and_paired_delta_are_finite() -> None:
    rng = np.random.default_rng(11)
    y = rng.normal(size=(20, 4))
    assert np.isfinite(trajectory_pearson(y, y + 0.1 * rng.normal(size=y.shape)))
    frame = pd.DataFrame(
        {
            "event_id": [f"e{i // 8}" for i in range(40)],
            "dataset_id": ["A", "B", "C", "A", "B", "C", "A", "B"] * 5,
            "target": ["T0", "T1", "T2", "T3"] * 10,
            "neuron": ["R0", "R1", "R2", "R3"] * 10,
            "y_true": rng.normal(size=40),
            "y_pred": rng.normal(size=40),
        }
    )
    assert np.isfinite(frame_metrics(frame)["pair_spearman"])
    delta = paired_metric_delta(frame, frame, n_boot=10, seed=3)
    assert abs(delta["delta"]) < 1e-12


def test_task_dataclass_rejects_nothing_but_manifest_is_stable() -> None:
    task = AnimalTask(
        animal="A",
        support_events=("A:1", "A:2"),
        test_events=("A:3",),
        support_targets=("T0",),
        test_targets=("T1",),
    )
    split = AnimalTaskSplit(
        split="unseen_animals",
        fold=0,
        seed=37,
        support_fraction=0.5,
        train_animals=("A",),
        val_animals=(),
        test_animals=(),
        tasks={"A": task},
    )
    assert task.manifest()["n_support_events"] == 2
    assert split.manifest()["fold"] == 0
