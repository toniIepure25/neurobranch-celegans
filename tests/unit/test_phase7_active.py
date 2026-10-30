"""Unit tests for Phase 7 active acquisition and leakage controls."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neurobranch.active.acquisition import AcquisitionContext, select_batch
from neurobranch.active.simulate import ActiveConfig, _choose_seed_targets, run_active_benchmark
from neurobranch.active.state_features import StateCache, fit_state_projector
from neurobranch.benchmark.conformal import conformal_interval, conformal_quantile
from neurobranch.benchmark.metrics import interval_score
from neurobranch.config import PROJECT_ROOT

CONNECTOME = PROJECT_ROOT / "references" / "external" / "aconnectome_witvliet_2020_8_utf8.csv"


def _synthetic_trials(n_targets: int = 20, n_responders: int = 8, n_animals: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    targets = [f"T{i:02d}" for i in range(n_targets)]
    responders = [f"R{i:02d}" for i in range(n_responders)]
    rows = []
    for a in range(n_animals):
        for e in range(3):
            target = targets[(a * 3 + e) % n_targets]
            for responder in responders:
                rows.append(
                    {
                        "dataset_id": f"a{a}",
                        "event_id": f"a{a}:{e}",
                        "event_index": e,
                        "target": target,
                        "neuron": responder,
                        "response_mean": float(
                            rng.normal(0.0, 1.0)
                            + 0.4 * (targets.index(target) % 3)
                            - 0.3 * (responders.index(responder) % 2)
                        ),
                        "pre_activity": float(rng.normal()),
                        "pre_sd": float(abs(rng.normal())),
                        "prev_stim_interval": float(e + 1),
                        "prev_same_target": 0.0,
                        "event_order": e,
                    }
                )
    return pd.DataFrame(rows)


def test_select_batch_no_duplicates_and_deterministic() -> None:
    rng = np.random.default_rng(3)
    ctx = AcquisitionContext(
        candidates=("A", "B", "C", "D", "E"),
        features=rng.normal(size=(5, 3)),
        uncertainty={"A": 1.0, "B": 0.5, "C": 0.2, "D": 0.1, "E": 0.0},
        effect={"A": 0.2, "B": 0.9, "C": 0.4, "D": 0.1, "E": 0.3},
        discrepancy={"A": 0.0, "B": 0.1, "C": 0.9, "D": 0.2, "E": 0.3},
        acquired_features=np.zeros((0, 3)),
        rng=np.random.default_rng(11),
    )
    first = select_batch(ctx, "hybrid", 3)
    second = select_batch(ctx, "hybrid", 3)
    assert first == second
    assert len(first) == len(set(first)) == 3
    assert all(t in ctx.candidates for t in first)


def test_diversity_prefers_far_point() -> None:
    ctx = AcquisitionContext(
        candidates=("near", "far"),
        features=np.array([[0.0, 0.0], [10.0, 10.0]]),
        uncertainty={},
        effect={},
        discrepancy={},
        acquired_features=np.array([[0.1, 0.1]]),
        rng=np.random.default_rng(1),
    )
    assert select_batch(ctx, "diversity", 1) == ["far"]


def test_seed_targets_are_subset_and_stratified() -> None:
    pool = [f"T{i}" for i in range(30)]
    counts = {t: (1 if i < 10 else 3 + i % 4) for i, t in enumerate(pool)}
    chosen = _choose_seed_targets(pool, counts, 12, np.random.default_rng(5))
    assert len(chosen) == 12
    assert set(chosen).issubset(pool)
    assert len(set(chosen)) == 12


def test_conformal_and_interval_score() -> None:
    rng = np.random.default_rng(2)
    y = rng.normal(size=100)
    mu = np.zeros(100)
    sigma = np.ones(100)
    q = conformal_quantile(y, mu, sigma, level=0.9)
    assert 0.5 < q < 5.0
    lower, upper = conformal_interval(mu, sigma, q)
    frame = pd.DataFrame({"y_true": y, "lower": lower, "upper": upper})
    assert 0.75 <= float(np.mean((y >= lower) & (y <= upper))) <= 1.0
    assert np.isfinite(interval_score(frame))


def test_state_projector_uses_train_fill_only() -> None:
    cache = StateCache(
        event_ids=tuple(f"e{i}" for i in range(12)),
        node_names=("A", "B", "C", "D"),
        values=np.arange(48, dtype=float).reshape(12, 4),
        mask=np.ones((12, 4), dtype=bool),
    )
    projector = fit_state_projector(cache, train_event_ids=tuple(f"e{i}" for i in range(10)), n_components=2)
    expected = cache.values[:10].mean(axis=0)
    assert np.allclose(projector.fill_values, expected)
    out = projector.transform(cache, ["e1", "e11"])
    assert out.shape == (2, 2)
    assert set(projector.train_event_ids) == {f"e{i}" for i in range(10)}


@pytest.mark.skipif(not CONNECTOME.exists(), reason="Witvliet connectome not present")
def test_active_end_to_end_audit(tmp_path: Path) -> None:
    trials = _synthetic_trials()
    cfg = ActiveConfig(
        trials_path="unused",
        output_dir=str(tmp_path),
        n_folds=4,
        initial_fraction=0.20,
        budgets=(0.20, 0.40),
        reference_budget=0.40,
        primary_budget=0.40,
        batch_targets=2,
        strategies=("random", "diversity"),
        acquisition_seeds=(101,),
        model_seeds=(17,),
        final_model_seeds=(17,),
        epochs=1,
        patience=1,
    )
    result = run_active_benchmark(trials, config=cfg)
    assert result["audit"]["no_test_target_selected"] is True
    assert result["audit"]["fold_leaks"] == []
    assert len(result["metrics"]) > 0
    assert result["metrics"]["pair_spearman"].notna().any()
    assert result["selection_log"]
