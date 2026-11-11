"""Tests for the NC-MCM assumption audit on synthetic chains."""

from __future__ import annotations

import numpy as np

from neurobranch.bundlenet_sim.causal_audit import (
    behavior_transport_test,
    markov_order_forecast,
    markov_order_forecast_all,
    markov_order_test,
    state_behavior_information,
    stationarity_test,
)
from neurobranch.bundlenet_sim.ncmcm import NCMCMWorm


def _worm(seq: np.ndarray, behavior: np.ndarray | None = None, worm_id: int = 0) -> NCMCMWorm:
    n = len(seq)
    clusters = np.zeros((n, 20), dtype=np.int64)
    clusters[:, :] = seq[:, None]
    if behavior is None:
        behavior = (seq - 1) % 8
    return NCMCMWorm(
        worm_id=worm_id,
        x=np.zeros((n, 3)),
        clusters=clusters,
        behavior=behavior.astype(np.int64),
        behavior_names=("fwd",),
        p_values=np.ones(20),
        pairwise_probabilities=np.zeros((n, 28)),
    )


def _markov_sequence(n: int, k: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(k), size=k)
    seq = np.empty(n, dtype=np.int64)
    seq[0] = 1
    for t in range(1, n):
        seq[t] = rng.choice(k, p=P[seq[t - 1] - 1]) + 1
    return seq


def test_first_order_chain_is_not_rejected() -> None:
    seq = _markov_sequence(3000, seed=1)
    result = markov_order_test(_worm(seq), 6, n_boot=200, seed=2)
    assert result["verdict"] in {"pass", "insufficient_data"}
    assert result["lr"] >= 0.0


def test_second_order_chain_is_rejected() -> None:
    k = 4
    seq = np.empty(2000, dtype=np.int64)
    seq[:2] = [1, 2]
    for t in range(2, len(seq)):
        seq[t] = ((seq[t - 1] + seq[t - 2]) % k) + 1
    result = markov_order_test(_worm(seq), k, n_boot=200, seed=4)
    assert result["verdict"] in {"fail", "insufficient_data"}


def test_second_order_improves_temporal_holdout_nll() -> None:
    k = 4
    seq = np.empty(1200, dtype=np.int64)
    seq[:2] = [1, 2]
    for t in range(2, len(seq)):
        seq[t] = ((seq[t - 1] + seq[t - 2]) % k) + 1
    worm = _worm(seq)
    result = markov_order_forecast(worm, k, n_boot=100, seed=5)
    assert result["verdict"] == "supports_order2"
    assert result["delta_nll_order1_minus_order2"] > 0
    assert result["delta_ci95"][0] > 0


def test_order_forecast_aggregate_has_uncertainty() -> None:
    worms = [_worm(_markov_sequence(600, seed=30 + i), worm_id=i) for i in range(3)]
    result = markov_order_forecast_all(worms, k=6, n_boot=100, seed=9)
    assert result["n_worms"] == 3
    assert result["verdict"] in {"supports_order2", "no_improvement", "order1_better"}
    low, high = result["delta_ci95"]
    assert np.isfinite(low) and np.isfinite(high) and low <= high


def test_stationary_chain_not_rejected() -> None:
    seq = _markov_sequence(4000, seed=5)
    result = stationarity_test(_worm(seq), 6, n_boot=100, seed=6)
    assert result["verdict"] in {"pass", "warn"}


def test_regime_change_is_detected() -> None:
    rng = np.random.default_rng(7)
    k = 6
    persistent = np.eye(k) * 0.8 + 0.2 / k
    cyclic = np.roll(np.eye(k), 1, axis=1) * 0.8 + 0.2 / k

    def draw(P: np.ndarray, n: int) -> np.ndarray:
        seq = np.empty(n, dtype=np.int64)
        seq[0] = 1
        for t in range(1, n):
            seq[t] = rng.choice(k, p=P[seq[t - 1] - 1]) + 1
        return seq

    result = stationarity_test(
        _worm(np.concatenate([draw(persistent, 2000), draw(cyclic, 2000)])),
        k,
        n_boot=100,
        seed=8,
    )
    assert result["verdict"] in {"fail", "warn"}


def test_state_behavior_information_detects_dependence() -> None:
    seq = _markov_sequence(1000, seed=9)
    behavior = (seq - 1) % 8
    result = state_behavior_information(_worm(seq, behavior), 6, n_perm=100, seed=10)
    assert result["verdict"] == "pass"
    assert result["mutual_information"] > 0.0


def test_behavior_transport_runs() -> None:
    worms = []
    for i in range(5):
        seq = _markov_sequence(800, seed=20 + i)
        worms.append(_worm(seq, (seq - 1) % 8, worm_id=i))
    result = behavior_transport_test(worms, n_boot=50, seed=11)
    assert result["verdict"] in {"pass", "warn", "insufficient_data"}
    assert "heldout_js" in result
