"""Simulator-side trajectory-outcome distributions and branching comparisons."""

from __future__ import annotations

from typing import Any

import numpy as np

from neurobranch.branching import LocalOutcome, local_outcome_diversity
from neurobranch.branching.stats import null_branching_test
from neurobranch.data.kato import MACRO_ORDER
from neurobranch.preprocessing import WormData, split_starts
from neurobranch.simulator.adapters import NeuralSimulator
from neurobranch.simulator.latent import PCAEncoder


def reversal_events(data: WormData, *, horizon: int = 20) -> np.ndarray:
    """First frame of each test-set REV run with a complete held-out future."""
    rev = data.y_macro == MACRO_ORDER.index("REV")
    starts = []
    for t in split_starts(data, "test", history=1, horizon=1):
        t = int(t)
        if not rev[t] or t + horizon >= data.n_frames:
            continue
        if t > 0 and rev[t - 1]:
            continue
        if data.masks["test"][t : t + horizon + 1].all():
            starts.append(t)
    return np.asarray(starts, dtype=np.int64)


def reversal_states(data: WormData, *, horizon: int = 20) -> np.ndarray:
    """Non-overlapping held-out REV states, sampled through each reversal run.

    Onsets alone are often degenerate (the worm is committed to turning at the
    first reversal frame). Sampling every ``horizon`` frames inside the run
    tests the conditional distributions where futures actually diverge, without
    reusing overlapping outcome windows.
    """
    events = reversal_events(data, horizon=horizon)
    rev = data.y_macro == MACRO_ORDER.index("REV")
    chosen: list[int] = []
    for onset in events:
        t = int(onset)
        while t + horizon < data.n_frames and rev[t] and data.masks["test"][t : t + horizon + 1].all():
            chosen.append(t)
            t += horizon
    return np.asarray(chosen, dtype=np.int64)


def observed_local(
    data: WormData,
    encoder: PCAEncoder,
    *,
    k: int = 40,
    horizon: int = 20,
) -> tuple[dict[int, LocalOutcome], np.ndarray]:
    Z = encoder.transform(data.X)
    candidates = split_starts(data, "test", history=1, horizon=1)
    candidates = np.asarray(
        [int(t) for t in candidates if data.masks["test"][int(t) : int(t) + horizon + 1].all()],
        dtype=np.int64,
    )
    candidates = candidates[data.y_macro[candidates] == MACRO_ORDER.index("REV")]
    results = local_outcome_diversity(
        Z,
        data,
        candidates,
        k=k,
        horizon=horizon,
        exclude_temporal_radius=12,
    )
    return {r.frame: r for r in results}, candidates


def outcome_counts(labels: np.ndarray, initial: int) -> np.ndarray:
    counts = np.zeros(4, dtype=np.float64)
    for label in labels:
        if int(label) != int(initial):
            counts[int(label)] += 1.0
            break
    else:
        counts[int(initial)] += 1.0
    return counts


def simulated_local(
    sim: NeuralSimulator,
    frame: int,
    *,
    horizon: int = 20,
    n_rollouts: int = 100,
    noise_std: float = 0.05,
    base_seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    initial = int(sim.data.y_macro[int(frame)])
    ensemble = sim.rollout_ensemble(
        sim.data.X[int(frame)],
        n_rollouts,
        horizon,
        base_seed=base_seed,
        noise_std=noise_std,
        include_latent=False,
    )
    counts = np.zeros(4, dtype=np.float64)
    for rollout in ensemble.rollouts:
        counts += outcome_counts(rollout.behavior_proba.argmax(axis=1), initial)
    total = counts.sum()
    return counts / max(1.0, total), {
        "stochastic": ensemble.stochastic,
        "ood_fraction": ensemble.ood_fraction,
        "n_rollouts": int(n_rollouts),
    }


def persistence_local(
    sim: NeuralSimulator,
    frame: int,
    *,
    horizon: int = 20,
    n_rollouts: int = 100,
    base_seed: int = 0,
) -> np.ndarray:
    initial = int(sim.data.y_macro[int(frame)])
    rollout = sim.rollout(sim.data.X[int(frame)], horizon, seed=base_seed)
    counts = outcome_counts(rollout.behavior_proba.argmax(axis=1), initial)
    return counts / max(1.0, counts.sum())


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def js(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    out = 0.0
    for a in (p, q):
        mask = a > 0
        out += 0.5 * float(np.sum(a[mask] * np.log2(a[mask] / m[mask])))
    return out


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def observed_null(
    data: WormData,
    encoder: PCAEncoder,
    candidates: np.ndarray,
    *,
    horizon: int = 20,
    n_permutations: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    Z = encoder.transform(data.X)
    # Use the actual first-outcome definition, not the modal label.
    from neurobranch.branching import outcome_sequence

    outcomes = outcome_sequence(data, candidates, horizon=horizon)
    return null_branching_test(
        Z,
        candidates,
        outcomes,
        k=40,
        n_permutations=n_permutations,
        method="block",
        block_size=30,
        seed=seed,
    )


def branch_summary(
    data: WormData,
    encoder: PCAEncoder,
    selected: NeuralSimulator,
    persistence: NeuralSimulator,
    *,
    horizon: int = 20,
    n_rollouts: int = 100,
    noise_std: float = 0.05,
    seed: int = 17,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    states = reversal_states(data, horizon=horizon)
    local, candidates = observed_local(data, encoder, horizon=horizon)
    if len(candidates) == 0:
        raise ValueError("No REV candidate frames")
    marginal_counts = np.zeros(4, dtype=np.float64)
    from neurobranch.branching import outcome_sequence

    outcomes = outcome_sequence(data, candidates, horizon=horizon)
    for s in outcomes:
        marginal_counts[MACRO_ORDER.index(str(s))] += 1
    marginal = marginal_counts / marginal_counts.sum()
    rows = []
    for t in states:
        obs = local.get(int(t))
        if obs is None:
            continue
        p_obs = np.asarray([obs.probabilities[s] for s in MACRO_ORDER], dtype=np.float64)
        p_sim, sim_meta = simulated_local(
            selected,
            int(t),
            horizon=horizon,
            n_rollouts=n_rollouts,
            noise_std=noise_std,
            base_seed=seed * 1000 + int(t),
        )
        p_pers = persistence_local(persistence, int(t), horizon=horizon, base_seed=seed)
        rows.append(
            {
                "worm_id": int(data.worm_id),
                "frame": int(t),
                "is_onset": bool(int(t) == 0 or data.y_macro[int(t) - 1] != MACRO_ORDER.index("REV")),
                "observed_entropy": entropy(p_obs),
                "simulated_entropy": entropy(p_sim),
                "persistence_entropy": entropy(p_pers),
                "marginal_entropy": entropy(marginal),
                "tv_sim_observed": tv(p_sim, p_obs),
                "tv_persistence_observed": tv(p_pers, p_obs),
                "tv_marginal_observed": tv(marginal, p_obs),
                "js_sim_observed": js(p_sim, p_obs),
                "stochastic": sim_meta["stochastic"],
                "ood_fraction": sim_meta["ood_fraction"],
                "observed_distribution": ";".join(f"{s}:{p_obs[i]:.4f}" for i, s in enumerate(MACRO_ORDER)),
                "simulated_distribution": ";".join(f"{s}:{p_sim[i]:.4f}" for i, s in enumerate(MACRO_ORDER)),
                "observed_n": int(obs.n_neighbors),
                "n_rollouts": int(n_rollouts),
            }
        )
    if not rows:
        raise ValueError("No evaluable REV events")
    summary = {
        "worm_id": int(data.worm_id),
        "n_events": len(rows),
        "n_states": len(rows),
        "n_observed_candidates": int(len(candidates)),
        "mean_tv_sim": float(np.mean([r["tv_sim_observed"] for r in rows])),
        "mean_tv_persistence": float(np.mean([r["tv_persistence_observed"] for r in rows])),
        "mean_tv_marginal": float(np.mean([r["tv_marginal_observed"] for r in rows])),
        "mean_entropy_observed": float(np.mean([r["observed_entropy"] for r in rows])),
        "mean_entropy_simulated": float(np.mean([r["simulated_entropy"] for r in rows])),
        "mean_entropy_persistence": float(np.mean([r["persistence_entropy"] for r in rows])),
        "mean_entropy_marginal": float(np.mean([r["marginal_entropy"] for r in rows])),
        "stochastic_fraction": float(np.mean([r["stochastic"] for r in rows])),
        "mean_ood_fraction": float(np.mean([r["ood_fraction"] for r in rows])),
        "marginal": {s: float(marginal[i]) for i, s in enumerate(MACRO_ORDER)},
    }
    return rows, summary
