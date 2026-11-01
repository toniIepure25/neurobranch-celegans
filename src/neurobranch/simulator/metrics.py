"""Evaluation metrics for free-running neural population simulations.

These metrics deliberately separate one-step prediction from recursive rollout.
A model may have a respectable teacher-forced R2 and a useless free-running
trajectory; the report must show both.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import r2_score

from neurobranch.data.kato import MACRO_ORDER
from neurobranch.preprocessing import split_starts
from neurobranch.simulator.adapters import NeuralSimulator


def _safe_r2(truth: np.ndarray, pred: np.ndarray) -> float:
    if truth.size == 0 or not np.isfinite(pred).all():
        return float("nan")
    if np.var(truth) <= 1e-12:
        return float("nan")
    return float(r2_score(truth.ravel(), pred.ravel()))


def one_step_metrics(
    sim: NeuralSimulator,
    starts: np.ndarray | None = None,
    *,
    history: int | None = None,
) -> dict[str, Any]:
    history = int(history or getattr(sim.model, "history_len", 1))
    if starts is None:
        starts = split_starts(sim.data, "test", history=history, horizon=1)
    starts = np.asarray(starts, dtype=np.int64)
    H = np.stack([sim.data.X[t - history + 1 : t + 1] for t in starts])
    pred = np.asarray(sim.model.predict_next(H), dtype=np.float64)
    truth = sim.data.X[starts + 1]
    persistence = sim.data.X[starts]
    err = pred - truth
    per_neuron_r2 = []
    for j in range(truth.shape[1]):
        per_neuron_r2.append(_safe_r2(truth[:, j], pred[:, j]))
    probs = sim.model.behavior_proba(H)
    labels = probs.argmax(axis=1)
    return {
        "n_starts": int(len(starts)),
        "mse": float(np.mean(err**2)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": _safe_r2(truth, pred),
        "correlation": float(np.corrcoef(truth.ravel(), pred.ravel())[0, 1])
        if truth.size > 2
        else float("nan"),
        "persistence_mse": float(np.mean((persistence - truth) ** 2)),
        "persistence_r2": _safe_r2(truth, persistence),
        "delta_r2_vs_persistence": _safe_r2(truth, pred) - _safe_r2(truth, persistence),
        "per_neuron_r2_median": float(np.nanmedian(per_neuron_r2)),
        "behavior_accuracy": float(np.mean(labels == sim.data.y_macro[starts + 1])),
    }


def _lag_correlation(x: np.ndarray, lag: int) -> np.ndarray:
    """Per-neuron Pearson correlation between x[t] and x[t+lag]."""
    a = np.asarray(x[:-lag], dtype=np.float64)
    b = np.asarray(x[lag:], dtype=np.float64)
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    denom = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    out = np.full(a.shape[1], np.nan, dtype=np.float64)
    ok = denom > 1e-12
    out[ok] = (a[:, ok] * b[:, ok]).sum(axis=0) / denom[ok]
    return out


def _autocorr_error(sim: np.ndarray, real: np.ndarray, lags: tuple[int, ...] = (1, 2, 5, 10)) -> float:
    """Mean absolute per-neuron lag-correlation error.

    The two arrays are separate trajectories; correlating their concatenation
    measures cross-block similarity, not temporal autocorrelation.
    """
    if len(sim) <= max(lags) + 1 or sim.shape != real.shape:
        return float("nan")
    errs = []
    for lag in lags:
        a = _lag_correlation(sim, lag)
        b = _lag_correlation(real, lag)
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.any():
            errs.append(float(np.mean(np.abs(a[valid] - b[valid]))))
    return float(np.nanmean(errs)) if errs else float("nan")


def _cross_corr_error(sim: np.ndarray, real: np.ndarray) -> float:
    if sim.shape[1] < 2 or len(sim) < 3:
        return float("nan")
    a = np.corrcoef(sim.T)
    b = np.corrcoef(real.T)
    return float(np.mean(np.abs(a - b)))


def _covariance_error(sim: np.ndarray, real: np.ndarray) -> float:
    """Relative Frobenius error of the neuron-by-neuron covariance matrix."""
    if sim.shape[1] < 2 or len(sim) < 3:
        return float("nan")
    a = np.cov(sim, rowvar=False)
    b = np.cov(real, rowvar=False)
    denom = np.linalg.norm(b, ord="fro")
    return float(np.linalg.norm(a - b, ord="fro") / denom) if denom > 1e-12 else float("nan")


def _energy_distance(a: np.ndarray, b: np.ndarray, *, max_n: int = 400, seed: int = 0) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if len(a) > max_n:
        a = a[rng.choice(len(a), max_n, replace=False)]
    if len(b) > max_n:
        b = b[rng.choice(len(b), max_n, replace=False)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")

    def mean_dist(x: np.ndarray, y: np.ndarray) -> float:
        # Chunked Euclidean distances to avoid a large (n,n,N) tensor.
        total, count = 0.0, 0
        for i in range(0, len(x), 64):
            d = np.linalg.norm(x[i : i + 64, None, :] - y[None, :, :], axis=2)
            total += float(d.sum())
            count += d.size
        return total / max(1, count)

    return float(2 * mean_dist(a, b) - mean_dist(a, a) - mean_dist(b, b))


def distribution_metrics(sim_states: np.ndarray, real_states: np.ndarray) -> dict[str, float]:
    sim_states = np.asarray(sim_states, dtype=np.float64)
    real_states = np.asarray(real_states, dtype=np.float64)
    if sim_states.shape != real_states.shape:
        raise ValueError(f"distribution shapes differ: {sim_states.shape} vs {real_states.shape}")
    mean_err = np.abs(sim_states.mean(0) - real_states.mean(0))
    sd_err = np.abs(sim_states.std(0) - real_states.std(0))
    sim_var, real_var = sim_states.var(), real_states.var()
    return {
        "mean_abs_error": float(np.mean(mean_err)),
        "mean_sd_error": float(np.mean(sd_err)),
        "energy_distance": _energy_distance(sim_states, real_states),
        "autocorr_error": _autocorr_error(sim_states, real_states),
        "cross_corr_error": _cross_corr_error(sim_states, real_states),
        "covariance_error": _covariance_error(sim_states, real_states),
        "variance_ratio": float(sim_var / real_var) if real_var > 1e-12 else float("nan"),
    }


def rollout_metrics(
    sim: NeuralSimulator,
    starts: np.ndarray | None = None,
    *,
    horizon: int = 20,
    noise_std: float = 0.0,
    base_seed: int = 0,
    use_external_input: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate free-running rollouts from held-out initial states."""
    if starts is None:
        starts = sim.test_starts(horizon=horizon)
    starts = np.asarray(starts, dtype=np.int64)
    if len(starts) == 0:
        raise ValueError("No eligible test starts for this horizon")
    rows: list[dict[str, Any]] = []
    all_sim, all_real = [], []
    history = int(getattr(sim.model, "history_len", 1))
    for i, t in enumerate(starts):
        seed = int(base_seed + i)
        hist = sim.initial_history(int(t), history=history)
        rollout = sim.rollout(
            None,
            horizon,
            initial_history=hist,
            seed=seed,
            noise_std=noise_std,
            external_input=(
                sim.input_slice(int(t), horizon) if use_external_input else None
            ),
        )
        real = sim.data.X[int(t) : int(t) + horizon + 1]
        if not rollout.finite:
            rows.append(
                {
                    "frame": int(t),
                    "seed": seed,
                    "finite": False,
                    "mse": float("nan"),
                    "r2": float("nan"),
                    "persistence_mse": float("nan"),
                    "mse_h5": float("nan"),
                    "mse_h10": float("nan"),
                    "ood_fraction": rollout.ood_fraction,
                }
            )
            continue
        pred = rollout.states[1:]
        mse = float(np.mean((pred - real[1:]) ** 2))
        constant = np.repeat(real[:1], horizon, axis=0)
        pers = float(np.mean((constant - real[1:]) ** 2))
        h5 = min(5, horizon)
        h10 = min(10, horizon)
        pers_h5 = float(np.mean((constant[:h5] - real[1 : h5 + 1]) ** 2))
        pers_h10 = float(np.mean((constant[:h10] - real[1 : h10 + 1]) ** 2))
        row = {
            "frame": int(t),
            "seed": seed,
            "finite": True,
            "mse": mse,
            "r2": _safe_r2(real[1:], pred),
            "persistence_mse": pers,
            "persistence_h5": pers_h5,
            "persistence_h10": pers_h10,
            "mse_h5": float(np.mean((pred[: min(5, horizon)] - real[1 : min(5, horizon) + 1]) ** 2)),
            "mse_h10": float(np.mean((pred[: min(10, horizon)] - real[1 : min(10, horizon) + 1]) ** 2)),
            "ood_fraction": rollout.ood_fraction,
            "max_support_distance": float(np.nanmax(rollout.support_distance)),
            "support_q90_fraction": float(
                np.mean(rollout.support_distance <= sim.support.thresholds["q90"])
            ),
            "support_q95_fraction": float(
                np.mean(rollout.support_distance <= sim.support.thresholds["q95"])
            ),
            "support_q99_fraction": float(
                np.mean(rollout.support_distance <= sim.support.thresholds["q99"])
            ),
            "readout_supported_fraction": float(
                rollout.metadata.get("readout_supported_fraction", float("nan"))
            ),
            "behavior_accuracy": float(
                np.mean(rollout.behavior_proba.argmax(1) == sim.data.y_macro[int(t) + 1 : int(t) + horizon + 1])
            ),
        }
        rows.append(row)
        all_sim.append(pred)
        all_real.append(real[1:])

    valid = [r for r in rows if r.get("finite")]
    sim_arr = np.concatenate(all_sim, axis=0) if all_sim else np.empty((0, sim.n_neurons))
    real_arr = np.concatenate(all_real, axis=0) if all_real else np.empty((0, sim.n_neurons))
    dist = distribution_metrics(sim_arr, real_arr) if len(sim_arr) else {}
    summary: dict[str, Any] = {
        "model_kind": sim.model_kind,
        "worm_id": int(sim.data.worm_id),
        "n_starts": int(len(starts)),
        "n_valid": int(len(valid)),
        "horizon": int(horizon),
        "noise_std": float(noise_std),
        "finite_fraction": float(len(valid) / len(rows)),
        "mse": float(np.nanmean([r["mse"] for r in valid])) if valid else float("nan"),
        "r2": float(np.nanmean([r["r2"] for r in valid])) if valid else float("nan"),
        "persistence_mse": float(np.nanmean([r["persistence_mse"] for r in valid]))
        if valid
        else float("nan"),
        "persistence_baseline": "constant initial state held over the rollout horizon (not one-step persistence)",
        "delta_mse_vs_persistence": float(
            np.nanmean([r["mse"] - r["persistence_mse"] for r in valid])
        )
        if valid
        else float("nan"),
        "skill_vs_persistence": float(
            1.0
            - np.nanmean([r["mse"] for r in valid])
            / np.nanmean([r["persistence_mse"] for r in valid])
        )
        if valid and np.nanmean([r["persistence_mse"] for r in valid]) > 0
        else float("nan"),
        "mse_h5": float(np.nanmean([r["mse_h5"] for r in valid])) if valid else float("nan"),
        "mse_h10": float(np.nanmean([r["mse_h10"] for r in valid])) if valid else float("nan"),
        "ood_fraction": float(np.nanmean([r["ood_fraction"] for r in valid])) if valid else float("nan"),
        "support_q90_fraction": float(np.nanmean([r["support_q90_fraction"] for r in valid]))
        if valid
        else float("nan"),
        "support_q95_fraction": float(np.nanmean([r["support_q95_fraction"] for r in valid]))
        if valid
        else float("nan"),
        "support_q99_fraction": float(np.nanmean([r["support_q99_fraction"] for r in valid]))
        if valid
        else float("nan"),
        "readout_supported_fraction": float(
            np.nanmean([r["readout_supported_fraction"] for r in valid])
        )
        if valid
        else float("nan"),
        "behavior_accuracy": float(np.nanmean([r["behavior_accuracy"] for r in valid]))
        if valid
        else float("nan"),
        **dist,
    }
    return summary, rows


def first_outcome(initial_macro: int, labels: np.ndarray) -> int:
    """First macro state different from the initial state, else the initial."""
    for label in labels:
        if int(label) != int(initial_macro):
            return int(label)
    return int(initial_macro)


def transition_distribution(
    sim: NeuralSimulator,
    starts: np.ndarray,
    *,
    horizon: int = 20,
    n_rollouts: int = 100,
    noise_std: float = 0.0,
    base_seed: int = 0,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Aggregate simulated first-outcome distributions over initial states."""
    counts = np.zeros(4, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for t in np.asarray(starts, dtype=np.int64):
        initial = int(sim.data.y_macro[int(t)])
        history = int(getattr(sim.model, "history_len", 1))
        ensemble = sim.rollout_ensemble(
            None,
            n_rollouts,
            horizon,
            initial_history=sim.initial_history(int(t), history=history),
            initial_macro=initial,
            base_seed=base_seed,
            noise_std=noise_std,
            include_latent=False,
        )
        outcomes = []
        for rollout in ensemble.rollouts:
            out = first_outcome(initial, rollout.behavior_proba.argmax(axis=1))
            outcomes.append(out)
            counts[out] += 1
        rows.append(
            {
                "frame": int(t),
                "initial_macro": MACRO_ORDER[initial],
                "n_rollouts": n_rollouts,
                "entropy": float(_entropy(np.bincount(outcomes, minlength=4))),
                "unique_outcomes": int(len(set(outcomes))),
                "stochastic": ensemble.stochastic,
            }
        )
    total = counts.sum()
    return (counts / total if total else counts), rows


def _entropy(counts: np.ndarray) -> float:
    p = np.asarray(counts, dtype=np.float64)
    p = p / max(1.0, p.sum())
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))
