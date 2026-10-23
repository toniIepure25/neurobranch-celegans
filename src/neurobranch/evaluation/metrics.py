"""Evaluation metrics and uncertainty for dynamics models.

The test split is never used for fitting, early stopping, PCA, scaling, or
neighbour retrieval.  Multi-step metrics use open-loop rollouts initialized
only from observed test frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from neurobranch.data.kato import MACRO_ORDER
from neurobranch.preprocessing import WormData, split_starts


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def bootstrap_ci(
    values: np.ndarray,
    *,
    n_boot: int = 500,
    statistic=np.mean,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.array(
        [statistic(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    )
    return (
        float(statistic(values)),
        float(np.quantile(boots, alpha / 2)),
        float(np.quantile(boots, 1 - alpha / 2)),
    )


def _first_transition_outcome(labels: np.ndarray, start: int, steps: int) -> str:
    current = MACRO_ORDER[int(labels[start])]
    for j in range(start + 1, min(len(labels), start + steps + 1)):
        nxt = MACRO_ORDER[int(labels[j])]
        if nxt != current:
            return nxt
    return current


def observed_transition_distribution(
    data: WormData,
    starts: np.ndarray,
    *,
    steps: int,
) -> dict[str, float]:
    counts = {s: 0 for s in MACRO_ORDER}
    for t in starts:
        counts[_first_transition_outcome(data.y_macro, int(t), steps)] += 1
    n = max(1, sum(counts.values()))
    return {s: counts[s] / n for s in MACRO_ORDER}


def model_transition_distribution(
    model: Any,
    data: WormData,
    starts: np.ndarray,
    *,
    steps: int,
    seeds: tuple[int, ...] = (0,),
    noise_std: float = 0.0,
) -> dict[str, float]:
    counts = {s: 0 for s in MACRO_ORDER}
    for seed in seeds:
        for t in starts:
            x0 = data.X[int(t)]
            seq = model.rollout(x0, steps, seed=seed, noise_std=noise_std)
            probs = model.behavior_proba(seq[:-1])
            labels = probs.argmax(axis=1)
            current = MACRO_ORDER[data.y_macro[int(t)]]
            outcome = current
            for lab in labels:
                nxt = MACRO_ORDER[int(lab)]
                if nxt != current:
                    outcome = nxt
                    break
            counts[outcome] += 1
    n = max(1, sum(counts.values()))
    return {s: counts[s] / n for s in MACRO_ORDER}


def distribution_distance(p: dict[str, float], q: dict[str, float]) -> dict[str, float]:
    a = np.array([p.get(s, 0.0) for s in MACRO_ORDER])
    b = np.array([q.get(s, 0.0) for s in MACRO_ORDER])
    tv = 0.5 * np.abs(a - b).sum()
    eps = 1e-9
    kl = float(np.sum(a * np.log((a + eps) / (b + eps))))
    return {"total_variation": float(tv), "kl_p_vs_q": kl}


@dataclass
class RolloutDiagnostics:
    mse_by_horizon: dict[int, float]
    correlation_by_horizon: dict[int, float]
    marginal_mean_error: float
    marginal_std_ratio: float
    outside_train_radius_fraction: float | None


def rollout_diagnostics(
    model: Any,
    data: WormData,
    starts: np.ndarray,
    *,
    horizons: list[int],
    max_starts: int = 200,
    seed: int = 0,
) -> tuple[RolloutDiagnostics, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    max_h = max(horizons)
    starts = np.asarray(starts)
    starts = starts[starts + max_h < data.n_frames]
    if len(starts) > max_starts:
        starts = rng.choice(starts, size=max_starts, replace=False)
    sims = np.stack([model.rollout(data.X[int(t)], max_h, seed=seed) for t in starts])
    truth = np.stack([data.X[int(t) : int(t) + max_h + 1] for t in starts])
    mse, corr = {}, {}
    for h in horizons:
        a, b = truth[:, h], sims[:, h]
        mse[h] = float(np.mean((a - b) ** 2))
        # Flatten all neurons and starts for a marginal correlation.
        corr[h] = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    mean_err = float(np.abs(sims[:, 1:].mean(axis=(0, 1)) - truth[:, 1:].mean(axis=(0, 1))).mean())
    std_ratio = float(sims[:, 1:].std() / max(truth[:, 1:].std(), 1e-8))
    radius_frac = None
    if hasattr(model, "ood_score"):
        train_X = data.X[data.masks["train"]]
        base = model.ood_score(train_X)
        thresh = float(np.quantile(base, 0.99))
        radius_frac = float(np.mean(model.ood_score(sims[:, -1]) > thresh))
    return (
        RolloutDiagnostics(
            mse_by_horizon=mse,
            correlation_by_horizon=corr,
            marginal_mean_error=mean_err,
            marginal_std_ratio=std_ratio,
            outside_train_radius_fraction=radius_frac,
        ),
        sims,
        truth,
    )


def evaluate_model(
    model: Any,
    data: WormData,
    *,
    model_kind: str,
    seed: int,
    horizons: tuple[int, ...] = (1, 3, 5, 10),
    transition_horizon: int = 20,
    transition_starts: int = 80,
    rollout_starts: int = 200,
    bootstrap_samples: int = 400,
) -> dict[str, Any]:
    history = getattr(model, "history_len", 1)
    starts = split_starts(data, "test", history=history, horizon=1)
    if len(starts) == 0:
        raise ValueError(f"No test starts for worm {data.worm_id}")
    H = np.stack([data.X[t - history + 1 : t + 1] for t in starts])
    y_next = data.y_macro[starts + 1]
    pred = model.predict_next(H)
    truth = data.X[starts + 1]
    probs = model.behavior_proba(H)
    pred_y = probs.argmax(axis=1)

    per_start_mse = ((pred - truth) ** 2).mean(axis=1)
    mse_mean, mse_lo, mse_hi = bootstrap_ci(per_start_mse, n_boot=bootstrap_samples, seed=seed)
    correct = (pred_y == y_next).astype(float)
    acc_mean, acc_lo, acc_hi = bootstrap_ci(correct, n_boot=bootstrap_samples, seed=seed)

    # Strong persistence baseline for behaviour (and dynamics).
    persist_acc = float(np.mean(data.y_macro[starts] == y_next))
    persist_probs = np.zeros((len(starts), 4))
    persist_probs[np.arange(len(starts)), data.y_macro[starts]] = 1.0
    brier = float(np.mean(np.sum((probs - persist_probs) ** 2, axis=1)))
    persistence_mse = float(np.mean((data.X[starts] - truth) ** 2))

    diag, _, _ = rollout_diagnostics(
        model, data, starts, horizons=list(horizons), max_starts=rollout_starts, seed=seed
    )

    # Transition behaviour from held-out reverse states.
    rev_all = starts[(data.y_macro[starts] == 1) & (starts + transition_horizon < data.n_frames)]
    rng = np.random.default_rng(seed + 1)
    if len(rev_all) > transition_starts:
        rev_all = rng.choice(rev_all, size=transition_starts, replace=False)
    observed = observed_transition_distribution(data, rev_all, steps=transition_horizon)
    pred_dist = model_transition_distribution(
        model, data, rev_all, steps=transition_horizon, seeds=(seed, seed + 101, seed + 202)
    )
    dist = distribution_distance(observed, pred_dist)

    # Persistence by construction predicts no transition; compare against observed.
    no_transition = {s: 0.0 for s in MACRO_ORDER}
    no_transition["REV"] = 1.0

    result = {
        "worm_id": data.worm_id,
        "model": model_kind,
        "seed": seed,
        "n_neurons": data.n_neurons,
        "n_test_starts": int(len(starts)),
        "one_step_mse": mse_mean,
        "one_step_mse_ci": [mse_lo, mse_hi],
        "one_step_r2": r2_score(truth, pred),
        "persistence_mse": persistence_mse,
        "behavior_accuracy": acc_mean,
        "behavior_accuracy_ci": [acc_lo, acc_hi],
        "behavior_persistence_accuracy": persist_acc,
        "behavior_brier_vs_persistence": brier,
        "rollout_mse": {str(k): v for k, v in diag.mse_by_horizon.items()},
        "rollout_corr": {str(k): v for k, v in diag.correlation_by_horizon.items()},
        "marginal_mean_error": diag.marginal_mean_error,
        "marginal_std_ratio": diag.marginal_std_ratio,
        "outside_train_radius_fraction": diag.outside_train_radius_fraction,
        "observed_transition_distribution": observed,
        "predicted_transition_distribution": pred_dist,
        "transition_total_variation": dist["total_variation"],
        "transition_kl": dist["kl_p_vs_q"],
        "transition_persistence_tv": distribution_distance(observed, no_transition)["total_variation"],
        "transition_n_starts": int(len(rev_all)),
        "latent_dim": getattr(model, "latent_dim", None),
    }
    return result


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate seed/worm results; worm is the independent unit for CIs."""
    if not rows:
        return {}
    keys = [
        "one_step_mse",
        "one_step_r2",
        "behavior_accuracy",
        "behavior_persistence_accuracy",
        "marginal_mean_error",
        "marginal_std_ratio",
        "transition_total_variation",
        "transition_persistence_tv",
    ]
    out: dict[str, Any] = {"n_rows": len(rows), "models": sorted({r["model"] for r in rows})}
    for k in keys:
        vals = np.array([r[k] for r in rows if r.get(k) is not None], dtype=float)
        out[k] = {
            "mean": float(np.nanmean(vals)),
            "sd": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals)),
        }
    per_worm = {}
    for r in rows:
        per_worm.setdefault(str(r["worm_id"]), []).append(r["behavior_accuracy"])
    out["per_worm_behavior_accuracy"] = {k: float(np.mean(v)) for k, v in per_worm.items()}
    return out
