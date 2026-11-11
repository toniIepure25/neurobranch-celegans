"""Held-out-worm benchmark for the three Tier-1 generative strategies.

Models compared
---------------
* ``quadratic``: one global quadratic map.
* ``switching``: behaviour-conditioned linear maps + Markov behaviour process.
* ``manifold_walk``: kernel sampler over observed transitions (descriptive).

The benchmark asks a *forecasting* question: starting from a held-out frame,
how well does each model recover the future latent state and behaviour?  It is
not a causal or mechanistic benchmark.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset import BEHAVIOR_NAMES, WormTrajectory, transition_arrays
from .dynamics import fit_models
from .evaluate import r2_score_np
from .manifold_walk import ManifoldWalk
from .switching import fit_switching


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / np.maximum(b[mask], 1e-12))))

    return 0.5 * (kl(p, m) + kl(q, m))


def _distribution(labels: np.ndarray, n: int = len(BEHAVIOR_NAMES)) -> np.ndarray:
    return np.bincount(labels, minlength=n).astype(np.float64)


def _transition(labels: np.ndarray, n: int = len(BEHAVIOR_NAMES)) -> np.ndarray:
    matrix = np.full((n, n), 1e-9, dtype=np.float64)
    for a, b in zip(labels[:-1], labels[1:]):
        matrix[int(a), int(b)] += 1.0
    return matrix / matrix.sum(axis=1, keepdims=True)


def _true_future(test: WormTrajectory, starts: np.ndarray, horizon: int) -> np.ndarray:
    return np.stack([test.behavior[s + 1 : s + 1 + horizon] for s in starts])


def _baselines(
    test: WormTrajectory,
    train: list[WormTrajectory],
    starts: np.ndarray,
    horizon: int,
) -> dict[str, float]:
    """Persistence and marginal-frequency baselines for the same starts."""
    h = min(10, horizon)
    true_latents = np.stack([test.latent[s + 1 : s + 1 + h] for s in starts])
    constant = np.repeat(test.latent[starts][:, None, :], h, axis=1)
    latent_mse = float(np.mean((constant - true_latents) ** 2))

    true_labels = np.stack([test.behavior[s + 1 : s + 1 + h] for s in starts])
    initial = test.behavior[starts]
    accuracy = float(np.mean(np.repeat(initial[:, None], h, axis=1) == true_labels))

    counts = np.bincount(np.concatenate([t.behavior for t in train]), minlength=len(BEHAVIOR_NAMES))
    probs = (counts.astype(float) + 0.5) / (counts.sum() + 0.5 * len(BEHAVIOR_NAMES))
    nll = float(-np.mean(np.log(probs[true_labels])))
    return {
        "persistence_latent_mse_h10": latent_mse,
        "persistence_behavior_accuracy_h10": accuracy,
        "marginal_behavior_nll_h10": nll,
    }


def _score_predictions(
    test: WormTrajectory,
    starts: np.ndarray,
    horizon: int,
    latents: np.ndarray,
    labels: np.ndarray,
    *,
    interval_scale_50: float = 1.0,
    interval_scale_90: float = 1.0,
) -> dict[str, float]:
    """``latents``/``labels`` are (n_starts, seeds, horizon)."""
    true_latents = np.stack([test.latent[s + 1 : s + 1 + horizon] for s in starts])
    true_labels = _true_future(test, starts, horizon)
    mean_latents = latents.mean(axis=1)
    latent_mse = float(np.mean((mean_latents - true_latents) ** 2))
    constant = np.repeat(test.latent[starts][:, None, :], horizon, axis=1)
    baseline_mse = float(np.mean((constant - true_latents) ** 2))

    # Ensemble CRPS: a proper score for the predictive distribution, not only
    # its mean.  The pairwise term is computed on the small seed ensemble.
    abs_error = np.abs(latents - true_latents[:, None, :, :]).mean(axis=1)
    pairwise = np.abs(latents[:, :, None, :, :] - latents[:, None, :, :, :]).mean(axis=(1, 2))
    crps_by_dim = abs_error - 0.5 * pairwise
    crps = float(np.mean(crps_by_dim))

    low50, high50 = np.quantile(latents, [0.25, 0.75], axis=1)
    low90, high90 = np.quantile(latents, [0.05, 0.95], axis=1)
    median = np.median(latents, axis=1)
    low50c = median - interval_scale_50 * (median - low50)
    high50c = median + interval_scale_50 * (high50 - median)
    low90c = median - interval_scale_90 * (median - low90)
    high90c = median + interval_scale_90 * (high90 - median)
    coverage50 = float(np.mean((true_latents >= low50) & (true_latents <= high50)))
    coverage90 = float(np.mean((true_latents >= low90) & (true_latents <= high90)))
    coverage50c = float(np.mean((true_latents >= low50c) & (true_latents <= high50c)))
    coverage90c = float(np.mean((true_latents >= low90c) & (true_latents <= high90c)))

    # Ensemble categorical probabilities with Laplace smoothing.  For the
    # switching model this is an approximation to its exact behaviour
    # probabilities; the benchmark uses the same read-out for every model.
    n_starts, n_seeds, n_steps = labels.shape
    probs = np.full((n_starts, n_steps, len(BEHAVIOR_NAMES)), 0.5, dtype=np.float64)
    for i in range(n_starts):
        for t in range(n_steps):
            probs[i, t] += np.bincount(labels[i, :, t], minlength=len(BEHAVIOR_NAMES))
    probs /= probs.sum(axis=-1, keepdims=True)
    true_onehot = np.eye(len(BEHAVIOR_NAMES))[true_labels]
    nll = float(
        -np.mean(np.log(probs[np.arange(n_starts)[:, None], np.arange(n_steps)[None, :], true_labels]))
    )
    brier = float(np.mean(np.sum((probs - true_onehot) ** 2, axis=-1)))

    acc = float(np.mean(labels == np.repeat(true_labels[:, None, :], labels.shape[1], axis=1)))
    generated = labels.reshape(-1)
    return {
        "latent_mse": latent_mse,
        "latent_mse_skill_vs_constant": float(1.0 - latent_mse / max(baseline_mse, 1e-12)),
        "latent_crps": crps,
        "interval_coverage_50": coverage50,
        "interval_coverage_90": coverage90,
        "interval_coverage_50_calibrated": coverage50c,
        "interval_coverage_90_calibrated": coverage90c,
        "interval_scale_50": float(interval_scale_50),
        "interval_scale_90": float(interval_scale_90),
        "behavior_accuracy": acc,
        "behavior_nll": nll,
        "behavior_brier": brier,
        "behavior_distribution_js": js_divergence(
            _distribution(generated), _distribution(true_labels.reshape(-1))
        ),
        "behavior_transition_js": js_divergence(_transition(generated), _transition(true_labels.reshape(-1))),
    }


@dataclass(frozen=True)
class ModelBenchmark:
    model: str
    folds: tuple[dict[str, object], ...]

    def aggregate(self) -> dict[str, object]:
        keys = [
            "one_step_r2",
            "latent_mse_h10",
            "latent_mse_skill_h10",
            "latent_crps_h10",
            "interval_coverage_50_h10",
            "interval_coverage_90_h10",
            "interval_coverage_50_h10_calibrated",
            "interval_coverage_90_h10_calibrated",
            "interval_scale_50_h10",
            "interval_scale_90_h10",
            "behavior_accuracy_h1",
            "behavior_accuracy_h10",
            "behavior_accuracy_h25",
            "behavior_nll_h10",
            "behavior_brier_h10",
            "behavior_transition_js",
            "behavior_distribution_js",
            "persistence_latent_mse_h10",
            "persistence_behavior_accuracy_h10",
            "marginal_behavior_nll_h10",
        ]
        out: dict[str, object] = {"model": self.model, "n_folds": len(self.folds)}
        for key in keys:
            values = [float(f[key]) for f in self.folds if key in f and np.isfinite(float(f[key]))]
            mean = float(np.mean(values)) if values else float("nan")
            std = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
            half = float(1.96 * std / np.sqrt(len(values))) if len(values) > 1 else float("nan")
            out[key] = {
                "mean": mean,
                "std": std,
                "ci95": [mean - half, mean + half] if len(values) > 1 else [float("nan")] * 2,
                "values": values,
            }
        for key in ("latent_mse_h10", "behavior_accuracy_h10"):
            values = [float(f[key]) for f in self.folds if key in f]
            if values:
                out[f"worst_worm_{key}"] = float(max(values))
        return out


def _forecast(
    kind: str,
    test: WormTrajectory,
    starts: np.ndarray,
    horizon: int,
    *,
    quadratic,
    switching,
    walk: ManifoldWalk,
    seeds: int,
    quadratic_noise: float = 0.0,
    switching_noise: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(starts)
    latents = np.empty((n, seeds, horizon, 3), dtype=np.float64)
    labels = np.empty((n, seeds, horizon), dtype=np.int64)
    for si, start in enumerate(starts):
        y0 = test.latent[start]
        b0 = int(test.behavior[start])
        for seed in range(seeds):
            if kind == "quadratic":
                traj = quadratic.dynamics.rollout(y0, n_steps=horizon, seed=seed, noise=quadratic_noise)
                lab = quadratic.behavior.predict(traj)
            elif kind == "switching":
                traj, lab, _ = switching.simulate(y0, b0, n_steps=horizon, noise=switching_noise, seed=seed)
            elif kind == "manifold_walk":
                traj, lab = walk.rollout_with_behavior(y0, n_steps=horizon, seed=seed, temperature=0.6)
            else:
                raise ValueError(kind)
            latents[si, seed] = traj
            labels[si, seed] = lab
    return latents, labels


def _fit_bundle(train: list[WormTrajectory]):
    """Fit the three forecasters and their train-residual noise scales."""
    quadratic = fit_models(train)
    switching = fit_switching(train)
    walk = ManifoldWalk(n_neighbors=20, noise=0.0).fit(train)
    train_x, train_y, _ = transition_arrays(train)
    quadratic_noise = float(np.std(quadratic.dynamics.predict(train_x) - train_y))
    switching_behavior = np.concatenate([t.behavior[:-1] for t in train])
    switching_pred = np.stack(
        [switching.dynamics.predict(train_x[i], int(switching_behavior[i])) for i in range(len(train_x))]
    )
    switching_noise = float(np.std(switching_pred - train_y))
    return quadratic, switching, walk, quadratic_noise, switching_noise


def _calibration_scales(
    train: list[WormTrajectory],
    *,
    horizon: int,
    stride: int,
    seeds: int,
    target_50: float = 0.50,
    target_90: float = 0.90,
) -> dict[str, dict[str, float]]:
    """Nested conformal scaling of ensemble intervals, without touching test data.

    For each outer training worm, fit on the remaining training worms and pool
    the normalised absolute residuals of the ensemble median.  The 90th (and
    50th) percentile becomes a scalar widening factor applied to the held-out
    test intervals.  This is a marginal calibration diagnostic; it cannot fix
    worm-specific miscalibration and it does not turn the intervals into
    posterior credible intervals.
    """
    if len(train) < 3:
        return {
            kind: {"scale_50": 1.0, "scale_90": 1.0, "n_calibration_worms": 0}
            for kind in ("quadratic", "switching", "manifold_walk")
        }
    h = min(10, horizon)
    ratios: dict[str, list[np.ndarray]] = {
        "quadratic": [],
        "switching": [],
        "manifold_walk": [],
    }
    for calibration in train:
        calibration_train = [t for t in train if t.worm_id != calibration.worm_id]
        if len(calibration_train) < 2:
            continue
        quadratic, switching, walk, q_noise, s_noise = _fit_bundle(calibration_train)
        starts = np.arange(0, max(1, len(calibration.latent) - h - 1), stride, dtype=np.int64)
        true = np.stack([calibration.latent[s + 1 : s + 1 + h] for s in starts])
        for kind in ratios:
            latents, _ = _forecast(
                kind,
                calibration,
                starts,
                h,
                quadratic=quadratic,
                switching=switching,
                walk=walk,
                seeds=seeds,
                quadratic_noise=q_noise,
                switching_noise=s_noise,
            )
            median = np.median(latents, axis=1)
            q25, q75 = np.quantile(latents, [0.25, 0.75], axis=1)
            q05, q95 = np.quantile(latents, [0.05, 0.95], axis=1)
            half50 = np.maximum.reduce([q75 - median, median - q25, np.full_like(median, 1e-12)])
            half90 = np.maximum.reduce([q95 - median, median - q05, np.full_like(median, 1e-12)])
            ratios[kind].extend(
                [
                    np.abs(true - median) / half50,
                    np.abs(true - median) / half90,
                ]
            )
    out: dict[str, dict[str, float]] = {}
    for kind, values in ratios.items():
        if not values:
            out[kind] = {"scale_50": 1.0, "scale_90": 1.0, "n_calibration_worms": 0}
            continue
        # values[0::2] are 50% ratios, values[1::2] are 90% ratios.
        r50 = np.concatenate(values[0::2]) if len(values) > 1 else values[0]
        r90 = np.concatenate(values[1::2]) if len(values) > 1 else values[-1]
        out[kind] = {
            "scale_50": float(max(1.0, np.quantile(r50, target_50))),
            "scale_90": float(max(1.0, np.quantile(r90, target_90))),
            "n_calibration_worms": int(len(train)),
        }
    return out


def benchmark_models(
    trajectories: list[WormTrajectory],
    *,
    horizon: int = 25,
    stride: int = 75,
    seeds: int = 16,
) -> dict[str, object]:
    if horizon < 1 or seeds < 1:
        raise ValueError("horizon and seeds must be positive")
    models = ("quadratic", "switching", "manifold_walk")
    folds_by_model: dict[str, list[dict[str, object]]] = {m: [] for m in models}
    curves_by_model: dict[str, dict[int, list[float]]] = {
        m: {h: [] for h in (1, 5, 10, 25) if h <= horizon} for m in models
    }

    for test in trajectories:
        train = [t for t in trajectories if t.worm_id != test.worm_id]
        quadratic, switching, walk, quadratic_noise, switching_noise = _fit_bundle(train)
        scales = _calibration_scales(
            train,
            horizon=horizon,
            stride=stride,
            seeds=seeds,
        )
        starts = np.arange(0, max(1, len(test.latent) - horizon - 1), stride, dtype=np.int64)

        # One-step deterministic diagnostics.
        one_step = quadratic.dynamics.predict(test.latent[:-1])
        one_step_r2 = r2_score_np(test.latent[1:], one_step)
        baselines = _baselines(test, train, starts, horizon)

        for kind in models:
            latents, labels = _forecast(
                kind,
                test,
                starts,
                horizon,
                quadratic=quadratic,
                switching=switching,
                walk=walk,
                seeds=seeds,
                quadratic_noise=quadratic_noise,
                switching_noise=switching_noise,
            )
            metrics_h = {
                h: _score_predictions(
                    test,
                    starts,
                    h,
                    latents[:, :, :h],
                    labels[:, :, :h],
                    interval_scale_50=scales[kind]["scale_50"],
                    interval_scale_90=scales[kind]["scale_90"],
                )
                for h in curves_by_model[kind]
            }
            for h, metrics in metrics_h.items():
                curves_by_model[kind][h].append(float(metrics["latent_mse"]))
            h1 = 1
            h10 = min(10, horizon)
            h25 = min(25, horizon)
            metrics1 = metrics_h.get(h1) or _score_predictions(
                test, starts, h1, latents[:, :, :h1], labels[:, :, :h1]
            )
            metrics10 = metrics_h.get(h10) or _score_predictions(
                test, starts, h10, latents[:, :, :h10], labels[:, :, :h10]
            )
            metrics25 = metrics_h.get(h25) or _score_predictions(
                test, starts, h25, latents[:, :, :h25], labels[:, :, :h25]
            )
            folds_by_model[kind].append(
                {
                    "worm_id": test.worm_id,
                    "n_starts": int(len(starts)),
                    "seed": "multi" if kind != "quadratic" else 0,
                    "one_step_r2": float(one_step_r2) if kind == "quadratic" else float("nan"),
                    "latent_mse_h10": metrics10["latent_mse"],
                    "latent_mse_skill_h10": metrics10["latent_mse_skill_vs_constant"],
                    "latent_crps_h10": metrics10["latent_crps"],
                    "interval_coverage_50_h10": metrics10["interval_coverage_50"],
                    "interval_coverage_90_h10": metrics10["interval_coverage_90"],
                    "interval_coverage_50_h10_calibrated": metrics10["interval_coverage_50_calibrated"],
                    "interval_coverage_90_h10_calibrated": metrics10["interval_coverage_90_calibrated"],
                    "interval_scale_50_h10": metrics10["interval_scale_50"],
                    "interval_scale_90_h10": metrics10["interval_scale_90"],
                    "n_calibration_worms": scales[kind]["n_calibration_worms"],
                    "behavior_accuracy_h1": metrics1["behavior_accuracy"],
                    "behavior_accuracy_h10": metrics10["behavior_accuracy"],
                    "behavior_accuracy_h25": metrics25["behavior_accuracy"],
                    "behavior_nll_h10": metrics10["behavior_nll"],
                    "behavior_brier_h10": metrics10["behavior_brier"],
                    "behavior_transition_js": metrics25["behavior_transition_js"],
                    "behavior_distribution_js": metrics25["behavior_distribution_js"],
                    **baselines,
                }
            )

    result = {
        "format": "vienna-tier1-benchmark-v3",
        "horizon": horizon,
        "stride": stride,
        "seeds": seeds,
        "claim": (
            "Held-out-worm open-loop forecast comparison. Not a mechanistic or causal test. "
            "Intervals are ensemble quantiles, not posterior credible intervals. Calibrated "
            "coverage uses nested training-worm conformal scaling; it is marginal, not "
            "worm-conditional, and does not use the test worm."
        ),
        "calibration": {
            "method": "nested leave-one-training-worm-out percentile scaling",
            "target_50": 0.50,
            "target_90": 0.90,
            "test_data_used": False,
            "scope": "marginal over starts and dimensions; not per-worm coverage",
        },
        "models": [ModelBenchmark(model, tuple(folds_by_model[model])).aggregate() for model in models],
        "horizon_curve": {
            model: {str(h): float(np.mean(values)) if values else float("nan") for h, values in curve.items()}
            for model, curve in curves_by_model.items()
        },
    }
    return result


def save_benchmark(result: dict[str, object], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .dataset import load_trajectories

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--stride", type=int, default=75)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "reports" / "vienna_workbench" / "tier1_benchmark.json",
    )
    args = parser.parse_args(argv)
    result = benchmark_models(load_trajectories(), horizon=args.horizon, stride=args.stride, seeds=args.seeds)
    path = save_benchmark(result, args.out)
    print(json.dumps(result, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
