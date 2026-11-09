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

from .dataset import BEHAVIOR_NAMES, WormTrajectory
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


def _score_predictions(
    test: WormTrajectory,
    starts: np.ndarray,
    horizon: int,
    latents: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """``latents``/``labels`` are (n_starts, seeds, horizon)."""
    true_latents = np.stack([test.latent[s + 1 : s + 1 + horizon] for s in starts])
    true_labels = _true_future(test, starts, horizon)
    mean_latents = latents.mean(axis=1)
    latent_mse = float(np.mean((mean_latents - true_latents) ** 2))
    constant = np.repeat(test.latent[starts][:, None, :], horizon, axis=1)
    baseline_mse = float(np.mean((constant - true_latents) ** 2))
    # Per-seed label accuracy, then average; this preserves seed uncertainty.
    acc = float(np.mean(labels == np.repeat(true_labels[:, None, :], labels.shape[1], axis=1)))
    generated = labels.reshape(-1)
    return {
        "latent_mse": latent_mse,
        "latent_mse_skill_vs_constant": float(1.0 - latent_mse / max(baseline_mse, 1e-12)),
        "behavior_accuracy": acc,
        "behavior_distribution_js": js_divergence(
            _distribution(generated), _distribution(true_labels.reshape(-1))
        ),
        "behavior_transition_js": js_divergence(
            _transition(generated), _transition(true_labels.reshape(-1))
        ),
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
            "behavior_accuracy_h1",
            "behavior_accuracy_h10",
            "behavior_accuracy_h25",
            "behavior_transition_js",
            "behavior_distribution_js",
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
) -> tuple[np.ndarray, np.ndarray]:
    n = len(starts)
    latents = np.empty((n, seeds, horizon, 3), dtype=np.float64)
    labels = np.empty((n, seeds, horizon), dtype=np.int64)
    for si, start in enumerate(starts):
        y0 = test.latent[start]
        b0 = int(test.behavior[start])
        for seed in range(seeds):
            if kind == "quadratic":
                traj = quadratic.dynamics.rollout(y0, n_steps=horizon, seed=seed)
                lab = quadratic.behavior.predict(traj)
            elif kind == "switching":
                traj, lab, _ = switching.simulate(
                    y0, b0, n_steps=horizon, noise=0.0, seed=seed
                )
            elif kind == "manifold_walk":
                traj, lab = walk.rollout_with_behavior(
                    y0, n_steps=horizon, seed=seed, temperature=0.6
                )
            else:
                raise ValueError(kind)
            latents[si, seed] = traj
            labels[si, seed] = lab
    return latents, labels


def benchmark_models(
    trajectories: list[WormTrajectory],
    *,
    horizon: int = 25,
    stride: int = 75,
    seeds: int = 4,
) -> dict[str, object]:
    if horizon < 1 or seeds < 1:
        raise ValueError("horizon and seeds must be positive")
    models = ("quadratic", "switching", "manifold_walk")
    folds_by_model: dict[str, list[dict[str, object]]] = {m: [] for m in models}

    for test in trajectories:
        train = [t for t in trajectories if t.worm_id != test.worm_id]
        quadratic = fit_models(train)
        switching = fit_switching(train)
        walk = ManifoldWalk(n_neighbors=20, noise=0.0).fit(train)
        starts = np.arange(0, max(1, len(test.latent) - horizon - 1), stride, dtype=np.int64)

        # One-step deterministic diagnostics.
        one_step = quadratic.dynamics.predict(test.latent[:-1])
        one_step_r2 = r2_score_np(test.latent[1:], one_step)

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
            )
            h1 = 1
            h10 = min(10, horizon)
            h25 = min(25, horizon)
            metrics1 = _score_predictions(
                test, starts, h1, latents[:, :, :h1], labels[:, :, :h1]
            )
            metrics10 = _score_predictions(
                test, starts, h10, latents[:, :, :h10], labels[:, :, :h10]
            )
            metrics25 = _score_predictions(
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
                    "behavior_accuracy_h1": metrics1["behavior_accuracy"],
                    "behavior_accuracy_h10": metrics10["behavior_accuracy"],
                    "behavior_accuracy_h25": metrics25["behavior_accuracy"],
                    "behavior_transition_js": metrics25["behavior_transition_js"],
                    "behavior_distribution_js": metrics25["behavior_distribution_js"],
                }
            )

    result = {
        "format": "vienna-tier1-benchmark-v1",
        "horizon": horizon,
        "stride": stride,
        "seeds": seeds,
        "claim": (
            "Held-out-worm open-loop forecast comparison. Not a mechanistic or causal test."
        ),
        "models": [ModelBenchmark(model, tuple(folds_by_model[model])).aggregate() for model in models],
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
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "reports"
        / "vienna_workbench"
        / "tier1_benchmark.json",
    )
    args = parser.parse_args(argv)
    result = benchmark_models(
        load_trajectories(), horizon=args.horizon, stride=args.stride, seeds=args.seeds
    )
    path = save_benchmark(result, args.out)
    print(json.dumps(result, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
