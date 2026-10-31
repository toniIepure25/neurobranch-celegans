"""Checkpoint discovery, model cards and validation-only model selection."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neurobranch.config import (
    PROJECT_ROOT,
    ExperimentConfig,
    git_state,
    resolve_path,
    sha256_file,
)
from neurobranch.dynamics.training import effective_history, load_worm_data, run_single
from neurobranch.preprocessing import WormData, split_starts
from neurobranch.simulator.adapters import NeuralSimulator
from neurobranch.simulator.metrics import one_step_metrics, rollout_metrics

DEFAULT_KINDS = (
    "persistence",
    "var",
    "mlp",
    "mlp_residual",
    "gru",
    "latent_mlp",
    "latent_knn",
)
DEFAULT_SEEDS = (17, 23, 31)
SIMPLICITY = {kind: i for i, kind in enumerate(DEFAULT_KINDS)}


def checkpoint_root(cfg: ExperimentConfig) -> Path:
    return resolve_path(cfg.output_dir)


def checkpoint_path(root: str | Path, kind: str, worm_id: int, seed: int) -> Path:
    return Path(root) / f"{kind}__worm{worm_id}__seed{seed}" / "model.pt"


def load_simulator_for(
    data: WormData,
    kind: str,
    seed: int,
    *,
    root: str | Path,
) -> NeuralSimulator:
    path = checkpoint_path(root, kind, data.worm_id, seed)
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    sim = NeuralSimulator.from_checkpoint(path, data)
    meta = sim.metadata
    if int(meta.get("worm_id", data.worm_id)) != int(data.worm_id):
        raise ValueError(f"Checkpoint worm mismatch at {path}")
    return sim


def validation_score(
    data: WormData,
    kind: str,
    seed: int,
    *,
    root: str | Path,
) -> dict[str, Any]:
    sim = load_simulator_for(data, kind, seed, root=root)
    history = int(getattr(sim.model, "history_len", 1))
    val = split_starts(data, "val", history=history, horizon=1)
    one = one_step_metrics(sim, val, history=history)
    h5 = min(5, max(1, len(val) - 1))
    selected = np.asarray(
        [int(t) for t in val if data.masks["val"][int(t) : int(t) + 6].all()],
        dtype=np.int64,
    )[:20]
    if len(selected) == 0:
        selected = val[: min(20, len(val))]
    roll, _ = rollout_metrics(
        sim,
        selected,
        horizon=5,
        noise_std=0.0,
        base_seed=1000 + seed,
    )
    return {
        "kind": kind,
        "worm_id": int(data.worm_id),
        "seed": int(seed),
        "val_r2": one["r2"],
        "val_mse": one["mse"],
        "val_persistence_r2": one["persistence_r2"],
        "val_mse5": roll["mse_h5"] if h5 else float("nan"),
        "n_val": int(len(val)),
        "history": history,
        "checkpoint": str(checkpoint_path(root, kind, data.worm_id, seed)),
    }


def select_on_validation(
    datasets: list[WormData],
    *,
    root: str | Path,
    kinds: tuple[str, ...] = DEFAULT_KINDS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> tuple[str, list[dict[str, Any]], dict[str, dict[str, float]]]:
    rows: list[dict[str, Any]] = []
    for kind in kinds:
        for data in datasets:
            for seed in seeds:
                try:
                    rows.append(validation_score(data, kind, seed, root=root))
                except (FileNotFoundError, ValueError) as exc:
                    rows.append(
                        {
                            "kind": kind,
                            "worm_id": int(data.worm_id),
                            "seed": int(seed),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
    valid = [r for r in rows if "error" not in r]
    if not valid:
        raise RuntimeError("No validation scores were computed")
    means: dict[str, dict[str, float]] = {}
    for kind in kinds:
        sub = [r for r in valid if r["kind"] == kind]
        if not sub:
            continue
        means[kind] = {
            "mean_val_r2": float(np.nanmean([r["val_r2"] for r in sub])),
            "mean_val_mse5": float(np.nanmean([r["val_mse5"] for r in sub])),
            "n": len(sub),
        }
    # Primary: mean validation R2. Tie: lower 5-step MSE. Tie: simpler model.
    winner = sorted(
        means,
        key=lambda k: (-means[k]["mean_val_r2"], means[k]["mean_val_mse5"], SIMPLICITY.get(k, 999)),
    )[0]
    return winner, rows, means


def model_card(
    sim: NeuralSimulator,
    data: WormData,
    *,
    root: str | Path,
) -> dict[str, Any]:
    path = checkpoint_path(root, sim.model_kind, data.worm_id, int(sim.metadata.get("seed", 0)))
    meta = sim.metadata
    metrics = meta.get("metrics", {})
    return {
        "schema_version": 1,
        "model_kind": sim.model_kind,
        "worm_id": int(data.worm_id),
        "seed": int(meta.get("seed", 0)),
        "history": int(getattr(sim.model, "history_len", 1)),
        "n_neurons": int(data.X.shape[1]),
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path) if path.is_file() else None,
        "dataset_sha256": data.recording.source_sha256,
        "split_bounds": {k: list(v) for k, v in data.split_bounds.items()},
        "predictor_names": list(data.predictor_names),
        "excluded_neurons": list(data.excluded_neurons),
        "existing_metrics": {
            key: metrics.get(key)
            for key in ("one_step_r2", "one_step_mse", "behavior_accuracy", "rollout_mse", "rollout_correlation")
            if key in metrics
        },
        "checkpoint_metadata": {
            key: meta.get(key)
            for key in ("kind", "worm_id", "seed", "checkpoint_version")
            if key in meta
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "git": git_state(PROJECT_ROOT),
    }


def write_cards(
    datasets: list[WormData],
    *,
    root: str | Path,
    output: str | Path,
    kinds: tuple[str, ...] = DEFAULT_KINDS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> list[dict[str, Any]]:
    output = Path(output)
    (output / "model_cards").mkdir(parents=True, exist_ok=True)
    cards = []
    for kind in kinds:
        for data in datasets:
            for seed in seeds:
                try:
                    sim = load_simulator_for(data, kind, seed, root=root)
                except (FileNotFoundError, ValueError):
                    continue
                card = model_card(sim, data, root=root)
                cards.append(card)
                name = f"{kind}__worm{data.worm_id}__seed{seed}.json"
                (output / "model_cards" / name).write_text(
                    json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
    return cards


def fit_missing(
    datasets: list[WormData],
    cfg: ExperimentConfig,
    *,
    root: str | Path,
    kinds: tuple[str, ...] = DEFAULT_KINDS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> list[dict[str, Any]]:
    rows = []
    for kind in kinds:
        for data in datasets:
            for seed in seeds:
                path = checkpoint_path(root, kind, data.worm_id, seed)
                if path.is_file():
                    continue
                metrics = run_single(kind, data, cfg, seed=seed, output_dir=root)
                rows.append(
                    {
                        "kind": kind,
                        "worm_id": int(data.worm_id),
                        "seed": int(seed),
                        "one_step_r2": metrics.get("one_step_r2"),
                        "checkpoint": str(path),
                    }
                )
    return rows


def discover_available(
    datasets: list[WormData],
    *,
    root: str | Path,
    kinds: tuple[str, ...] = DEFAULT_KINDS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> dict[str, int]:
    return {
        kind: sum(checkpoint_path(root, kind, d.worm_id, s).is_file() for d in datasets for s in seeds)
        for kind in kinds
    }


def load_datasets(cfg: ExperimentConfig) -> list[WormData]:
    return load_worm_data(cfg)


def configured_kinds(cfg: ExperimentConfig) -> tuple[str, ...]:
    del cfg
    return DEFAULT_KINDS


def effective_history_for(kind: str, cfg: ExperimentConfig) -> int:
    return effective_history(kind, cfg)
