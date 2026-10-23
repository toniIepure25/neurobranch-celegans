"""Training and evaluation orchestration for a single worm/model/seed."""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from neurobranch.config import (
    PROJECT_ROOT,
    ExperimentConfig,
    git_state,
    resolve_path,
)
from neurobranch.dynamics.checkpointing import save_model
from neurobranch.dynamics.models import fit_model
from neurobranch.evaluation.metrics import aggregate_results, evaluate_model
from neurobranch.preprocessing import WormData, prepare_all, split_starts


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git": git_state(),
    }


def effective_history(kind: str, cfg: ExperimentConfig) -> int:
    if kind == "gru":
        return max(2, cfg.model.history)
    if kind in {"var", "mlp", "mlp_residual"}:
        return max(1, cfg.model.history)
    return 1


def load_worm_data(cfg: ExperimentConfig) -> list[WormData]:
    from neurobranch.data.kato import load_kato

    dataset = load_kato(cfg.data.path)
    return prepare_all(
        dataset.recordings,
        cfg.split,
        exclude_behavior=cfg.data.exclude_behavior_neurons,
        smoothing_sigma_frames=cfg.data.smoothing_sigma_frames,
        detrend_window_frames=cfg.data.detrend_window_frames,
        detrend_warmup_frames=cfg.data.detrend_warmup_frames,
    )


def run_single(
    kind: str,
    data: WormData,
    cfg: ExperimentConfig,
    *,
    seed: int,
    output_dir: str | Path | None = None,
    save_predictions: bool = True,
) -> dict[str, Any]:
    t0 = time.time()
    history = effective_history(kind, cfg)
    train_starts = split_starts(data, "train", history=history, horizon=1)
    val_starts = split_starts(data, "val", history=history, horizon=1)
    if len(train_starts) < 50 or len(val_starts) < 10:
        raise ValueError(f"Worm {data.worm_id}: insufficient train/val samples")
    model, hist = fit_model(
        kind,
        data.X,
        data.y_macro,
        train_starts,
        val_starts,
        cfg.model,
        replace(cfg.train, seed=seed),
    )
    metrics = evaluate_model(
        model,
        data,
        model_kind=kind,
        seed=seed,
        horizons=tuple(cfg.evaluation.horizons),
        transition_horizon=cfg.evaluation.transition_horizon,
        transition_starts=80,
        rollout_starts=180,
        bootstrap_samples=min(cfg.evaluation.bootstrap_samples, 300),
    )
    metrics.update(
        {
            "kind": kind,
            "n_train_starts": int(len(train_starts)),
            "n_val_starts": int(len(val_starts)),
            "best_epoch": hist.best_epoch,
            "best_val_loss": hist.best_val_loss,
            "elapsed_seconds": time.time() - t0,
            "history": history,
            "split_bounds": {k: list(v) for k, v in data.split_bounds.items()},
            "dataset_sha256": data.recording.source_sha256,
            "predictor_names": list(data.predictor_names),
            "excluded_neurons": list(data.excluded_neurons),
        }
    )
    root = resolve_path(output_dir or cfg.output_dir)
    run_dir = root / f"{kind}__worm{data.worm_id}__seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = save_model(
        model,
        run_dir / "model.pt",
        metadata={
            "kind": kind,
            "worm_id": data.worm_id,
            "seed": seed,
            "config": cfg.to_dict(),
            "metrics": metrics,
            "environment": _environment(),
            "checkpoint_version": 1,
        },
    )
    metrics["checkpoint"] = str(checkpoint.relative_to(PROJECT_ROOT)) if checkpoint.is_relative_to(PROJECT_ROOT) else str(checkpoint)

    if save_predictions:
        starts = split_starts(data, "test", history=history, horizon=1)
        H = np.stack([data.X[t - history + 1 : t + 1] for t in starts])
        pred = model.predict_next(H)
        probs = model.behavior_proba(H)
        np.savez_compressed(
            run_dir / "test_predictions.npz",
            starts=starts,
            pred=pred,
            behavior_probs=probs,
            truth=data.X[starts + 1],
        )
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    with open(run_dir / "train_history.json", "w", encoding="utf-8") as fh:
        json.dump(hist.as_dict(), fh, indent=2, default=float)
    cfg.save(run_dir / "resolved_config.yaml")
    return metrics


def run_suite(
    kinds: Iterable[str],
    *,
    cfg: ExperimentConfig,
    seeds: Iterable[int] | None = None,
    worm_ids: Iterable[int] | None = None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    data = load_worm_data(cfg)
    wanted = set(worm_ids) if worm_ids is not None else {d.worm_id for d in data}
    selected = [d for d in data if d.worm_id in wanted]
    seed_list = list(seeds) if seeds is not None else [cfg.train.seed]
    rows: list[dict[str, Any]] = []
    for kind in kinds:
        kind_cfg = replace(cfg, model=replace(cfg.model, history=12 if kind == "gru" else cfg.model.history))
        for d in selected:
            for seed in seed_list:
                try:
                    m = run_single(kind, d, kind_cfg, seed=seed, output_dir=output_dir)
                    rows.append(m)
                    print(
                        f"[ok] {kind:13s} worm{d.worm_id} seed{seed:3d} "
                        f"R2={m['one_step_r2']:.3f} acc={m['behavior_accuracy']:.3f} "
                        f"TV={m['transition_total_variation']:.3f}",
                        flush=True,
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "kind": kind,
                            "model": kind,
                            "worm_id": d.worm_id,
                            "seed": seed,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    print(f"[FAIL] {kind} worm{d.worm_id} seed{seed}: {exc}", flush=True)
    df = pd.DataFrame(rows)
    root = resolve_path(output_dir or cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    df.to_csv(root / "suite_metrics.csv", index=False)
    valid = [r for r in rows if "error" not in r]
    summary = {
        "experiment": cfg.name,
        "config": cfg.to_dict(),
        "environment": _environment(),
        "n_success": len(valid),
        "n_failure": len(rows) - len(valid),
        "aggregate": {
            model: aggregate_results([r for r in valid if r["model"] == model])
            for model in sorted({r["model"] for r in valid})
        },
        "failures": [r for r in rows if "error" in r],
    }
    with open(root / "suite_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)

    from neurobranch.provenance import write_experiment_manifest

    write_experiment_manifest(
        cfg,
        data,
        kinds=kinds,
        seeds=seed_list,
        output_dir=root,
        artifacts=[
            root / "suite_metrics.csv",
            root / "suite_summary.json",
            root / "run_manifest.json",
        ],
    )
    return df
