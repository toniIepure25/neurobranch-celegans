"""Rebuild Phase 8 summary/audit from a completed prediction export.

The main runner writes row-level predictions before the decision bootstrap.  The
bootstrap is intentionally re-run here with the optimized animal-cluster
implementation so a long summary step can be recovered without retraining.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]

from neurobranch.config import PROJECT_ROOT, git_state, resolve_path
from neurobranch.twin.evaluate import (
    add_intervals,
    calibration_scale,
    frame_metrics,
    metric_by_block,
    paired_metric_delta,
)
from neurobranch.twin.sequence import sequence_metrics_to_dict, sequence_rollout
from neurobranch.twin.simulate import TwinRunConfig, _template_trajectory_metrics
from neurobranch.twin.tasks import build_animal_tasks
from neurobranch.twin.train import _make_twin_data, load_fit
from neurobranch.twin.windows import load_window_cache


def _load_config(path: Path) -> TwinRunConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    fields = {k: payload[k] for k in TwinRunConfig.__dataclass_fields__ if k in payload}
    for key in ("model_seeds", "acquisition_seeds", "support_sizes", "strategies"):
        if key in fields:
            fields[key] = tuple(fields[key])
    return TwinRunConfig(**fields)


def _safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        return _safe(obj.item())
    return obj


def _decision(a: pd.DataFrame, b: pd.DataFrame, *, cfg: TwinRunConfig, label: str) -> dict[str, Any]:
    paired = paired_metric_delta(a, b, metric="pair_spearman", n_boot=cfg.n_boot, seed=17)
    blocks = metric_by_block(a)
    compare = metric_by_block(b)
    wins = 0
    n_blocks = 0
    for row in compare.itertuples(index=False):
        mask = (
            (blocks["fold"] == row.fold)
            & (blocks["model_seed"] == row.model_seed)
            & (blocks["strategy"] == row.strategy)
            & (blocks["k"] == row.k)
        )
        base = blocks[mask]
        if not base.empty:
            n_blocks += 1
            wins += int(float(row.pair_spearman) > float(base["pair_spearman"].iloc[0]))
    return {
        "label": label,
        "delta": float(paired["delta"]),
        "ci_low": float(paired["low"]),
        "ci_high": float(paired["high"]),
        "p_one_sided": float(paired["p_one_sided"]),
        "n_boot": float(paired["n_boot"]),
        "wins": wins,
        "n_blocks": n_blocks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase8/apt.yaml")
    parser.add_argument("--skip-h4", action="store_true")
    parser.add_argument("--n-boot", type=int, default=None)
    args = parser.parse_args()
    cfg = _load_config(resolve_path(args.config, PROJECT_ROOT))
    if args.n_boot is not None:
        cfg.n_boot = args.n_boot
    out = resolve_path(cfg.output_dir, PROJECT_ROOT)
    cache = load_window_cache(resolve_path(cfg.cache_path, PROJECT_ROOT))
    tasks = build_animal_tasks(
        cache, split=cfg.split, n_folds=cfg.n_folds, seed=cfg.split_seed
    )
    predictions = pd.read_parquet(out / "predictions.parquet")
    metrics = pd.read_csv(out / "metrics.csv")
    data = _make_twin_data(
        cache,
        trials_path=resolve_path(cfg.trials_path, PROJECT_ROOT),
        connectome_path=resolve_path(cfg.connectome_path, PROJECT_ROOT),
        config=cfg.twin_config(cfg.model_seeds[0]),
    )
    fold_extras: list[dict[str, Any]] = []
    for split in tasks:
        seeds = cfg.model_seeds if not args.skip_h4 else cfg.model_seeds[:1]
        fits = [
            load_fit(
                cache,
                split,
                out / "models" / f"fold{split.fold}_seed{seed}.pt",
                data=data,
            )
            for seed in seeds
        ]
        val_frames = []
        if not args.skip_h4:
            for fit in fits:
                for task in split.val_tasks.values():
                    _, frame = fit.predict_events(task.test_events, support_events=task.support_events)
                    val_frames.append(add_intervals(frame, df=fit.config.df))
        scale = float(np.mean([calibration_scale(f, df=fits[0].config.df) for f in val_frames])) if val_frames else 1.0
        print(f"[phase8-summary] fold {split.fold}: template", flush=True)
        template = _template_trajectory_metrics(fits[0], split)
        seq_agg: dict[str, float] = {}
        if not args.skip_h4:
            seq = [
                sequence_metrics_to_dict(sequence_rollout(fits, task, horizon=3, scale=scale))
                for task in split.test_tasks.values()
            ]
            seq_agg = {k: float(np.nanmean([s[k] for s in seq])) for k in seq[0]} if seq else {}
        fold_extras.append({"fold": split.fold, "calibration_scale": scale, **template, "sequence": seq_agg})

    twin0 = predictions[(predictions["model"] == "twin") & (predictions["k"] == 0)]
    cro = predictions[predictions["model"] == "cro_static"]
    glob = predictions[predictions["model"] == "global_mean"]
    summary: dict[str, Any] = {
        "phase": 8,
        "name": "adaptive_perturbation_twin",
        "split": cfg.split,
        "split_seed": cfg.split_seed,
        "n_folds": len(tasks),
        "n_predictions": int(len(predictions)),
        "warning": "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.",
        "fold_extras": fold_extras,
        "calibration_scale": float(np.mean([e["calibration_scale"] for e in fold_extras])),
        "git": git_state(PROJECT_ROOT),
    }
    print("[phase8-summary] H1 bootstrap", flush=True)
    if not cro.empty and not twin0.empty:
        h1 = _decision(cro, twin0, cfg=cfg, label="twin_k0_vs_cro")
        twin_traj = float(
            metrics[(metrics["model"] == "twin") & (metrics["k"] == 0)]["trajectory_pearson"].mean()
        )
        template_traj = float(np.nanmean([e.get("trajectory_pearson", np.nan) for e in fold_extras]))
        h1.update(
            trajectory_twin=twin_traj,
            trajectory_template=template_traj,
            trajectory_delta=float(twin_traj - template_traj),
            pair_gate=bool(h1["delta"] >= 0.010 and h1["ci_low"] > 0),
            trajectory_gate=bool(twin_traj - template_traj >= 0.050),
        )
        h1["supported"] = bool(h1["pair_gate"] and h1["trajectory_gate"])
        summary["h1_twin_vs_cro"] = h1
    if not twin0.empty and not glob.empty:
        summary["h1_twin_vs_global"] = _decision(glob, twin0, cfg=cfg, label="twin_k0_vs_global")
    first_acq = cfg.acquisition_seeds[0]
    random_k = predictions[
        (predictions["model"] == "twin")
        & (predictions["strategy"] == "random")
        & (predictions["k"] == cfg.primary_k)
        & (predictions["acquisition_seed"] == first_acq)
    ]
    print("[phase8-summary] H2 bootstrap", flush=True)
    if not random_k.empty and not twin0.empty:
        h2 = _decision(twin0, random_k, cfg=cfg, label="random_k8_vs_k0")
        h2["supported"] = bool(h2["delta"] >= 0.010 and h2["ci_low"] > 0)
        summary["h2_k8_vs_k0"] = h2
    eig_k = predictions[
        (predictions["model"] == "twin")
        & (predictions["strategy"] == "eig")
        & (predictions["k"] == cfg.primary_k)
        & (predictions["acquisition_seed"] == first_acq)
    ]
    print("[phase8-summary] H3 bootstrap", flush=True)
    if not eig_k.empty and not random_k.empty:
        h3 = _decision(random_k, eig_k, cfg=cfg, label="eig_vs_random_k8")
        h3["supported"] = bool(h3["delta"] >= 0.010 and h3["ci_low"] > 0)
        h3["confirmatory"] = bool(h3["supported"] and summary.get("h2_k8_vs_k0", {}).get("supported", False))
        summary["h3_eig_vs_random"] = h3
    seqs = [e["sequence"] for e in fold_extras if e.get("sequence")]
    if seqs:
        h4 = {k: float(np.nanmean([s[k] for s in seqs])) for k in seqs[0]}
        h4["supported"] = bool(
            h4.get("trajectory_delta", -1.0) >= 0.050
            and 0.80 <= h4.get("coverage", 0.0) <= 0.95
        )
        summary["h4_sequence_rollout"] = h4
    primary = predictions[(predictions["model"] == "twin") & (predictions["k"] == cfg.primary_k)]
    summary["strategy_summary_primary"] = [
        {"strategy": str(s), **frame_metrics(g)} for s, g in primary.groupby("strategy")
    ]
    summary = _safe(summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    audit: dict[str, Any] = {
        "split": cfg.split,
        "split_seed": cfg.split_seed,
        "n_folds": len(tasks),
        "animal_overlaps": [],
        "support_test_target_overlaps": [],
        "support_test_event_overlaps": [],
        "test_event_in_train_or_val": [],
        "fold_leaks": [],
        "prediction_sha256": __import__("hashlib").sha256((out / "predictions.parquet").read_bytes()).hexdigest(),
    }
    for split in tasks:
        train_a = set(split.train_animals)
        val_a = set(split.val_animals)
        test_a = set(split.test_animals)
        if train_a & val_a or train_a & test_a or val_a & test_a:
            audit["fold_leaks"].append({"fold": split.fold})
            audit["animal_overlaps"].append({"fold": split.fold})
        train_events = {
            e
            for task in split.train_tasks.values()
            for e in (*task.support_events, *task.test_events)
        }
        val_events = {
            e
            for task in split.val_tasks.values()
            for e in (*task.support_events, *task.test_events)
        }
        for task in split.test_tasks.values():
            if set(task.support_targets) & set(task.test_targets):
                audit["support_test_target_overlaps"].append({"fold": split.fold, "animal": task.animal})
            if set(task.support_events) & set(task.test_events):
                audit["support_test_event_overlaps"].append({"fold": split.fold, "animal": task.animal})
            leaked = set(task.test_events) & (train_events | val_events)
            if leaked:
                audit["test_event_in_train_or_val"].append(
                    {"fold": split.fold, "animal": task.animal, "n": len(leaked)}
                )
    audit["audit_clean"] = not (
        audit["animal_overlaps"]
        or audit["support_test_target_overlaps"]
        or audit["support_test_event_overlaps"]
        or audit["test_event_in_train_or_val"]
        or audit["fold_leaks"]
    )
    (out / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
