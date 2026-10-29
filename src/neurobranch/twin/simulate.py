"""Phase 8 benchmark runner: fit twins, calibrate and evaluate support sets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurobranch.causal.features import build_feature_bundle
from neurobranch.causal.response_operator import CROConfig, fit_cro
from neurobranch.config import PROJECT_ROOT, git_state, resolve_path, sha256_file
from neurobranch.twin.design import select_support_prefix
from neurobranch.twin.evaluate import (
    add_intervals,
    calibration_scale,
    frame_metrics,
    paired_metric_delta,
    trajectory_pearson,
)
from neurobranch.twin.model import TwinConfig
from neurobranch.twin.sequence import sequence_metrics_to_dict, sequence_rollout
from neurobranch.twin.tasks import AnimalTask, AnimalTaskSplit, build_animal_tasks
from neurobranch.twin.train import TwinFit, _make_twin_data, fit_twin, save_fit
from neurobranch.twin.windows import EventWindowCache, WindowConfig, build_event_windows, load_window_cache

DEFAULT_TRIALS = "reports/phase3/randi_trial_table_phase3.csv.gz"
DEFAULT_RAW = "data/raw/randi2023/exported_data"
DEFAULT_CONNECTOME = "references/external/aconnectome_witvliet_2020_8_utf8.csv"


@dataclass
class TwinRunConfig:
    trials_path: str = DEFAULT_TRIALS
    raw_dir: str = DEFAULT_RAW
    connectome_path: str = DEFAULT_CONNECTOME
    output_dir: str = "reports/phase8"
    cache_path: str = "reports/phase8/cache/windows.npz"
    split: str = "unseen_animals"
    split_seed: int = 29
    n_folds: int = 5
    model_seeds: tuple[int, ...] = (17, 23, 31)
    acquisition_seeds: tuple[int, ...] = (101, 202)
    support_sizes: tuple[int, ...] = (0, 2, 4, 8, 16)
    primary_k: int = 8
    strategies: tuple[str, ...] = ("random", "uncertainty", "diversity", "eig", "hybrid")
    rank: int = 16
    hidden: int = 64
    temporal_basis: int = 8
    support_dim: int = 16
    dropout: float = 0.10
    epochs: int = 30
    batch_size: int = 4096
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    patience: int = 6
    traj_loss_weight: float = 1.0
    df: float = 5.0
    n_boot: int = 1000
    cro_epochs: int = 12
    max_test_animals: int | None = None
    smoke: bool = False

    def twin_config(self, seed: int) -> TwinConfig:
        return TwinConfig(
            rank=self.rank,
            hidden=self.hidden,
            temporal_basis=self.temporal_basis,
            support_dim=self.support_dim,
            dropout=self.dropout,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            patience=self.patience,
            seed=seed,
            traj_loss_weight=self.traj_loss_weight,
            df=self.df,
        )


def _load_or_build_cache(cfg: TwinRunConfig, base: Path) -> EventWindowCache:
    path = resolve_path(cfg.cache_path, base)
    if path.exists():
        return load_window_cache(path)
    return build_event_windows(
        cfg.trials_path,
        cfg.raw_dir,
        config=WindowConfig(),
        output_path=path,
        root=base,
    )


def _ensemble_frames(
    fits: list[TwinFit],
    events: list[str],
    *,
    support_events: list[str],
    df: float,
    scale: float,
) -> tuple[pd.DataFrame, float]:
    preds = []
    frames = []
    for fit in fits:
        pred, frame = fit.predict_events(events, support_events=support_events)
        preds.append(pred)
        frames.append(frame)
    if not preds:
        raise ValueError("no fits")
    mu = np.mean([p.scalar_mu for p in preds], axis=0)
    var = np.mean(
        [p.scalar_sigma**2 + (p.scalar_mu - mu) ** 2 for p in preds],
        axis=0,
    )
    sigma = np.sqrt(var)
    traj = np.mean([p.trajectory_mu for p in preds], axis=0)
    frame = frames[0].copy()
    frame["y_pred"] = mu
    frame["sigma"] = sigma
    frame = add_intervals(frame, df=df, scale=scale)
    frame["traj_pred_mean"] = traj.mean(axis=1)
    # Mean trajectory correlation across ensemble members; the first member is
    # sufficient for the stored scalar table, the aggregate is computed here.
    return frame, float(traj.mean())


def _trajectory_for_frame(fit: TwinFit, frame: pd.DataFrame, events: list[str], support: list[str]) -> tuple[np.ndarray, np.ndarray]:
    pred, _ = fit.predict_events(events, support_events=support)
    ev, ridx = fit.data.rows_for_events(events)
    if ev.size != len(pred.scalar_mu):
        raise RuntimeError("trajectory rows do not align")
    return fit.data.cache.post[ev, ridx, :], pred.trajectory_mu


def _cro_frame(
    cfg: TwinRunConfig,
    cache: EventWindowCache,
    split: AnimalTaskSplit,
    base: Path,
) -> pd.DataFrame:
    trials = pd.read_csv(resolve_path(cfg.trials_path, base))
    bundle = build_feature_bundle(trials, connectome_path=resolve_path(cfg.connectome_path, base))
    event_to_row = {str(e): i for i, e in enumerate(bundle.event_ids.astype(str))}
    train_events = {e for t in split.train_tasks.values() for e in t.test_events}
    val_events = {e for t in split.val_tasks.values() for e in t.test_events}
    test_events = {e for t in split.test_tasks.values() for e in t.test_events}
    train_idx = np.array([i for e, i in event_to_row.items() if e in train_events], dtype=np.int64)
    val_idx = np.array([i for e, i in event_to_row.items() if e in val_events], dtype=np.int64)
    test_idx = np.array([i for e, i in event_to_row.items() if e in test_events], dtype=np.int64)
    if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
        return pd.DataFrame()
    train_targets = {bundle.target_names[int(bundle.target_idx[i])] for i in train_idx}
    test_targets = {bundle.target_names[int(bundle.target_idx[i])] for i in test_idx}
    held_out_targets = test_targets - train_targets
    held_out_animals = set(split.test_animals)
    cro = fit_cro(
        bundle,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        held_out_targets=held_out_targets,
        held_out_animals=held_out_animals,
        config=CROConfig(
            rank=16,
            hidden=64,
            epochs=cfg.cro_epochs,
            batch_size=8192,
            learning_rate=2e-3,
            weight_decay=1e-3,
            patience=6,
            seed=17,
        ),
    )
    mu, sigma = cro.predict(bundle, test_idx)
    frame = pd.DataFrame(
        {
            "fold": split.fold,
            "model_seed": 0,
            "acquisition_seed": 0,
            "model": "cro_static",
            "strategy": "none",
            "k": 0,
            "event_id": bundle.event_ids[test_idx].astype(str),
            "dataset_id": [bundle.animal_names[int(i)] for i in bundle.animal_idx[test_idx]],
            "target": [bundle.target_names[int(i)] for i in bundle.target_idx[test_idx]],
            "neuron": [bundle.responder_names[int(i)] for i in bundle.responder_idx[test_idx]],
            "y_true": bundle.y[test_idx].astype(float),
            "y_pred": mu,
            "sigma": sigma,
        }
    )
    frame = frame.drop_duplicates(["event_id", "target", "neuron"]).reset_index(drop=True)
    return add_intervals(frame)


def _global_frame(
    cfg: TwinRunConfig,
    cache: EventWindowCache,
    split: AnimalTaskSplit,
) -> pd.DataFrame:
    train_events = {e for t in split.train_tasks.values() for e in t.test_events}
    lookup = {str(e): i for i, e in enumerate(cache.event_ids.astype(str))}
    rows = [lookup[e] for e in train_events if e in lookup]
    if not rows:
        return pd.DataFrame()
    ridx = np.concatenate([np.flatnonzero(cache.mask[i]) for i in rows])
    y = cache.scalar[np.repeat(np.asarray(rows), [int(cache.mask[i].sum()) for i in rows]), ridx]
    mean = float(np.mean(y)) if y.size else 0.0
    test_events = [e for t in split.test_tasks.values() for e in t.test_events]
    ev, rr = [], []
    for e in test_events:
        i = lookup.get(e)
        if i is None:
            continue
        idx = np.flatnonzero(cache.mask[i])
        ev.extend([i] * len(idx))
        rr.extend(idx.tolist())
    if not ev:
        return pd.DataFrame()
    ev_arr = np.asarray(ev, dtype=np.int64)
    rr_arr = np.asarray(rr, dtype=np.int64)
    frame = pd.DataFrame(
        {
            "fold": split.fold,
            "model_seed": 0,
            "acquisition_seed": 0,
            "model": "global_mean",
            "strategy": "none",
            "k": 0,
            "event_id": cache.event_ids[ev_arr].astype(str),
            "dataset_id": cache.dataset_ids[ev_arr].astype(str),
            "target": [cache.target_names[int(i)] for i in cache.target_idx[ev_arr]],
            "neuron": [cache.responder_names[int(i)] for i in rr_arr],
            "y_true": cache.scalar[ev_arr, rr_arr].astype(float),
            "y_pred": mean,
            "sigma": float(np.std(y)) if y.size else 1.0,
        }
    )
    frame = frame.drop_duplicates(["event_id", "target", "neuron"]).reset_index(drop=True)
    return add_intervals(frame)


def _template_trajectory_metrics(
    fit: TwinFit,
    split: AnimalTaskSplit,
    *,
    max_animals: int | None = None,
) -> dict[str, float]:
    """Temporal-template baseline: per-responder mean training trajectory."""
    train_events = [e for t in split.train_tasks.values() for e in t.test_events]
    ev, ridx = fit.data.rows_for_events(train_events)
    if ev.size == 0:
        return {"trajectory_pearson": float("nan"), "n_events": 0.0}
    post = fit.data.cache.post[ev, ridx, :]
    n_resp = len(fit.data.cache.responder_names)
    template = np.zeros((n_resp, post.shape[-1]), dtype=float)
    counts = np.zeros(n_resp, dtype=int)
    for ri in range(n_resp):
        sel = ridx == ri
        if sel.any():
            template[ri] = post[sel].mean(axis=0)
            counts[ri] = int(sel.sum())
    global_mean = post.mean(axis=0)
    template[counts == 0] = global_mean
    tasks = list(split.test_tasks.values())
    if max_animals is not None:
        tasks = tasks[:max_animals]
    true = []
    pred = []
    for task in tasks:
        ev, ridx = fit.data.rows_for_events(task.test_events)
        if ev.size == 0:
            continue
        true.append(fit.data.cache.post[ev, ridx, :])
        pred.append(template[ridx])
    if not true:
        return {"trajectory_pearson": float("nan"), "n_events": 0.0}
    return {
        "trajectory_pearson": trajectory_pearson(np.concatenate(true), np.concatenate(pred)),
        "n_events": float(sum(len(x) for x in true)),
    }


def _support_frame(
    fits: list[TwinFit],
    task: AnimalTask,
    *,
    support_events: list[str],
    strategy: str,
    k: int,
    acquisition_seed: int,
    model_seed: int,
    df: float,
    scale: float,
) -> tuple[pd.DataFrame, float]:
    frame, _ = _ensemble_frames(
        fits,
        list(task.test_events),
        support_events=support_events,
        df=df,
        scale=scale,
    )
    frame["model_seed"] = model_seed
    frame["acquisition_seed"] = acquisition_seed
    frame["model"] = "twin"
    frame["strategy"] = strategy
    frame["k"] = k
    # Trajectory correlation uses the first ensemble member; scalar metrics use
    # the ensemble mean.  This is reported as a secondary endpoint.
    true_traj, pred_traj = _trajectory_for_frame(fits[0], frame, list(task.test_events), support_events)
    return frame, trajectory_pearson(true_traj, pred_traj)


def _evaluate_fold(
    cfg: TwinRunConfig,
    cache: EventWindowCache,
    split: AnimalTaskSplit,
    fits: list[TwinFit],
    *,
    base: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    selection_log: list[dict[str, Any]] = []
    # Validation-only calibration scale, averaged over folds/fits.
    scales: list[float] = []
    for fit in fits:
        val_frame = []
        for task in split.val_tasks.values():
            _, f = fit.predict_events(task.test_events, support_events=task.support_events)
            val_frame.append(add_intervals(f, df=fit.config.df))
        if val_frame:
            scales.append(calibration_scale(pd.concat(val_frame, ignore_index=True), df=fit.config.df))
    scale = float(np.mean(scales)) if scales else 1.0
    df = fits[0].config.df

    # Static baselines.
    cro = _cro_frame(cfg, cache, split, base)
    glob = _global_frame(cfg, cache, split)
    for base_frame in (cro, glob):
        if not base_frame.empty:
            frames.append(base_frame)
            metric_rows.append({**frame_metrics(base_frame), "model": base_frame["model"].iloc[0], "k": 0, "strategy": "none", "fold": split.fold, "model_seed": 0, "acquisition_seed": 0, "trajectory_pearson": float("nan")})

    test_tasks = list(split.test_tasks.values())
    if cfg.max_test_animals is not None:
        test_tasks = test_tasks[: cfg.max_test_animals]
    for task in test_tasks:
        # k=0 posterior prior; predictions are the deep-ensemble mean so the
        # paired comparison has exactly one row per test cell.
        frame0, traj0 = _support_frame(
            fits,
            task,
            support_events=[],
            strategy="none",
            k=0,
            acquisition_seed=0,
            model_seed=0,
            df=df,
            scale=scale,
        )
        frame0["fold"] = split.fold
        frames.append(frame0)
        metric_rows.append({**frame_metrics(frame0), "model": "twin", "k": 0, "strategy": "none", "fold": split.fold, "model_seed": 0, "acquisition_seed": 0, "trajectory_pearson": traj0})
        for strategy in cfg.strategies:
            for acq_seed in cfg.acquisition_seeds:
                max_k = (
                    max(cfg.support_sizes)
                    if strategy == "random"
                    else cfg.primary_k
                )
                max_k = min(max_k, len(task.support_events))
                prefix = select_support_prefix(
                    fits[0],
                    task,
                    n=max_k,
                    strategy=strategy,
                    seed=acq_seed + 7919 * split.fold,
                    ensemble=fits,
                )
                selection_log.append(
                    {
                        "fold": split.fold,
                        "animal": task.animal,
                        "strategy": strategy,
                        "acquisition_seed": acq_seed,
                        "selected": prefix,
                        "support_targets": list(task.support_targets),
                    }
                )
                # Full budget curves only for the random arm; the other
                # strategies are compared at the preregistered primary k to
                # keep the row-level audit small.
                ks = cfg.support_sizes if strategy == "random" else (cfg.primary_k,)
                for k in ks:
                    if k == 0 or k > len(prefix):
                        continue
                    support = prefix[:k]
                    frame, traj = _support_frame(
                        fits,
                        task,
                        support_events=support,
                        strategy=strategy,
                        k=int(k),
                        acquisition_seed=acq_seed,
                        model_seed=0,
                        df=df,
                        scale=scale,
                    )
                    frame["fold"] = split.fold
                    frames.append(frame)
                    metric_rows.append({**frame_metrics(frame), "model": "twin", "k": int(k), "strategy": strategy, "fold": split.fold, "model_seed": 0, "acquisition_seed": acq_seed, "trajectory_pearson": traj})
    template_metrics = _template_trajectory_metrics(
        fits[0], split, max_animals=cfg.max_test_animals
    )
    sequence_rows = []
    for task in test_tasks:
        sequence_rows.append(sequence_metrics_to_dict(sequence_rollout(fits, task, horizon=3, scale=scale)))
    sequence_agg: dict[str, float] = {}
    if sequence_rows:
        for key in sequence_rows[0]:
            sequence_agg[key] = float(np.nanmean([row[key] for row in sequence_rows]))
    return (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
        pd.DataFrame(metric_rows),
        selection_log,
        {"calibration_scale": scale, **template_metrics, "sequence": sequence_agg},
    )


def _decision(
    twin0: pd.DataFrame,
    comparator: pd.DataFrame,
    *,
    cfg: TwinRunConfig,
    label: str,
) -> dict[str, Any]:
    delta = paired_metric_delta(twin0, comparator, n_boot=cfg.n_boot)
    block_wins = 0
    block_n = 0
    for (fold, seed), g0 in twin0.groupby(["fold", "model_seed"]):
        gc = comparator[(comparator["fold"] == fold) & (comparator["model_seed"] == seed)]
        if gc.empty:
            continue
        block_n += 1
        if frame_metrics(gc)["pair_spearman"] > frame_metrics(g0)["pair_spearman"]:
            block_wins += 1
    return {
        "label": label,
        "delta": delta["delta"],
        "ci_low": delta["low"],
        "ci_high": delta["high"],
        "p_one_sided": delta.get("p_one_sided", float("nan")),
        "n_boot": delta["n_boot"],
        "wins": block_wins,
        "n_blocks": block_n,
    }


def run_twin_benchmark(cfg: TwinRunConfig | None = None, *, root: str | Path | None = None) -> dict[str, Any]:
    cfg = cfg or TwinRunConfig()
    base = Path(root) if root is not None else PROJECT_ROOT
    out = resolve_path(cfg.output_dir, base)
    out.mkdir(parents=True, exist_ok=True)
    cache = _load_or_build_cache(cfg, base)
    tasks = build_animal_tasks(
        cache,
        split=cfg.split,
        n_folds=cfg.n_folds,
        seed=cfg.split_seed,
        min_support_targets=2,
        min_test_targets=2,
    )
    data = _make_twin_data(
        cache,
        trials_path=resolve_path(cfg.trials_path, base),
        connectome_path=resolve_path(cfg.connectome_path, base),
        config=cfg.twin_config(cfg.model_seeds[0]),
    )
    all_frames: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    all_logs: list[dict[str, Any]] = []
    fit_summaries: list[dict[str, Any]] = []
    fold_extras: list[dict[str, Any]] = []
    for split in tasks:
        fits: list[TwinFit] = []
        for seed in cfg.model_seeds:
            fit = fit_twin(cache, split, config=cfg.twin_config(seed), data=data)
            save_fit(fit, out / "models" / f"fold{split.fold}_seed{seed}.pt")
            fits.append(fit)
            fit_summaries.append(
                {
                    "fold": split.fold,
                    "seed": seed,
                    "epochs_ran": len(fit.history),
                    "best_val_loss": min((h["val_loss"] for h in fit.history), default=float("nan")),
                }
            )
        frames, metrics, logs, extra = _evaluate_fold(cfg, cache, split, fits, base=base)
        if not frames.empty:
            all_frames.append(frames)
        if not metrics.empty:
            all_metrics.append(metrics)
        all_logs.extend(logs)
        fold_extras.append({"fold": split.fold, **extra})
    predictions = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    if not predictions.empty:
        keep = (
            (predictions["model"] != "twin")
            | (predictions["k"] == 0)
            | (
                (predictions["k"] == cfg.primary_k)
                & predictions["strategy"].isin(["random", "eig", "hybrid"])
            )
        )
        predictions = predictions[keep].reset_index(drop=True)
    metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    if not predictions.empty:
        predictions.to_parquet(out / "predictions.parquet", index=False)
    if not metrics.empty:
        metrics.to_csv(out / "metrics.csv", index=False)
    with (out / "selection_log.jsonl").open("w", encoding="utf-8") as fh:
        for row in all_logs:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    summary: dict[str, Any] = {
        "config": asdict(cfg),
        "fit_summaries": fit_summaries,
        "fold_extras": fold_extras,
        "n_predictions": int(len(predictions)),
        "n_metrics": int(len(metrics)),
    }
    if not predictions.empty:
        twin0 = predictions[(predictions["model"] == "twin") & (predictions["k"] == 0)]
        cro = predictions[predictions["model"] == "cro_static"]
        glob = predictions[predictions["model"] == "global_mean"]
        if not twin0.empty and not cro.empty:
            h1 = _decision(cro, twin0, cfg=cfg, label="twin_k0_vs_cro")
            twin_traj = metrics[
                (metrics["model"] == "twin") & (metrics["k"] == 0)
            ]["trajectory_pearson"].mean()
            template_traj = float(
                np.nanmean([e.get("trajectory_pearson", np.nan) for e in fold_extras])
            )
            h1["trajectory_twin"] = float(twin_traj)
            h1["trajectory_template"] = template_traj
            h1["trajectory_delta"] = float(twin_traj - template_traj)
            h1["pair_gate"] = bool(h1["delta"] >= 0.010 and h1["ci_low"] > 0.0)
            h1["trajectory_gate"] = bool(h1["trajectory_delta"] >= 0.05)
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
        if not random_k.empty:
            h2 = _decision(twin0, random_k, cfg=cfg, label="random_k8_vs_k0")
            h2["supported"] = bool(h2["delta"] >= 0.010 and h2["ci_low"] > 0.0)
            summary["h2_k8_vs_k0"] = h2
        eig_k = predictions[
            (predictions["model"] == "twin")
            & (predictions["strategy"] == "eig")
            & (predictions["k"] == cfg.primary_k)
            & (predictions["acquisition_seed"] == first_acq)
        ]
        if not eig_k.empty and not random_k.empty:
            h3 = _decision(random_k, eig_k, cfg=cfg, label="eig_vs_random_k8")
            h3["supported"] = bool(h3["delta"] >= 0.010 and h3["ci_low"] > 0.0)
            h3["confirmatory"] = bool(h3["supported"] and summary.get("h2_k8_vs_k0", {}).get("supported", False))
            summary["h3_eig_vs_random"] = h3
        seqs = [e["sequence"] for e in fold_extras if e.get("sequence")]
        if seqs:
            keys = seqs[0].keys()
            h4 = {key: float(np.nanmean([s[key] for s in seqs])) for key in keys}
            h4["supported"] = bool(
                h4.get("trajectory_delta", -1.0) >= 0.05
                and 0.80 <= h4.get("coverage", 0.0) <= 0.95
            )
            summary["h4_sequence_rollout"] = h4
        # Strategy summaries at the primary k.
        primary = predictions[(predictions["model"] == "twin") & (predictions["k"] == cfg.primary_k)]
        if not primary.empty:
            summary["strategy_summary_primary"] = [
                {"strategy": str(strategy), **frame_metrics(group)}
                for strategy, group in primary.groupby("strategy")
            ]
    def _json_safe(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        if isinstance(obj, (np.floating, np.integer)):
            return _json_safe(obj.item())
        return obj

    summary = _json_safe(summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    audit: dict[str, Any] = {
        "split": cfg.split,
        "split_seed": cfg.split_seed,
        "n_folds": len(tasks),
        "fold_leaks": [],
        "support_test_target_overlaps": [],
        "support_test_event_overlaps": [],
        "animal_overlaps": [],
        "test_event_in_train_or_val": [],
        "trials_sha256": sha256_file(resolve_path(cfg.trials_path, base)),
        "connectome_sha256": sha256_file(resolve_path(cfg.connectome_path, base)),
        "git": git_state(base),
    }
    for split in tasks:
        train_a = set(split.train_animals)
        val_a = set(split.val_animals)
        test_a = set(split.test_animals)
        if train_a & val_a or train_a & test_a or val_a & test_a:
            audit["fold_leaks"].append({"fold": split.fold})
            audit["animal_overlaps"].append(
                {
                    "fold": split.fold,
                    "train_val": sorted(train_a & val_a),
                    "train_test": sorted(train_a & test_a),
                    "val_test": sorted(val_a & test_a),
                }
            )
        train_events = {e for t in split.train_tasks.values() for e in t.test_events}
        val_events = {e for t in split.val_tasks.values() for e in t.test_events}
        for task in split.tasks.values():
            if set(task.support_targets) & set(task.test_targets):
                audit["support_test_target_overlaps"].append({"fold": split.fold, "animal": task.animal})
            if set(task.support_events) & set(task.test_events):
                audit["support_test_event_overlaps"].append({"fold": split.fold, "animal": task.animal})
        for task in split.test_tasks.values():
            if set(task.test_events) & (train_events | val_events):
                audit["test_event_in_train_or_val"].append({"fold": split.fold, "animal": task.animal})
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return summary
