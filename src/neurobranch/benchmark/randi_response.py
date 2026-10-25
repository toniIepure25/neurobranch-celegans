"""Phase 6 Randi response benchmark: CRO versus leakage-safe baselines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from neurobranch.benchmark.baselines import predict_baselines
from neurobranch.benchmark.connectivity import evaluate_connectivity
from neurobranch.benchmark.metrics import (
    bootstrap_pair_spearman,
    event_metrics,
    pair_metrics,
    paired_pair_bootstrap_delta,
)
from neurobranch.benchmark.reliability import (
    attenuation_corrected_profile,
    split_half_target_profiles,
)
from neurobranch.benchmark.splits import Fold, build_folds
from neurobranch.causal.export import export_operator
from neurobranch.causal.features import build_feature_bundle
from neurobranch.causal.response_operator import CROConfig, calibration_scale, train_cro
from neurobranch.config import PROJECT_ROOT, git_state, resolve_path, sha256_file

DEFAULT_TRIALS = "reports/phase3/randi_trial_table_phase3.csv.gz"
DEFAULT_CONNECTOME = "references/external/aconnectome_witvliet_2020_8_utf8.csv"


@dataclass
class BenchmarkConfig:
    trials_path: str = DEFAULT_TRIALS
    connectome_path: str = DEFAULT_CONNECTOME
    output_dir: str = "reports/phase6/cro"
    splits: tuple[str, ...] = ("unseen_targets",)
    n_folds: int = 5
    seeds: tuple[int, ...] = (17, 23, 31)
    n_boot: int = 500
    response_column: str = "response_mean"
    rank: int = 16
    hidden: int = 64
    epochs: int = 45
    batch_size: int = 8192
    learning_rate: float = 2e-3
    weight_decay: float = 1e-3
    patience: int = 8
    min_pair_events: int = 2
    max_rows: int | None = None

    def cro_config(self, seed: int) -> CROConfig:
        return CROConfig(
            rank=self.rank,
            hidden=self.hidden,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            patience=self.patience,
            seed=seed,
            df=4.0,
        )


def load_trials(path: str | Path, *, max_rows: int | None = None) -> pd.DataFrame:
    p = resolve_path(path, PROJECT_ROOT)
    if not p.exists():
        raise FileNotFoundError(f"Randi trial table not found: {p}")
    table = pd.read_csv(p)
    required = {"dataset_id", "event_id", "target", "neuron", "response_mean"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"trial table missing {sorted(missing)}")
    table = table.dropna(subset=["target", "neuron", "response_mean"]).reset_index(drop=True)
    if max_rows is not None and len(table) > max_rows:
        # Deterministic animal-stratified subset for smoke tests only.
        animals = np.sort(table["dataset_id"].unique())
        rng = np.random.default_rng(17)
        chosen = rng.choice(animals, size=max(2, int(len(animals) * max_rows / len(table))), replace=False)
        table = table[table["dataset_id"].isin(chosen)].reset_index(drop=True)
    return table


def _unseen_sets(bundle: Any, fold: Fold) -> tuple[set[str], set[str]]:
    train_t = set(bundle.target_idx[fold.train_idx].tolist())
    train_a = set(bundle.animal_idx[fold.train_idx].tolist())
    eval_t = set(bundle.target_idx[fold.test_idx].tolist()) | set(bundle.target_idx[fold.val_idx].tolist())
    eval_a = set(bundle.animal_idx[fold.test_idx].tolist()) | set(bundle.animal_idx[fold.val_idx].tolist())
    unseen_t = {bundle.target_names[i] for i in eval_t - train_t}
    unseen_a = {bundle.animal_names[i] for i in eval_a - train_a}
    return unseen_t, unseen_a


def _event_frame(
    bundle: Any,
    idx: np.ndarray,
    y_pred: np.ndarray,
    *,
    split: str,
    fold: int,
    model: str,
    y_lower: np.ndarray | None = None,
    y_upper: np.ndarray | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "split": split,
            "fold": fold,
            "model": model,
            "event_id": bundle.event_ids[idx],
            "dataset_id": [bundle.animal_names[i] for i in bundle.animal_idx[idx]],
            "target": [bundle.target_names[i] for i in bundle.target_idx[idx]],
            "neuron": [bundle.responder_names[i] for i in bundle.responder_idx[idx]],
            "y_true": bundle.y[idx],
            "y_pred": y_pred,
        }
    )
    if y_lower is not None:
        frame["lower"] = y_lower
        frame["upper"] = y_upper if y_upper is not None else y_lower
    return frame


def _mean_seed_predictions(preds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    mus = np.stack([p[0] for p in preds], axis=0)
    sigmas = np.stack([p[1] for p in preds], axis=0)
    mu = mus.mean(axis=0)
    sigma = np.sqrt(np.mean(sigmas**2 + (mus - mu) ** 2, axis=0))
    return mu, sigma


def _metric_row(
    frame: pd.DataFrame, *, split: str, fold: int, model: str, n_boot: int, seed: int
) -> dict[str, Any]:
    pm = pair_metrics(frame)
    em = event_metrics(frame)
    row = {
        "split": split,
        "fold": fold,
        "model": model,
        "n_events": len(frame),
        "n_pairs": pm["n_pairs"],
        "pair_spearman": pm["spearman"],
        "pair_pearson": pm["pearson"],
        "pair_mae": pm["mae"],
        "pair_sign_accuracy": pm["sign_accuracy"],
        "event_mae": em["mae"],
        "event_rmse": em["rmse"],
        "event_spearman": em["spearman"],
        "event_sign_accuracy": em["sign_accuracy"],
        "pair_spearman_ci_low": np.nan,
        "pair_spearman_ci_high": np.nan,
    }
    if "lower" in frame.columns:
        from neurobranch.benchmark.metrics import coverage

        row["interval_90_coverage"] = coverage(frame)
    return row


def run_split(
    bundle: Any,
    *,
    split: str,
    config: BenchmarkConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = build_folds(_trials_frame(bundle), split, n_folds=config.n_folds, seed=17)
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    all_predictions: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    manifests = []
    for fold in folds:
        manifests.append(fold.manifest())
        unseen_t, unseen_a = _unseen_sets(bundle, fold)
        seed_preds = []
        for seed in config.seeds:
            result = train_cro(
                bundle,
                train_idx=fold.train_idx,
                val_idx=fold.val_idx,
                test_idx=fold.test_idx,
                held_out_targets=unseen_t,
                held_out_animals=unseen_a,
                config=config.cro_config(seed),
            )
            seed_preds.append(
                (
                    result["test_mu"],
                    result["test_sigma"],
                    result["val_mu"],
                    result["val_sigma"],
                )
            )
        mu, sigma = _mean_seed_predictions([(p[0], p[1]) for p in seed_preds])
        scales = [
            calibration_scale(bundle.y[fold.val_idx], p[2], p[3], level=0.90, df=4.0) for p in seed_preds
        ]
        scale = float(np.mean(scales))
        q = stats.t.ppf(0.95, df=4.0)
        cro_frame = _event_frame(
            bundle,
            fold.test_idx,
            mu,
            split=split,
            fold=fold.fold,
            model="cro",
            y_lower=mu - q * sigma * scale,
            y_upper=mu + q * sigma * scale,
        )
        cro_frame["lower_raw"] = mu - q * sigma
        cro_frame["upper_raw"] = mu + q * sigma
        cro_frame["calibration_scale"] = scale
        all_predictions.append(cro_frame)
        metric_rows.append(
            _metric_row(
                cro_frame,
                split=split,
                fold=fold.fold,
                model="cro",
                n_boot=config.n_boot,
                seed=17 + fold.fold,
            )
        )
        baseline_preds = predict_baselines(bundle, train_idx=fold.train_idx, test_idx=fold.test_idx)
        for name, pred in baseline_preds.items():
            frame = _event_frame(bundle, fold.test_idx, pred.y_pred, split=split, fold=fold.fold, model=name)
            all_predictions.append(frame)
            metric_rows.append(
                _metric_row(
                    frame,
                    split=split,
                    fold=fold.fold,
                    model=name,
                    n_boot=config.n_boot,
                    seed=17 + fold.fold,
                )
            )
    (split_dir / f"{split}.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    return pd.concat(all_predictions, ignore_index=True), pd.DataFrame(metric_rows)


def _trials_frame(bundle: Any) -> pd.DataFrame:
    """Reconstruct the minimal grouping columns for split construction."""
    return pd.DataFrame(
        {
            "dataset_id": [bundle.animal_names[i] for i in bundle.animal_idx],
            "target": [bundle.target_names[i] for i in bundle.target_idx],
            "neuron": [bundle.responder_names[i] for i in bundle.responder_idx],
        }
    )


def _pooled_metrics(predictions: pd.DataFrame, *, n_boot: int) -> pd.DataFrame:
    rows = []
    for (split, model), frame in predictions.groupby(["split", "model"]):
        pm = pair_metrics(frame)
        em = event_metrics(frame)
        # Full bootstrap for the primary model; a smaller budget for the many
        # descriptive baselines keeps the run tractable without affecting the
        # preregistered CRO comparison.
        budget = n_boot if model == "cro" else min(200, n_boot)
        ci = bootstrap_pair_spearman(frame, n_boot=budget, seed=17)
        row = {
            "split": split,
            "model": model,
            "n_events": len(frame),
            "n_pairs": pm["n_pairs"],
            "pair_spearman": pm["spearman"],
            "pair_pearson": pm["pearson"],
            "pair_mae": pm["mae"],
            "pair_sign_accuracy": pm["sign_accuracy"],
            "event_mae": em["mae"],
            "event_rmse": em["rmse"],
            "event_spearman": em["spearman"],
            "event_sign_accuracy": em["sign_accuracy"],
            "pair_spearman_ci_low": ci["low"],
            "pair_spearman_ci_high": ci["high"],
        }
        if "lower" in frame.columns:
            from neurobranch.benchmark.metrics import coverage

            row["interval_90_coverage"] = coverage(frame)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["split", "pair_spearman"], ascending=[True, False])


def _paired_delta(
    predictions: pd.DataFrame,
    *,
    split: str,
    model: str,
    baseline: str,
    n_boot: int = 500,
) -> dict[str, Any]:
    cro = predictions[(predictions["split"] == split) & (predictions["model"] == model)]
    base = predictions[(predictions["split"] == split) & (predictions["model"] == baseline)]
    keys = ["event_id", "target", "neuron"]
    merged = cro.merge(
        base[keys + ["y_pred"]],
        on=keys,
        suffixes=("", "_baseline"),
        how="inner",
    )
    if merged.empty:
        return {"delta": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0.0}
    result = paired_pair_bootstrap_delta(
        merged,
        pred_a="y_pred",
        pred_b="y_pred_baseline",
        n_boot=n_boot,
        seed=23,
    )
    result["n"] = float(len(merged))
    return result


def write_phase6_cro(
    *,
    config: BenchmarkConfig | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config or BenchmarkConfig()
    base = Path(root) if root is not None else PROJECT_ROOT
    out = resolve_path(cfg.output_dir, base)
    out.mkdir(parents=True, exist_ok=True)
    trials = load_trials(cfg.trials_path, max_rows=cfg.max_rows)
    bundle = build_feature_bundle(
        trials, connectome_path=cfg.connectome_path, response_column=cfg.response_column
    )
    all_predictions = []
    all_metrics = []
    split_summaries = {}
    for split in cfg.splits:
        preds, metrics = run_split(bundle, split=split, config=cfg, output_dir=out)
        all_predictions.append(preds)
        all_metrics.append(metrics)
        pooled = _pooled_metrics(preds, n_boot=cfg.n_boot)
        split_summaries[split] = pooled.to_dict(orient="records")
    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    pooled = _pooled_metrics(predictions, n_boot=cfg.n_boot)
    predictions.to_parquet(out / "predictions.parquet", index=False)
    metrics.to_csv(out / "metrics_by_fold.csv", index=False)
    pooled.to_csv(out / "metrics_pooled.csv", index=False)

    reliability_summary, reliability_detail = split_half_target_profiles(
        trials, response_column=cfg.response_column, n_splits=20, seed=17
    )
    reliability_detail.to_csv(out / "reliability_by_target.csv", index=False)
    profile_rows = []
    for model in sorted(predictions["model"].unique()):
        profile = attenuation_corrected_profile(predictions, reliability_detail, model=model)
        if profile.empty:
            continue
        profile["model"] = model
        profile_rows.append(profile)
    profile_table = pd.concat(profile_rows, ignore_index=True) if profile_rows else pd.DataFrame()
    profile_table.to_csv(out / "profile_reliability.csv", index=False)
    profile_summary = (
        profile_table.groupby("model")
        .agg(
            n_targets=("target", "nunique"),
            mean_profile_spearman=("profile_spearman", "mean"),
            median_profile_spearman=("profile_spearman", "median"),
            mean_normalized_profile_spearman=("normalized_profile_spearman", "mean"),
        )
        .reset_index()
        if not profile_table.empty
        else pd.DataFrame()
    )
    profile_summary.to_csv(out / "profile_summary.csv", index=False)

    operator_audit = export_operator(
        bundle,
        predictions,
        operator_path=out / "operator.npz",
        uncertainty_path=out / "uncertainty.npz",
    )
    connectivity = evaluate_connectivity(bundle, predictions, output_dir=out / "connectivity", root=base)

    deltas = []
    for split in cfg.splits:
        for baseline in (
            "target_mean",
            "additive",
            "responder_mean",
            "connectome_ridge",
            "context_only",
            "global_mean",
        ):
            d = _paired_delta(predictions, split=split, model="cro", baseline=baseline, n_boot=cfg.n_boot)
            d.update({"split": split, "model": "cro", "baseline": baseline})
            deltas.append(d)
    pd.DataFrame(deltas).to_csv(out / "paired_deltas.csv", index=False)

    audit = {
        "n_rows": int(len(trials)),
        "n_events": int(trials["event_id"].nunique()),
        "n_animals": int(trials["dataset_id"].nunique()),
        "n_targets": int(trials["target"].nunique()),
        "n_responders": int(trials["neuron"].nunique()),
        "n_pairs": int(trials[["target", "neuron"]].drop_duplicates().shape[0]),
        "n_pairs_ge2": int((trials.groupby(["target", "neuron"]).size() >= 2).sum()),
        "response_column": cfg.response_column,
        "connectome_coverage": bundle.connectome_coverage,
        "reliability": reliability_summary.as_dict(),
        "operator": operator_audit,
        "trials_sha256": sha256_file(resolve_path(cfg.trials_path, base)),
        "preregistration_sha256": sha256_file(base / "docs/PHASE6_PREREGISTRATION.md"),
        "git": git_state(base),
        "config": asdict(cfg),
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    summary = {
        "audit": audit,
        "split_summaries": split_summaries,
        "pooled": pooled.to_dict(orient="records"),
        "profile_summary": profile_summary.to_dict(orient="records"),
        "connectivity": connectivity,
        "deltas": deltas,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
