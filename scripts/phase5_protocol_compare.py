"""Phase 5B: compare original-style, animal-disjoint, chronological and external protocols.

This script uses the phase-aligned raw-Dryad reconstruction and an
architecture-compatible PyTorch GRU.  It does not claim exact reproduction of
the published JAX/Equinox run.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurobranch.behavior.protocols import (
    OFFLINE_MODES,
    animal_overlap,
    event_metrics,
    flatten_sections,
    make_folds,
    make_sections,
    paired_cluster_bootstrap,
    prior_shift,
    summarize_by_animal,
)
from neurobranch.behavior.saav_alignment import AlignedDataset, build_aligned_dataset
from neurobranch.data.kramer import load_all
from neurobranch.models.small_rnn import RNNConfig, predict_rnn, train_rnn

OUT = Path("reports/phase5/protocol_comparison")
DEFAULT_REGIMES = ("original", "grouped", "chronological", "external")
MODE_CHANNELS = {
    "behavior": ("hc",),
    "saav": ("saav",),
    "both": ("hc", "saav"),
}


def _mode_traces(data: AlignedDataset, mode: str) -> np.ndarray:
    """Two-channel official-style input; absent channel is exactly zero."""
    hc = data.channel("hc")
    saav = data.channel("saav")
    zero = np.zeros_like(hc)
    if mode == "behavior":
        return np.stack([hc, zero], axis=-1).astype(np.float32)
    if mode == "saav":
        return np.stack([zero, saav], axis=-1).astype(np.float32)
    if mode == "both":
        return np.stack([hc, saav], axis=-1).astype(np.float32)
    raise ValueError(f"unknown mode {mode!r}")


def _section_accuracy(section_y: np.ndarray, section_p: np.ndarray, section_ids: np.ndarray) -> float:
    frame = pd.DataFrame({"y": section_y, "p": section_p, "sid": section_ids})
    per_section = frame.groupby("sid", sort=False).agg({"y": "first", "p": "mean"})
    if per_section["y"].nunique() < 2:
        return float("nan")
    correct = (per_section["p"] >= 0.5).astype(int) == per_section["y"].astype(int)
    return float(correct.groupby(per_section["y"]).mean().mean())


def evaluate_protocol(
    data: AlignedDataset,
    regime: str,
    mode: str,
    *,
    epochs: int,
    max_folds: int | None,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    traces = _mode_traces(data, mode)
    s = make_sections(traces, section_frames=4)
    n_events, n_sections = s.shape[:2]
    event_ids = np.arange(n_events)
    folds = make_folds(data, regime, seed=seed)  # type: ignore[arg-type]
    if max_folds is not None:
        folds = folds[:max_folds]
    event_sum = np.zeros(n_events, dtype=float)
    event_count = np.zeros(n_events, dtype=int)
    section_sum = np.zeros(n_events * n_sections, dtype=float)
    section_count = np.zeros(n_events * n_sections, dtype=int)
    fold_rows: list[dict[str, Any]] = []
    started = time.time()

    for fold in folds:
        xtr, ytr, gtr = flatten_sections(s, data.y, event_ids, fold.train)
        xva, yva, gva = flatten_sections(s, data.y, event_ids, fold.valid)
        xte, yte, gte = flatten_sections(s, data.y, event_ids, fold.test)
        cfg = RNNConfig(epochs=epochs, patience=8, batch_size=64, seed=seed + 100 * fold.outer + fold.inner)
        model, info = train_rnn(xtr, ytr, gtr, xva, yva, gva, config=cfg)
        pte, gte_pred = predict_rnn(model, xte, gte)
        # Correct the class-balanced training prior using training prevalence.
        pte = prior_shift(pte, float(np.mean(ytr)))
        # Event prediction: mean over all four-frame sections and timesteps.
        for event_idx in fold.test:
            mask = gte_pred == event_idx
            if np.any(mask):
                event_sum[event_idx] += float(np.mean(pte[mask]))
                event_count[event_idx] += 1
        # Section prediction: keep one value per (event, section).
        flat_offset = np.repeat(fold.test * n_sections, n_sections)
        flat_offset = flat_offset + np.tile(np.arange(n_sections), len(fold.test))
        section_sum[flat_offset] += pte
        section_count[flat_offset] += 1
        overlap = animal_overlap(data.meta, fold.train, fold.test)
        fold_rows.append(
            {
                "regime": regime,
                "mode": mode,
                "outer": fold.outer,
                "inner": fold.inner,
                "n_train_events": int(len(fold.train)),
                "n_valid_events": int(len(fold.valid)),
                "n_test_events": int(len(fold.test)),
                "n_train_animals": overlap["n_train_animals"],
                "n_test_animals": overlap["n_test_animals"],
                "n_overlap_animals": overlap["n_overlap_animals"],
                "fraction_test_events_with_train_animal": overlap["fraction_test_events_with_train_animal"],
                "best_epoch": info["best_epoch"],
                "best_valid_loss": info["best_valid_loss"],
                "epochs_run": info["epochs_run"],
            }
        )

    event_p = np.divide(event_sum, event_count, out=np.full(n_events, np.nan), where=event_count > 0)
    section_p = np.divide(
        section_sum,
        section_count,
        out=np.full(n_events * n_sections, np.nan),
        where=section_count > 0,
    )
    section_available = section_count > 0
    section_y = np.repeat(data.y, n_sections)
    section_ids = np.arange(n_events * n_sections)
    tested = event_count > 0
    metrics = event_metrics(data.y[tested], event_p[tested])
    metrics["section_accuracy"] = _section_accuracy(
        section_y[section_available], section_p[section_available], section_ids[section_available]
    )
    metrics["regime"] = regime
    metrics["mode"] = mode
    metrics["n_animals"] = int(data.meta["animal_id"].nunique())
    metrics["runtime_s"] = float(time.time() - started)
    metrics["event_coverage"] = float(np.mean(event_count > 0))
    metrics["mean_inner_models"] = float(event_count[event_count > 0].mean()) if np.any(event_count > 0) else 0.0

    pred = data.meta[["animal_id", "cohort", "event_index", "start", "end", "direction"]].copy()
    pred["event_uid"] = np.arange(n_events)
    pred["y"] = data.y
    pred["p"] = event_p
    pred["regime"] = regime
    pred["mode"] = mode
    pred["n_inner_predictions"] = event_count

    sec_frame = data.meta[["animal_id", "cohort", "event_index"]].copy()
    sec_frame = sec_frame.loc[np.repeat(np.arange(n_events), n_sections)].reset_index(drop=True)
    sec_frame["event_uid"] = np.repeat(np.arange(n_events), n_sections)
    sec_frame["section_index"] = np.tile(np.arange(n_sections), n_events)
    sec_frame["y"] = section_y
    sec_frame["p"] = section_p
    sec_frame["available"] = section_available
    sec_frame["regime"] = regime
    sec_frame["mode"] = mode
    folds_frame = pd.DataFrame(fold_rows)
    return metrics, pred, sec_frame, folds_frame


def _add_deltas(
    predictions: pd.DataFrame,
    sections: pd.DataFrame,
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for regime, group in predictions.groupby("regime"):
        group = group[group["n_inner_predictions"] > 0].copy()
        wide = group.pivot_table(index="event_uid", columns="mode", values="p", aggfunc="first")
        base = group.drop_duplicates("event_uid").set_index("event_uid")
        if not {"behavior", "both"}.issubset(wide.columns):
            continue
        common = wide.dropna(subset=["behavior", "both"]).index
        y = base.loc[common, "y"].to_numpy()
        groups = base.loc[common, "animal_id"].to_numpy()
        for metric in ("auroc", "log_loss", "brier", "balanced_accuracy"):
            res = paired_cluster_bootstrap(
                y,
                wide.loc[common, "behavior"].to_numpy(),
                wide.loc[common, "both"].to_numpy(),
                groups,
                metric=metric,
                seed=seed,
            )
            rows.append({"regime": regime, "comparison": "both_vs_behavior", "metric": metric, **res})
        # Section-level official metric delta, clustered by animal.
        sec = sections[
            (sections["regime"] == regime)
            & sections["available"]
            & sections["mode"].isin(["behavior", "both"])
        ]
        sec_wide = sec.pivot_table(index=["event_uid", "section_index"], columns="mode", values="p")
        sec_meta = sec.drop_duplicates(["event_uid", "section_index"]).set_index(
            ["event_uid", "section_index"]
        )
        common_sec = sec_wide.dropna(subset=["behavior", "both"]).index
        y_sec = sec_meta.loc[common_sec, "y"].to_numpy()
        a_sec = sec_wide.loc[common_sec, "behavior"].to_numpy()
        b_sec = sec_wide.loc[common_sec, "both"].to_numpy()
        g_sec = sec_meta.loc[common_sec, "animal_id"].to_numpy()
        sec_res = paired_cluster_bootstrap(
            y_sec,
            a_sec,
            b_sec,
            g_sec,
            metric="balanced_accuracy",
            n_boot=1000,
            seed=seed,
        )
        rows.append({"regime": regime, "comparison": "both_vs_behavior_sections", "metric": "balanced_accuracy", **sec_res})
        log_row = next(
            r for r in rows if r["regime"] == regime and r["metric"] == "log_loss"
        )
        summary[regime] = {
            "n_events": int(len(common)),
            "n_animals": int(len(np.unique(groups))),
            "classification_log_loss": _classify(log_row),
            "classification_auroc": _classify_auroc(
                next(r for r in rows if r["regime"] == regime and r["metric"] == "auroc")
            ),
            "log_loss_delta": log_row["delta"],
            "log_loss_ci": [log_row["ci_low"], log_row["ci_high"]],
        }
    return rows, summary


def _classify(delta: dict[str, Any]) -> str:
    """Classify a proper-scoring delta (negative log loss is better)."""
    if not np.isfinite(delta.get("ci_low", np.nan)) or not np.isfinite(delta.get("ci_high", np.nan)):
        return "not_estimable"
    if delta["ci_high"] < 0:
        return "positive_increment"
    if delta["ci_low"] > 0:
        return "negative_increment"
    return "no_incremental_evidence"


def _classify_auroc(delta: dict[str, Any]) -> str:
    if not np.isfinite(delta.get("ci_low", np.nan)) or not np.isfinite(delta.get("ci_high", np.nan)):
        return "not_estimable"
    if delta["ci_low"] > 0:
        return "positive_increment"
    if delta["ci_high"] < 0:
        return "negative_increment"
    return "no_incremental_evidence"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--regimes", nargs="+", default=list(DEFAULT_REGIMES))
    parser.add_argument("--modes", nargs="+", default=list(OFFLINE_MODES))
    args = parser.parse_args()
    torch_threads = max(1, min(4, os.cpu_count() or 1))
    import torch

    torch.set_num_threads(torch_threads)
    OUT.mkdir(parents=True, exist_ok=True)
    recordings, audit = load_all()
    data, recon_audit = build_aligned_dataset(recordings)
    recon_audit.to_csv(OUT / "reconstruction_audit.csv", index=False)
    data.meta.to_csv(OUT / "reconstruction_events.csv", index=False)
    summary = {
        "status": "PARTIAL_REPRODUCTION",
        "n_recordings_loaded": len(recordings),
        "n_events": data.n_events,
        "n_animals": int(data.meta["animal_id"].nunique()),
        "n_dorsal": int(data.y.sum()),
        "n_ventral": int(len(data.y) - data.y.sum()),
        "cohorts": data.meta.groupby("cohort").size().to_dict(),
        "channels": list(data.channels),
        "frames": data.n_frames,
        "epochs": args.epochs,
        "seed": args.seed,
        "torch_threads": torch_threads,
        "regimes": args.regimes,
        "modes": args.modes,
        "reconstruction_exclusions": recon_audit["reason"].value_counts().to_dict(),
    }
    (OUT / "reconstruction_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    all_metrics: list[dict[str, Any]] = []
    all_pred: list[pd.DataFrame] = []
    all_sec: list[pd.DataFrame] = []
    all_folds: list[pd.DataFrame] = []
    for regime in args.regimes:
        for mode in args.modes:
            metrics, pred, sec, folds = evaluate_protocol(
                data,
                regime,
                mode,
                epochs=args.epochs,
                max_folds=args.max_folds,
                seed=args.seed,
            )
            all_metrics.append(metrics)
            all_pred.append(pred)
            all_sec.append(sec)
            all_folds.append(folds)
            print(json.dumps(metrics, indent=2), flush=True)
    metrics_frame = pd.DataFrame(all_metrics)
    pred_frame = pd.concat(all_pred, ignore_index=True)
    sec_frame = pd.concat(all_sec, ignore_index=True)
    folds_frame = pd.concat(all_folds, ignore_index=True)
    metrics_frame.to_csv(OUT / "protocol_metrics.csv", index=False)
    pred_frame.to_parquet(OUT / "protocol_event_predictions.parquet", index=False)
    sec_frame.to_parquet(OUT / "protocol_section_predictions.parquet", index=False)
    folds_frame.to_csv(OUT / "protocol_folds.csv", index=False)
    deltas, delta_summary = _add_deltas(pred_frame, sec_frame, seed=args.seed)
    pd.DataFrame(deltas).to_csv(OUT / "protocol_deltas.csv", index=False)
    (OUT / "protocol_delta_summary.json").write_text(
        json.dumps(delta_summary, indent=2) + "\n", encoding="utf-8"
    )
    per_animal = pd.concat(
        [
            summarize_by_animal(
                group[["animal_id", "cohort"]],
                group["y"].to_numpy(),
                group["p"].to_numpy(),
                mode=f"{regime}:{mode}",
            )
            for (regime, mode), group in pred_frame.groupby(["regime", "mode"])
        ],
        ignore_index=True,
    )
    per_animal.to_csv(OUT / "protocol_per_animal.csv", index=False)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
