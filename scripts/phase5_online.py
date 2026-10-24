"""Phase 5C: genuinely causal online-prospective turn-direction prediction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from neurobranch.behavior.online_turning import (
    BEHAVIOR_CHANNELS,
    LANDMARKS_S,
    SENSORY_CHANNELS,
    OnlineDataset,
    build_online_dataset,
    excluded_reason_summary,
    risk_set_summary,
)
from neurobranch.behavior.protocols import event_metrics, paired_cluster_bootstrap, prior_shift
from neurobranch.data.kramer import load_all
from neurobranch.models.small_rnn import RNNConfig, predict_rnn, train_rnn

OUT = Path("reports/phase5/online_decoding")
MODE_CHANNELS: dict[str, tuple[str, ...]] = {
    "behavior": BEHAVIOR_CHANNELS,
    "behavior_sensory": (*BEHAVIOR_CHANNELS, *SENSORY_CHANNELS),
    "behavior_saav": (*BEHAVIOR_CHANNELS, "saav"),
    # Only identities with usable coverage in the SAAV cohort are included.
    # Missing values are imputed after train-fold scaling and disclosed.
    "behavior_circuit": (
        *BEHAVIOR_CHANNELS,
        "neuron:SAAV",
        "neuron:SMDD",
        "neuron:SMBV",
        "neuron:SMBD",
    ),
    "behavior_population": (*BEHAVIOR_CHANNELS, "population_mean", "population_std"),
}
RNN_MODES = ("behavior", "behavior_saav", "behavior_circuit", "behavior_population")


def _select_columns(data: OnlineDataset, mode: str) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    names = MODE_CHANNELS[mode]
    missing = [n for n in names if n not in data.channels]
    if missing:
        raise ValueError(f"dataset lacks channels {missing}")
    x = np.stack([data.channel(n) for n in names], axis=-1)
    finite = np.isfinite(x).all(axis=(1, 2))
    return x, finite, names


def _summary_features(x: np.ndarray) -> np.ndarray:
    """Per-sequence summary features for the linear baseline."""
    if x.ndim != 3:
        raise ValueError("x must be samples × frames × channels")
    n, t, c = x.shape
    out = np.empty((n, 6 * c), dtype=float)
    tt = np.arange(t, dtype=float)
    tt = (tt - tt.mean()) / max(float(np.std(tt)), 1e-9)
    for k in range(c):
        v = x[:, :, k]
        out[:, 6 * k + 0] = v.mean(axis=1)
        out[:, 6 * k + 1] = v.std(axis=1)
        out[:, 6 * k + 2] = v[:, -1]
        out[:, 6 * k + 3] = v.min(axis=1)
        out[:, 6 * k + 4] = v.max(axis=1)
        out[:, 6 * k + 5] = (v * tt).mean(axis=1)
    return out


def _fit_linear(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    scaler = StandardScaler().fit(_summary_features(x_train))
    model = LogisticRegression(
        C=0.1,
        max_iter=3000,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(scaler.transform(_summary_features(x_train)), y_train)
    return model.predict_proba(scaler.transform(_summary_features(x_test)))[:, 1]


def _animal_folds(meta: pd.DataFrame, y: np.ndarray, *, n_splits: int, seed: int) -> list[np.ndarray]:
    """Greedy animal folds balancing dorsal-event counts, not outcomes at test."""
    frame = pd.DataFrame({"animal_id": meta["animal_id"].astype(str).to_numpy(), "y": y})
    counts = frame.groupby("animal_id").agg(n=("y", "size"), dorsal=("y", "sum")).reset_index()
    counts = counts.sort_values(["dorsal", "n"], ascending=False, kind="mergesort")
    fold_dorsal = np.zeros(n_splits)
    fold_n = np.zeros(n_splits)
    assignment: dict[str, int] = {}
    for row in counts.itertuples(index=False):
        k = int(np.lexsort((fold_n, fold_dorsal))[0])
        assignment[row.animal_id] = k
        fold_dorsal[k] += row.dorsal
        fold_n[k] += row.n
    test_folds = [np.flatnonzero(frame["animal_id"].isin([a for a, k in assignment.items() if k == j])) for j in range(n_splits)]
    return test_folds


def _valid_split(train_idx: np.ndarray, y: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    valid_parts = []
    train_parts = []
    for label in (0, 1):
        idx = train_idx[y[train_idx] == label]
        idx = idx[rng.permutation(len(idx))]
        n_valid = max(1, int(round(0.2 * len(idx))))
        valid_parts.append(idx[:n_valid])
        train_parts.append(idx[n_valid:])
    return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(valid_parts))


def _scale_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = x.reshape(-1, x.shape[-1])
    mean = np.nanmean(flat, axis=0)
    sd = np.nanstd(flat, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
    return mean, sd


def _scale_apply(x: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    z = (x - mean) / sd
    return np.clip(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), -8.0, 8.0)


def evaluate_landmark(
    data: OnlineDataset,
    landmark: float,
    *,
    seed: int,
    epochs: int,
    run_rnn_modes: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], pd.DataFrame]:
    group = data.meta["landmark_s"].to_numpy() == landmark
    sub = data.subset(group)
    if sub.n_samples == 0:
        return [], [], pd.DataFrame()
    y = sub.y
    animals = sub.meta["animal_id"].astype(str).to_numpy()
    event_groups = sub.meta["event_uid"].astype(str).to_numpy()
    test_folds = _animal_folds(sub.meta, y, n_splits=5, seed=seed)
    rows: list[dict[str, Any]] = []
    preds: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []

    for mode in ("behavior", "behavior_sensory", "behavior_saav", "behavior_circuit", "behavior_population"):
        x_raw, finite, names = _select_columns(sub, mode)
        if mode in {"behavior_saav", "behavior_circuit", "behavior_population"}:
            finite &= sub.meta["saav_available"].to_numpy(dtype=bool)
        if mode == "behavior_circuit":
            # SAAV is present for every primary sample; other identities are
            # allowed to be missing and are imputed after scaling.
            finite = np.isfinite(x_raw[:, :, 0]).all(axis=1) & sub.meta[
                "saav_available"
            ].to_numpy(dtype=bool)
        usable = finite
        if usable.sum() < 20 or len(np.unique(y[usable])) < 2:
            continue
        for fold_idx, test in enumerate(test_folds):
            test = test[usable[test]]
            if len(test) == 0:
                continue
            train_pool = np.flatnonzero(usable & ~np.isin(np.arange(sub.n_samples), test))
            if len(np.unique(y[train_pool])) < 2:
                continue
            train, valid = _valid_split(train_pool, y, seed=seed + 31 * fold_idx)
            if len(np.unique(y[train])) < 2 or len(np.unique(y[valid])) < 2:
                continue
            mean, sd = _scale_fit(x_raw[train])
            xtr = _scale_apply(x_raw[train], mean, sd)
            xva = _scale_apply(x_raw[valid], mean, sd)
            xte = _scale_apply(x_raw[test], mean, sd)
            # Prior uses the training prevalence only.
            prior = float(np.mean(y[train]))
            for model_name in ("prior", "linear"):
                if model_name == "prior":
                    p = np.full(len(test), prior)
                else:
                    p = _fit_linear(xtr, y[train], xte, seed=seed + fold_idx)
                    p = prior_shift(p, float(np.mean(y[train])))
                frame = sub.meta.iloc[test][
                    ["animal_id", "cohort", "event_uid", "landmark_s", "prediction_frame", "end", "direction"]
                ].copy()
                frame["lead_time_s"] = sub.meta.iloc[test]["lead_time_s"].to_numpy()
                frame["y"] = y[test]
                frame["p"] = p
                frame["mode"] = mode
                frame["model"] = model_name
                preds.append(frame)
            fold_rows.append(
                {
                    "landmark_s": landmark,
                    "mode": mode,
                    "fold": fold_idx,
                    "n_train": len(train),
                    "n_valid": len(valid),
                    "n_test": len(test),
                    "n_test_animals": int(len(np.unique(animals[test]))),
                    "prior": prior,
                }
            )
            if mode in run_rnn_modes:
                cfg = RNNConfig(epochs=epochs, patience=8, batch_size=64, seed=seed + 1000 + fold_idx)
                model, info = train_rnn(
                    xtr,
                    y[train],
                    event_groups[train],
                    xva,
                    y[valid],
                    event_groups[valid],
                    sequence_pool="last",
                    config=cfg,
                )
                p_rnn, _ = predict_rnn(
                    model,
                    xte,
                    event_groups[test],
                    sequence_pool="last",
                )
                p_rnn = prior_shift(p_rnn, float(np.mean(y[train])))
                frame = sub.meta.iloc[test][
                    ["animal_id", "cohort", "event_uid", "landmark_s", "prediction_frame", "end", "direction"]
                ].copy()
                frame["lead_time_s"] = sub.meta.iloc[test]["lead_time_s"].to_numpy()
                frame["y"] = y[test]
                frame["p"] = p_rnn
                frame["mode"] = mode
                frame["model"] = "rnn"
                preds.append(frame)
                fold_rows[-1]["rnn_best_epoch"] = info["best_epoch"]
                fold_rows[-1]["rnn_valid_loss"] = info["best_valid_loss"]

    pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    if pred.empty:
        return rows, preds, pd.DataFrame(fold_rows)
    for (mode, model_name), group_pred in pred.groupby(["mode", "model"]):
        m = event_metrics(group_pred["y"].to_numpy(), group_pred["p"].to_numpy())
        m.update({"landmark_s": landmark, "mode": mode, "model": model_name})
        rows.append(m)
    return rows, preds, pd.DataFrame(fold_rows)


def add_online_deltas(pred: pd.DataFrame, *, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    summary: dict[str, Any] = {}
    for (landmark, model_name), group in pred.groupby(["landmark_s", "model"]):
        wide = group.pivot_table(index="event_uid", columns="mode", values="p", aggfunc="first")
        base = group.drop_duplicates("event_uid").set_index("event_uid")
        if "behavior" not in wide.columns:
            continue
        for comparator in ("behavior_saav", "behavior_circuit", "behavior_population"):
            if comparator not in wide.columns:
                continue
            common = wide.dropna(subset=["behavior", comparator]).index
            if len(common) < 10:
                continue
            y = base.loc[common, "y"].to_numpy()
            animals = base.loc[common, "animal_id"].to_numpy()
            for metric in ("log_loss", "auroc", "brier", "balanced_accuracy"):
                res = paired_cluster_bootstrap(
                    y,
                    wide.loc[common, "behavior"].to_numpy(),
                    wide.loc[common, comparator].to_numpy(),
                    animals,
                    metric=metric,
                    n_boot=1000,
                    seed=seed,
                )
                rows.append(
                    {
                        "landmark_s": landmark,
                        "model": model_name,
                        "comparison": f"{comparator}_vs_behavior",
                        "metric": metric,
                        **res,
                    }
                )
            log_res = next(
                r
                for r in rows
                if r["landmark_s"] == landmark
                and r["model"] == model_name
                and r["comparison"] == f"{comparator}_vs_behavior"
                and r["metric"] == "log_loss"
            )
            summary[f"{landmark}:{model_name}:{comparator}"] = {
                "n_events": int(len(common)),
                "n_animals": int(len(np.unique(animals))),
                "log_loss_delta": log_res["delta"],
                "log_loss_ci": [log_res["ci_low"], log_res["ci_high"]],
                "classification": _classify_log_loss(log_res),
            }
    return rows, summary


def _classify_log_loss(result: dict[str, Any]) -> str:
    low, high = result.get("ci_low", np.nan), result.get("ci_high", np.nan)
    if not np.isfinite(low) or not np.isfinite(high):
        return "not_estimable"
    if high < 0:
        return "positive_increment"
    if low > 0:
        return "negative_increment"
    return "no_incremental_evidence"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--landmarks", nargs="+", type=float, default=list(LANDMARKS_S))
    parser.add_argument("--rnn-modes", nargs="+", default=list(RNN_MODES))
    args = parser.parse_args()
    import torch

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    OUT.mkdir(parents=True, exist_ok=True)
    recordings, _ = load_all()
    raw, audit = build_online_dataset(recordings, landmarks_s=tuple(args.landmarks))
    audit.to_csv(OUT / "online_audit.csv", index=False)
    risk_set_summary(audit).to_csv(OUT / "online_risk_sets.csv", index=False)
    excluded_reason_summary(audit).to_csv(OUT / "online_exclusions.csv", index=False)
    # Primary common cohort: events with an unambiguous SAAV identity, so every
    # neural and behaviour baseline is compared on exactly the same events.
    primary = raw.subset(raw.meta["saav_available"].to_numpy(dtype=bool))
    primary.meta.to_csv(OUT / "online_primary_samples.csv", index=False)
    all_metrics: list[dict[str, Any]] = []
    all_pred: list[pd.DataFrame] = []
    all_folds: list[pd.DataFrame] = []
    for landmark in args.landmarks:
        metrics, preds, folds = evaluate_landmark(
            primary,
            float(landmark),
            seed=args.seed,
            epochs=args.epochs,
            run_rnn_modes=tuple(args.rnn_modes),
        )
        all_metrics.extend(metrics)
        all_pred.extend(preds)
        all_folds.append(folds)
    metrics_frame = pd.DataFrame(all_metrics)
    pred_frame = pd.concat(all_pred, ignore_index=True) if all_pred else pd.DataFrame()
    folds_frame = pd.concat(all_folds, ignore_index=True) if all_folds else pd.DataFrame()
    metrics_frame.to_csv(OUT / "online_metrics.csv", index=False)
    pred_frame.to_parquet(OUT / "online_predictions.parquet", index=False)
    folds_frame.to_csv(OUT / "online_folds.csv", index=False)
    deltas, delta_summary = add_online_deltas(pred_frame, seed=args.seed)
    pd.DataFrame(deltas).to_csv(OUT / "online_deltas.csv", index=False)
    (OUT / "online_delta_summary.json").write_text(json.dumps(delta_summary, indent=2) + "\n", encoding="utf-8")
    # Behaviour-only sensitivity across all WT events.
    behavior_rows = []
    behavior_preds = []
    for landmark in args.landmarks:
        sub = raw.subset(raw.meta["landmark_s"].to_numpy() == landmark)
        if sub.n_samples == 0:
            continue
        x, finite, _ = _select_columns(sub, "behavior")
        test_folds = _animal_folds(sub.meta, sub.y, n_splits=5, seed=args.seed)
        preds = []
        for fold_idx, test in enumerate(test_folds):
            test = test[finite[test]]
            train_pool = np.flatnonzero(finite & ~np.isin(np.arange(sub.n_samples), test))
            if len(test) < 2 or len(np.unique(sub.y[train_pool])) < 2:
                continue
            train, _ = _valid_split(train_pool, sub.y, seed=args.seed + fold_idx)
            mean, sd = _scale_fit(x[train])
            p = _fit_linear(_scale_apply(x[train], mean, sd), sub.y[train], _scale_apply(x[test], mean, sd), seed=args.seed)
            p = prior_shift(p, float(np.mean(sub.y[train])))
            frame = sub.meta.iloc[test][["animal_id", "cohort", "event_uid", "landmark_s", "direction"]].copy()
            frame["y"] = sub.y[test]
            frame["p"] = p
            preds.append(frame)
        if preds:
            behavior_preds.append(pd.concat(preds, ignore_index=True))
            behavior_rows.append({"landmark_s": landmark, "mode": "behavior", "model": "linear", **event_metrics(
                pd.concat(preds)["y"].to_numpy(), pd.concat(preds)["p"].to_numpy()
            )})
    pd.DataFrame(behavior_rows).to_csv(OUT / "online_behavior_all.csv", index=False)
    if behavior_preds:
        pd.concat(behavior_preds, ignore_index=True).to_parquet(OUT / "online_behavior_all_predictions.parquet", index=False)

    config = {
        "status": "CAUSAL_INDEXED_ONLINE_BENCHMARK",
        "onset_mode": "state_rising_edge",
        "landmarks_s": args.landmarks,
        "history_s": 7.2,
        "min_history_s": 3.6,
        "n_raw_samples": raw.n_samples,
        "n_primary_samples": primary.n_samples,
        "n_primary_events": int(primary.meta["event_uid"].nunique()),
        "n_primary_animals": int(primary.meta["animal_id"].nunique()),
        "n_dorsal_samples": int(primary.y.sum()),
        "n_ventral_samples": int(len(primary.y) - primary.y.sum()),
        "epochs": args.epochs,
        "seed": args.seed,
        "rnn_modes": args.rnn_modes,
        "mode_channels": {k: list(v) for k, v in MODE_CHANNELS.items()},
        "normalization": "train-fold channel mean/SD; no future frames",
        "evaluation": "animal-disjoint 5-fold; event-level proper scoring",
    }
    (OUT / "online_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
