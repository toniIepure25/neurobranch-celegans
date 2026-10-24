"""Finalize online reports after the expensive model fits have completed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import phase5_online as p5o  # noqa: E402

from neurobranch.behavior.online_turning import LANDMARKS_S, build_online_dataset  # noqa: E402
from neurobranch.behavior.protocols import event_metrics, paired_cluster_bootstrap, prior_shift  # noqa: E402
from neurobranch.data.kramer import load_all  # noqa: E402

OUT = Path("reports/phase5/online_decoding")


def classify_log_loss(result: dict) -> str:
    low, high = result.get("ci_low", np.nan), result.get("ci_high", np.nan)
    if not np.isfinite(low) or not np.isfinite(high):
        return "not_estimable"
    if high < 0:
        return "positive_increment"
    if low > 0:
        return "negative_increment"
    return "no_incremental_evidence"


def main() -> None:
    pred = pd.read_parquet(OUT / "online_predictions.parquet")
    rows = []
    summary = {}
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
                    n_boot=400,
                    seed=17,
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
                r for r in rows
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
                "classification": classify_log_loss(log_res),
            }
    pd.DataFrame(rows).to_csv(OUT / "online_deltas.csv", index=False)
    (OUT / "online_delta_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Behaviour-only sensitivity on all WT events at every landmark.
    recordings, _ = load_all()
    raw, _ = build_online_dataset(recordings, landmarks_s=LANDMARKS_S)
    behavior_rows = []
    behavior_preds = []
    for landmark in LANDMARKS_S:
        sub = raw.subset(raw.meta["landmark_s"].to_numpy() == landmark)
        if sub.n_samples == 0:
            continue
        x, finite, _ = p5o._select_columns(sub, "behavior")
        test_folds = p5o._animal_folds(sub.meta, sub.y, n_splits=5, seed=17)
        preds = []
        for fold_idx, test in enumerate(test_folds):
            test = test[finite[test]]
            train_pool = np.flatnonzero(finite & ~np.isin(np.arange(sub.n_samples), test))
            if len(test) < 2 or len(np.unique(sub.y[train_pool])) < 2:
                continue
            train, _ = p5o._valid_split(train_pool, sub.y, seed=17 + fold_idx)
            mean, sd = p5o._scale_fit(x[train])
            p = p5o._fit_linear(
                p5o._scale_apply(x[train], mean, sd),
                sub.y[train],
                p5o._scale_apply(x[test], mean, sd),
                seed=17,
            )
            p = prior_shift(p, float(np.mean(sub.y[train])))
            frame = sub.meta.iloc[test][["animal_id", "cohort", "event_uid", "landmark_s", "direction"]].copy()
            frame["y"] = sub.y[test]
            frame["p"] = p
            preds.append(frame)
        if preds:
            joined = pd.concat(preds, ignore_index=True)
            behavior_preds.append(joined)
            behavior_rows.append(
                {"landmark_s": landmark, "mode": "behavior", "model": "linear", **event_metrics(joined["y"].to_numpy(), joined["p"].to_numpy())}
            )
    pd.DataFrame(behavior_rows).to_csv(OUT / "online_behavior_all.csv", index=False)
    if behavior_preds:
        pd.concat(behavior_preds, ignore_index=True).to_parquet(OUT / "online_behavior_all_predictions.parquet", index=False)

    primary = raw.subset(raw.meta["saav_available"].to_numpy(dtype=bool))
    config = {
        "status": "CAUSAL_INDEXED_ONLINE_BENCHMARK",
        "onset_mode": "state_rising_edge",
        "landmarks_s": list(LANDMARKS_S),
        "history_s": 7.2,
        "min_history_s": 3.6,
        "n_raw_samples": raw.n_samples,
        "n_primary_samples": primary.n_samples,
        "n_primary_events": int(primary.meta["event_uid"].nunique()),
        "n_primary_animals": int(primary.meta["animal_id"].nunique()),
        "n_dorsal_samples": int(primary.y.sum()),
        "n_ventral_samples": int(len(primary.y) - primary.y.sum()),
        "epochs": 40,
        "seed": 17,
        "bootstrap_resamples": 400,
        "normalization": "train-fold channel mean/SD; no future frames",
        "evaluation": "animal-disjoint 5-fold; event-level proper scoring",
    }
    (OUT / "online_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print("finalized", OUT)


if __name__ == "__main__":
    main()
