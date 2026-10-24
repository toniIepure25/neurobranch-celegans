"""Paired baseline comparisons for the online benchmark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from neurobranch.behavior.protocols import paired_cluster_bootstrap

OUT = Path("reports/phase5/online_decoding")
pred = pd.read_parquet(OUT / "online_predictions.parquet")
rows = []
for landmark, group in pred.groupby("landmark_s"):
    # Behaviour and prior are the same rows for every learned mode.
    base = group.drop_duplicates("event_uid").set_index("event_uid")
    behavior = group[group["mode"] == "behavior"]
    prior = group[(group["model"] == "prior") & (group["mode"] == "behavior")]
    if not behavior.empty and not prior.empty:
        bw = behavior[behavior.model == "linear"].set_index("event_uid")["p"]
        pw = prior.set_index("event_uid")["p"]
        common = bw.index.intersection(pw.index)
        y = base.loc[common, "y"].to_numpy()
        animals = base.loc[common, "animal_id"].to_numpy()
        for metric in ("log_loss", "auroc", "brier", "balanced_accuracy"):
            res = paired_cluster_bootstrap(
                y, pw.loc[common].to_numpy(), bw.loc[common].to_numpy(), animals,
                metric=metric, n_boot=400, seed=17,
            )
            rows.append({"landmark_s": landmark, "comparison": "behavior_linear_vs_prior", "metric": metric, **res})
    sensory = group[(group["mode"] == "behavior_sensory") & (group["model"] == "linear")]
    if not sensory.empty:
        sw = sensory.set_index("event_uid")["p"]
        bw = behavior[behavior.model == "linear"].set_index("event_uid")["p"]
        common = sw.index.intersection(bw.index)
        y = base.loc[common, "y"].to_numpy()
        animals = base.loc[common, "animal_id"].to_numpy()
        for metric in ("log_loss", "auroc", "brier", "balanced_accuracy"):
            res = paired_cluster_bootstrap(
                y, bw.loc[common].to_numpy(), sw.loc[common].to_numpy(), animals,
                metric=metric, n_boot=400, seed=17,
            )
            rows.append({"landmark_s": landmark, "comparison": "behavior_sensory_vs_behavior", "metric": metric, **res})
pd.DataFrame(rows).to_csv(OUT / "online_baselines.csv", index=False)
print("wrote", OUT / "online_baselines.csv")
