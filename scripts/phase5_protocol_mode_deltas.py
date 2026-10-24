"""Mode-specific (SAAV vs behaviour) offline deltas without refitting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from neurobranch.behavior.protocols import paired_cluster_bootstrap

OUT = Path("reports/phase5/protocol_comparison")
pred = pd.read_parquet(OUT / "protocol_event_predictions.parquet")
pred = pred[pred["n_inner_predictions"] > 0]
rows = []
for regime, group in pred.groupby("regime"):
    wide = group.pivot_table(index="event_uid", columns="mode", values="p", aggfunc="first")
    base = group.drop_duplicates("event_uid").set_index("event_uid")
    for comparator in ("saav", "both"):
        if comparator not in wide.columns or "behavior" not in wide.columns:
            continue
        common = wide.dropna(subset=["behavior", comparator]).index
        y = base.loc[common, "y"].to_numpy()
        animals = base.loc[common, "animal_id"].to_numpy()
        for metric in ("log_loss", "auroc", "brier"):
            res = paired_cluster_bootstrap(
                y,
                wide.loc[common, "behavior"].to_numpy(),
                wide.loc[common, comparator].to_numpy(),
                animals,
                metric=metric,
                n_boot=1000,
                seed=17,
            )
            rows.append(
                {
                    "regime": regime,
                    "comparison": f"{comparator}_vs_behavior",
                    "metric": metric,
                    "n": int(len(common)),
                    "n_animals": int(len(set(animals))),
                    **res,
                }
            )
pd.DataFrame(rows).to_csv(OUT / "protocol_mode_deltas.csv", index=False)
print("wrote mode deltas")
