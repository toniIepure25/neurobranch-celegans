"""Finalize offline per-animal and prevalence-reference tables without refitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neurobranch.behavior.protocols import summarize_by_animal

OUT = Path("reports/phase5/protocol_comparison")
pred = pd.read_parquet(OUT / "protocol_event_predictions.parquet")
pred = pred[pred["n_inner_predictions"] > 0].copy()
parts = []
for (regime, mode), group in pred.groupby(["regime", "mode"]):
    parts.append(
        summarize_by_animal(
            group[["animal_id", "cohort"]],
            group["y"].to_numpy(),
            group["p"].to_numpy(),
            mode=f"{regime}:{mode}",
        )
    )
pd.concat(parts, ignore_index=True).to_csv(OUT / "protocol_per_animal.csv", index=False)

rows = []
for (regime, mode), group in pred.groupby(["regime", "mode"]):
    y = group["y"].to_numpy()
    p = float(np.mean(y))
    entropy = float(-(p * np.log(max(p, 1e-12)) + (1 - p) * np.log(max(1 - p, 1e-12))))
    rows.append(
        {
            "regime": regime,
            "mode": mode,
            "n": int(len(group)),
            "prevalence": p,
            "prior_log_loss": entropy,
            "prior_brier": float(p * (1 - p)),
            "note": "descriptive held-out prevalence reference; not a trained model",
        }
    )
pd.DataFrame(rows).to_csv(OUT / "protocol_prior_reference.csv", index=False)
print("wrote per-animal and prior reference")
