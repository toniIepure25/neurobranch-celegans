"""Summarize per-animal differences for the offline and online benchmarks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "reports" / "phase5" / "protocol_comparison"
ONLINE = ROOT / "reports" / "phase5" / "online_decoding"


def offline_summary() -> pd.DataFrame:
    frame = pd.read_csv(PROTOCOL / "protocol_per_animal.csv")
    frame[["regime", "mode"]] = frame["mode"].str.split(":", n=1, expand=True)
    rows = []
    for regime in ("original", "grouped", "chronological", "external"):
        part = frame[frame["regime"] == regime]
        wide = part.pivot_table(index="animal_id", columns="mode", values="log_loss")
        if "behavior" not in wide:
            continue
        for mode in ("saav", "both"):
            if mode not in wide:
                continue
            delta = (wide[mode] - wide["behavior"]).dropna()
            if delta.empty:
                continue
            rows.append(
                {
                    "regime": regime,
                    "mode": mode,
                    "n_animals": int(len(delta)),
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(delta.median()),
                    "min_delta": float(delta.min()),
                    "max_delta": float(delta.max()),
                    "n_improved": int((delta < 0).sum()),
                    "fraction_improved": float((delta < 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def online_summary() -> pd.DataFrame:
    predictions = pd.read_parquet(ONLINE / "online_predictions.parquet")
    rows = []
    for landmark in sorted(predictions["landmark_s"].unique()):
        part = predictions[
            (predictions["landmark_s"] == landmark) & (predictions["model"] == "linear")
        ]
        events = (
            part.groupby(["mode", "event_uid"])
            .agg(p=("p", "mean"), y=("y", "first"), animal=("animal_id", "first"))
            .reset_index()
        )
        for mode in ("behavior_saav", "behavior_population"):
            values = []
            for _, group in events.groupby("animal"):
                scores = {}
                for candidate in ("behavior", mode):
                    selected = group[group["mode"] == candidate]
                    if len(selected) and selected["y"].nunique() == 2:
                        scores[candidate] = log_loss(
                            selected["y"], selected["p"], labels=[0, 1]
                        )
                if "behavior" in scores and mode in scores:
                    values.append(scores[mode] - scores["behavior"])
            if not values:
                continue
            delta = np.asarray(values)
            rows.append(
                {
                    "landmark_s": float(landmark),
                    "mode": mode,
                    "n_animals": int(len(delta)),
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(np.median(delta)),
                    "min_delta": float(delta.min()),
                    "max_delta": float(delta.max()),
                    "n_improved": int((delta < 0).sum()),
                    "fraction_improved": float((delta < 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    offline = offline_summary()
    online = online_summary()
    offline.to_csv(PROTOCOL / "protocol_per_animal_deltas.csv", index=False)
    online.to_csv(ONLINE / "online_per_animal_deltas.csv", index=False)
    summary = {
        "offline": offline.round(6).to_dict(orient="records"),
        "online": online.round(6).to_dict(orient="records"),
    }
    (ROOT / "reports" / "phase5" / "per_animal_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(offline)} offline and {len(online)} online per-animal rows")


if __name__ == "__main__":
    main()
