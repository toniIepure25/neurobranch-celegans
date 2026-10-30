"""Phase 12: animal-cluster benchmark inference from saved predictions."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from neurobranch.benchmark.inference import inference_summary, paired_cluster_inference

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "reports" / "phase8" / "predictions.parquet"
OUT = ROOT / "reports" / "phase12"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not PREDICTIONS.is_file():
        payload = {
            "status": "insufficient_animal_level_input",
            "detail": "reports/phase8/predictions.parquet is not present; no interval was fabricated.",
        }
        (OUT / "benchmark_cluster.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return
    predictions = pd.read_parquet(PREDICTIONS)
    frame = paired_cluster_inference(predictions, n_boot=1000, seed=29)
    frame.to_csv(OUT / "benchmark_cluster.csv", index=False)
    summary = inference_summary(frame)
    summary["bootstrap"] = 1000
    summary["baseline"] = "cro_static"
    (OUT / "benchmark_cluster.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not frame.empty:
        print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
