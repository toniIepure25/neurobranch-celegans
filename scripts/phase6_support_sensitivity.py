"""Support-threshold sensitivity for the Phase 6 selected families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT, ExperimentConfig
from neurobranch.preprocessing import split_starts
from neurobranch.simulator import NeuralSimulator
from neurobranch.simulator.adapters import SupportModel
from neurobranch.simulator.train import checkpoint_root, load_datasets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/blocked_suite.yaml")
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports/simulator/phase6")
    args = parser.parse_args(argv)
    cfg = ExperimentConfig.load(args.config)
    root = checkpoint_root(cfg)
    reports = Path(args.reports)
    rows = []
    for data in load_datasets(cfg):
        starts = split_starts(data, "test", history=3, horizon=20)[:10]
        for family in ("var", "stochastic_var", "switching_var"):
            sim = NeuralSimulator.from_checkpoint(root / "phase6" / family / f"worm{data.worm_id}.pt", data)
            hist_len = int(getattr(sim.model, "history_len", 1))
            states = np.concatenate(
                [
                    sim.rollout(
                        None, 20, initial_history=sim.initial_history(int(t), history=hist_len),
                        seed=17 + int(t), stochastic=family != "var",
                    ).states[1:]
                    for t in starts
                ]
            )
            for q in (0.90, 0.95, 0.99):
                support = SupportModel(
                    data.X[data.masks["train"]],
                    quantile=q,
                    calibration_X=data.X[data.masks["val"]],
                )
                rows.append(
                    {
                        "family": family,
                        "worm_id": int(data.worm_id),
                        "quantile": q,
                        "threshold": support.threshold,
                        "rollout_in_support": float(np.mean(support.score(states) <= support.threshold)),
                        "observed_test_in_support": float(
                            np.mean(support.score(data.X[data.masks["test"]]) <= support.threshold)
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(reports / "support_sensitivity.csv", index=False)
    summary = (
        frame.groupby(["family", "quantile"], as_index=False)
        .agg(
            mean_rollout_in_support=("rollout_in_support", "mean"),
            mean_observed_test_in_support=("observed_test_in_support", "mean"),
            mean_threshold=("threshold", "mean"),
        )
        .to_dict(orient="records")
    )
    (reports / "support_sensitivity.json").write_text(
        json.dumps({"summary": summary, "note": "threshold calibrated on validation block 4"}, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
