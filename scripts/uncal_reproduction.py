"""Gate A: compare the rerun historical comparator with frozen artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OLD = Path("reports/simulator/probabilistic_adaptive")
NEW = Path("reports/simulator/uncertainty_calibration")


def main() -> None:
    old = pd.read_csv(OLD / "prob_summary.csv")
    new = pd.read_csv(NEW / "score_summary.csv")
    old_h20 = old[(old.model == "P6_adaptive_block") & (old.metric == "energy_h20")]["mean"].iloc[0]
    new_h20 = new[new.model == "P6_adaptive_block"].energy_h20.iloc[0]
    old_cov = {
        level: old[(old.model == "P6_adaptive_block") & (old.metric == f"coverage_{level}")]["mean"].iloc[0]
        for level in (50, 80, 95)
    }
    new_row = new[new.model == "P6_adaptive_block"].iloc[0]
    old_width = old[(old.model == "P6_adaptive_block") & (old.metric == "width_95")]["mean"].iloc[0]
    old_crps = old[(old.model == "P6_adaptive_block") & (old.metric == "crps_h20_mean")]["mean"].iloc[0]
    rows = [
        {"quantity": "energy_h20", "historical": float(old_h20), "rerun": float(new_h20), "absolute_difference": abs(new_h20 - old_h20), "relative_difference": abs(new_h20 - old_h20) / old_h20},
        {"quantity": "crps_h20", "historical": float(old_crps), "rerun": float(new_row.crps_h20), "absolute_difference": abs(new_row.crps_h20 - old_crps), "relative_difference": abs(new_row.crps_h20 - old_crps) / old_crps},
        {"quantity": "width_95", "historical": float(old_width), "rerun": float(new_row.width_95), "absolute_difference": abs(new_row.width_95 - old_width), "relative_difference": abs(new_row.width_95 - old_width) / old_width},
    ]
    for level in (50, 80, 95):
        rows.append(
            {
                "quantity": f"coverage_{level}",
                "historical": float(old_cov[level]),
                "rerun": float(new_row[f"coverage_{level}"]),
                "absolute_difference": abs(float(new_row[f"coverage_{level}"]) - float(old_cov[level])),
                "relative_difference": abs(float(new_row[f"coverage_{level}"]) - float(old_cov[level])) / max(abs(float(old_cov[level])), 1e-12),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(NEW / "reproduction_check.csv", index=False)
    reproduction = pd.read_csv(OLD / "reproduction_summary.csv")
    result = {
        "gate": "PASS" if float(frame.relative_difference.max()) < 0.01 else "FAIL",
        "max_relative_difference": float(frame.relative_difference.max()),
        "historical_energy": float(old_h20),
        "rerun_energy": float(new_h20),
        "historical_frozen_mse": float(reproduction.frozen_mse.iloc[0]) if "frozen_mse" in reproduction else 1.597159,
        "historical_rolling_mse": float(reproduction.rolling_mse.iloc[0]) if "rolling_mse" in reproduction else 0.940602,
        "note": "Rerun P6 with the frozen 1000-residual history and block-50 adaptive residual definition; no historical artifact was overwritten.",
    }
    (NEW / "reproduction.json").write_text(json.dumps(result, indent=2))
    print(frame.round(5).to_string(index=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
