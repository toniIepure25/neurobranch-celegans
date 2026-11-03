"""Audit dependence and uncertainty using the committed paired landmarks only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from neurobranch.kato_interactions.metrics import session_block_ci  # noqa: E402

BASE = ROOT / "reports" / "simulator" / "kato_common_state"
OUT = ROOT / "reports" / "simulator" / "kato_final_audit"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(BASE / "model_per_landmark.csv")
    pivot = frame.pivot_table(index=["session", "landmark"], columns="model", values="pca10_energy_h20")
    diff = (pivot["B5_common_k10"] - pivot["B3_population"]).dropna().rename("b3_advantage").reset_index()
    diff = diff.sort_values(["session", "landmark"]).reset_index(drop=True)

    boot_rows = []
    for block in (3, 5, 10, 20):
        ci = session_block_ci(diff, "b3_advantage", session="session", block=block, seed=1700 + block)
        boot_rows.append(
            {
                "block": block,
                "n_landmarks": ci["n"],
                "mean_advantage": ci["mean"],
                "ci_low": ci["lo"],
                "ci_high": ci["hi"],
                "positive": bool(ci["lo"] > 0),
            }
        )
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(OUT / "uncertainty_bootstrap.csv", index=False)

    session_rows = []
    for session, group in diff.groupby("session"):
        values = group["b3_advantage"].to_numpy()
        ci = session_block_ci(group, "b3_advantage", session="session", block=5, seed=1705)
        session_rows.append(
            {
                "session": int(session),
                "n_landmarks": int(len(values)),
                "mean_advantage": float(values.mean()),
                "median_advantage": float(np.median(values)),
                "min_advantage": float(values.min()),
                "max_advantage": float(values.max()),
                "fraction_better": float((values > 0).mean()),
                "within_session_block5_low": ci["lo"],
                "within_session_block5_high": ci["hi"],
                "positive": bool(values.mean() > 0),
            }
        )
    sessions = pd.DataFrame(session_rows)
    sessions.to_csv(OUT / "session_estimates.csv", index=False)

    loo = []
    for session in sorted(diff["session"].unique()):
        subset = diff[diff["session"] != session]
        loo.append(
            {
                "excluded_session": int(session),
                "mean_advantage": float(subset["b3_advantage"].mean()),
                "n": int(len(subset)),
                "positive": bool(subset["b3_advantage"].mean() > 0),
            }
        )
    pd.DataFrame(loo).to_csv(OUT / "leave_one_session_out.csv", index=False)

    # Serial dependence of the paired difference within each session.
    serial_rows = []
    for session, group in diff.groupby("session"):
        values = group["b3_advantage"].to_numpy()
        centered = values - values.mean()
        denom = float(np.dot(centered, centered))
        for lag in (1, 2, 3, 5, 10):
            if len(values) <= lag:
                corr = float("nan")
            else:
                corr = float(np.dot(centered[:-lag], centered[lag:]) / denom) if denom > 0 else float("nan")
            serial_rows.append({"session": int(session), "lag": lag, "autocorrelation": corr})
    pd.DataFrame(serial_rows).to_csv(OUT / "paired_serial_dependence.csv", index=False)

    # Subsampling sensitivity: adjacent landmarks overlap heavily, so a subset
    # every k-th landmark is a dependence-reducing sensitivity check only.
    subset_rows = []
    for step in (1, 2, 3, 5):
        subset = diff[diff.groupby("session").cumcount() % step == 0]
        subset_rows.append(
            {
                "every_kth_landmark": step,
                "n": int(len(subset)),
                "mean_advantage": float(subset["b3_advantage"].mean()),
                "fraction_better": float((subset["b3_advantage"] > 0).mean()),
            }
        )
    pd.DataFrame(subset_rows).to_csv(OUT / "landmark_subsample_sensitivity.csv", index=False)

    horizon_rows = []
    for horizon in (1, 5, 10, 20):
        metric = f"pca10_energy_h{horizon}"
        if metric not in frame.columns:
            continue
        hp = frame.pivot_table(index=["session", "landmark"], columns="model", values=metric)
        hd = (hp["B5_common_k10"] - hp["B3_population"]).dropna().rename("b3_advantage").reset_index()
        ci = session_block_ci(hd, "b3_advantage", session="session", block=5, seed=1705)
        horizon_rows.append(
            {
                "horizon": horizon,
                "n": ci["n"],
                "mean_advantage": ci["mean"],
                "ci_low": ci["lo"],
                "ci_high": ci["hi"],
                "fraction_better": float((hd["b3_advantage"] > 0).mean()),
                "positive_interval": bool(ci["lo"] > 0),
            }
        )
    pd.DataFrame(horizon_rows).to_csv(OUT / "horizon_uncertainty.csv", index=False)

    # Descriptive influence of high-energy landmarks; not a new test.
    frame_p = pivot.reset_index()
    merged = diff.merge(
        frame_p[["session", "landmark", "B3_population", "B5_common_k10"]],
        on=["session", "landmark"],
        how="left",
    )
    merged["b3_energy_quartile"] = pd.qcut(merged["B3_population"], 4, labels=False, duplicates="drop")
    strata = (
        merged.groupby("b3_energy_quartile", observed=True)["b3_advantage"]
        .agg(["mean", "count", lambda x: float((x > 0).mean())])
        .reset_index()
        .rename(columns={"<lambda_0>": "fraction_better"})
    )
    strata.to_csv(OUT / "energy_strata_exploratory.csv", index=False)

    summary = {
        "schema": "neurobranch.kato_final_audit.uncertainty.v1",
        "source": "committed reports/simulator/kato_common_state/model_per_landmark.csv",
        "contrast": "B5_common_k10 - B3_population on pca10_energy_h20; positive = B3 better",
        "blocks": boot.to_dict(orient="records"),
        "session_positive_count": int(sessions["positive"].sum()),
        "sessions": sessions.to_dict(orient="records"),
        "leave_one_session_out_positive": bool(pd.DataFrame(loo)["positive"].all()),
        "landmark_subsample": subset_rows,
        "horizon": horizon_rows,
        "interpretation": (
            "Intervals resample overlapping forecast landmarks within recordings; they do not "
            "represent independent animals or fully independent forecast windows."
        ),
    }
    (OUT / "uncertainty_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
