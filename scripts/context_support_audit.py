"""Stage A support audit: q90/q95/q99 thresholds and raw distances.

Phase 6 is read-only here.  Its official support choice remains q99; this script
reports the lower-quantile audit without relabelling any frozen result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT, ExperimentConfig, git_state
from neurobranch.dynamics.context_models import StableLatentLDS
from neurobranch.dynamics.stable_var import StableVARModel
from neurobranch.dynamics.training import load_worm_data
from neurobranch.preprocessing import split_starts
from neurobranch.simulator.adapters import NeuralSimulator


def _phase6(phase6: Path, out: Path) -> dict:
    thresholds = pd.read_csv(phase6 / "support_sensitivity.csv")
    thresholds = thresholds.rename(columns={"quantile": "q", "threshold": "calibration_threshold"})
    starts = pd.read_csv(phase6 / "test_per_start.csv")
    merged = starts.merge(
        thresholds[["family", "worm_id", "q", "calibration_threshold"]],
        on=["family", "worm_id"],
        how="left",
        validate="many_to_many",
    )
    merged["within_support_at_q"] = merged["max_support_distance"] <= merged["calibration_threshold"]
    merged.to_csv(out / "phase6_support_event_audit.csv", index=False)
    summary = (
        merged.groupby(["family", "worm_id", "q"], as_index=False)
        .agg(
            n_start_rows=("frame", "size"),
            n_unique_starts=("frame", "nunique"),
            mean_max_support_distance=("max_support_distance", "mean"),
            max_max_support_distance=("max_support_distance", "max"),
            fraction_starts_fully_inside_at_q=("within_support_at_q", "mean"),
        )
    )
    summary.to_csv(out / "phase6_support_summary.csv", index=False)
    official = (
        summary[summary["q"] == 0.99]
        .groupby("family", as_index=False)["fraction_starts_fully_inside_at_q"]
        .mean()
        .rename(columns={"fraction_starts_fully_inside_at_q": "mean_q99_fraction"})
    )
    return {
        "official_quantile": 0.99,
        "note": (
            "Phase 6 official support remains q99. The per-start table stores only each rollout's "
            "maximum distance, so the audit is conservative and cannot recover the full per-event "
            "distance distribution without re-simulation."
        ),
        "official_q99_fractions": official.to_dict(orient="records"),
    }


def _context(out: Path, selected: dict, stable_var: dict) -> dict:
    datasets = load_worm_data(ExperimentConfig.load(PROJECT_ROOT / "configs/experiments/forward_stress.yaml"))
    threshold_rows = []
    for data in datasets:
        latent = StableLatentLDS.fit(
            data.X,
            data.y_macro,
            split_starts(data, "train", history=1, horizon=1),
            train_mask=data.masks["train"],
            latent_dim=int(selected["latent_dim"]),
            ridge=float(selected["ridge"]),
            spectral_cap=float(selected["spectral_cap"]),
        )
        var = StableVARModel.fit(
            data.X,
            data.y_macro,
            split_starts(data, "train", history=1, horizon=1),
            history=int(stable_var["lag"]),
            ridge=float(stable_var["ridge"]),
            spectral_cap=float(stable_var.get("spectral_cap", 0.995)),
        )
        for model_name, model in (("latent_lds", latent), ("var_stable", var)):
            sim = NeuralSimulator(model, data)
            th = sim.support.thresholds
            scores = np.asarray(sim.support.calibration_scores, dtype=float)
            threshold_rows.append(
                {
                    "model": model_name,
                    "worm_id": int(data.worm_id),
                    "q90_threshold": float(th["q90"]),
                    "q95_threshold": float(th["q95"]),
                    "q99_threshold": float(th["q99"]),
                    "n_calibration_scores": int(len(scores)),
                    "calibration_score_mean": float(scores.mean()),
                    "calibration_score_max": float(scores.max()),
                }
            )
    pd.DataFrame(threshold_rows).to_csv(out / "context_support_thresholds.csv", index=False)
    per_start = pd.read_csv(out / "free_running_latent_per_start.csv")
    joined = per_start.merge(
        pd.DataFrame([
            r for r in threshold_rows if r["model"] == "latent_lds"
        ])[["worm_id", "q90_threshold", "q95_threshold", "q99_threshold"]],
        on="worm_id", how="left",
    )
    for q in (90, 95, 99):
        joined[f"max_inside_q{q}"] = joined["max_support_distance"] <= joined[f"q{q}_threshold"]
    joined.to_csv(out / "context_latent_support_event_audit.csv", index=False)
    summary = (
        joined.groupby("worm_id", as_index=False)
        .agg(
            n_starts=("frame", "size"),
            mean_max_support_distance=("max_support_distance", "mean"),
            fraction_max_inside_q90=("max_inside_q90", "mean"),
            fraction_max_inside_q95=("max_inside_q95", "mean"),
            fraction_max_inside_q99=("max_inside_q99", "mean"),
        )
    )
    summary.to_csv(out / "context_latent_support_summary.csv", index=False)
    return {"thresholds": threshold_rows, "summary": summary.to_dict(orient="records")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase6", type=Path, default=PROJECT_ROOT / "reports/simulator/phase6")
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports/simulator/context_dynamics")
    args = parser.parse_args(argv)
    out = Path(args.reports)
    out.mkdir(parents=True, exist_ok=True)
    selection = json.loads((out / "model_selection.json").read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "stage": "A_support_audit",
        "q90_q95_q99": True,
        "phase6": _phase6(Path(args.phase6), out),
        "context": _context(out, selection["selected_latent"], selection["selected_stable_var"]),
        "warning": (
            "A support threshold is a train/validation calibration diagnostic. Being inside it does "
            "not make a rollout scientifically valid or a forecast correct."
        ),
        "git": git_state(),
    }
    (out / "support_audit.json").write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "context"}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
