"""Stage A: reproduce the frozen adaptive result and reconcile metrics."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from neurobranch.adaptive.evaluation import block_bootstrap_ci, evaluation_landmarks
from neurobranch.adaptive.var import FrozenVAR, RollingVAR
from neurobranch.config import PROJECT_ROOT, git_state
from neurobranch.diagnostics.common import load_nostim
from neurobranch.preprocessing import split_starts

OUT = PROJECT_ROOT / "reports/simulator/probabilistic_adaptive"
OUT.mkdir(parents=True, exist_ok=True)
HISTORICAL = PROJECT_ROOT / "reports/simulator/adaptive_dynamics/model_per_landmark.csv"


def _mse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((pred - truth) ** 2))


def main() -> int:
    _cfg, data, _recs = load_nostim()
    hist = pd.read_csv(HISTORICAL)
    hist_h20 = hist[hist.horizon == 20]
    old = hist_h20.pivot_table(index=["worm_id", "landmark"], columns="model", values="mse")
    rows, overlap_rows = [], []
    for d in data:
        frozen = FrozenVAR.fit(
            d.X, split_starts(d, "train", history=1, horizon=1), ridge=1.0, spectral_cap=0.995
        )
        rolling = RollingVAR(window=500)
        for t in evaluation_landmarks(d, horizon_max=100, stride=10):
            t = int(t)
            truth = d.X[t + 20]
            frozen_pred = frozen.forecast(d.X[t], 20)[-1]
            roll_fit = rolling.fit_at(d.X, t)
            roll_pred = roll_fit.forecast(d.X[t], 20)[-1]
            rows.append(
                {
                    "worm_id": int(d.worm_id),
                    "landmark": t,
                    "frozen_mse": _mse(frozen_pred, truth),
                    "rolling_w500_mse": _mse(roll_pred, truth),
                    "frozen_spectral_radius": frozen.spectral_radius,
                    "rolling_spectral_radius": roll_fit.spectral_radius,
                }
            )
        overlap_rows.append(
            {
                "worm_id": int(d.worm_id),
                "landmark_stride_frames": 10,
                "horizon_frames": 20,
                "adjacent_forecast_overlap_frames": 10,
                "adjacent_forecast_jaccard": 10 / 30,
                "n_landmarks": int(len(evaluation_landmarks(d, horizon_max=100, stride=10))),
                "test_frames": int(d.masks["test"].sum()),
            }
        )
    reproduced = pd.DataFrame(rows).set_index(["worm_id", "landmark"])
    merged = old.join(reproduced, how="inner")
    diff = merged["frozen_var"] - merged["frozen_mse"]
    diff_r = merged["rolling_w500"] - merged["rolling_w500_mse"]
    frame = reproduced.reset_index()
    frame.to_csv(OUT / "reproduction_landmarks.csv", index=False)

    f = frame.frozen_mse.to_numpy()
    a = frame.rolling_w500_mse.to_numpy()
    diff_vals = f - a
    per_landmark_rel = diff_vals / f
    aggregate_rel = float(1.0 - a.mean() / f.mean())
    ratio_of_means = float(diff_vals.mean() / f.mean())
    rows_summary = [
        {
            "quantity": "n_landmarks",
            "value": int(len(frame)),
            "definition": "35 chronological test landmarks per session, stride 10 frames",
        },
        {"quantity": "frozen_mean_h20_mse", "value": float(f.mean()), "definition": "unweighted mean over 175 landmarks"},
        {"quantity": "rolling_w500_mean_h20_mse", "value": float(a.mean()), "definition": "unweighted mean over 175 landmarks"},
        {"quantity": "mean_paired_difference", "value": float(diff_vals.mean()), "definition": "mean_i(frozen_i - adaptive_i)"},
        {
            "quantity": "relative_improvement_aggregate",
            "value": aggregate_rel,
            "definition": "1 - mean_i(adaptive_i) / mean_i(frozen_i); ratio of aggregate MSEs",
        },
        {
            "quantity": "relative_improvement_per_landmark_mean",
            "value": float(per_landmark_rel.mean()),
            "definition": "mean_i[(frozen_i - adaptive_i) / frozen_i]; equal weight per landmark",
        },
        {
            "quantity": "mean_difference_over_frozen_mean",
            "value": ratio_of_means,
            "definition": "mean_i(frozen_i - adaptive_i) / mean_i(frozen_i); a third aggregation of the same numbers",
        },
        {
            "quantity": "max_abs_reproduction_difference_frozen",
            "value": float(np.max(np.abs(diff))),
            "definition": "max |historical frozen MSE - recomputed frozen MSE|",
        },
        {
            "quantity": "max_abs_reproduction_difference_rolling",
            "value": float(np.max(np.abs(diff_r))),
            "definition": "max |historical rolling w500 MSE - recomputed rolling MSE|",
        },
        {
            "quantity": "train_standardized_predictor_mean_abs",
            "value": float(np.mean([np.abs(d.X[d.masks["train"]].mean(axis=0)).mean() for d in data])),
            "definition": "mean absolute train-split predictor mean after existing standardization",
        },
        {
            "quantity": "train_standardized_predictor_std_mean",
            "value": float(np.mean([d.X[d.masks["train"]].std(axis=0).mean() for d in data])),
            "definition": "mean train-split predictor standard deviation after existing standardization",
        },
    ]
    summary = pd.DataFrame(rows_summary)
    summary.to_csv(OUT / "reproduction_summary.csv", index=False)

    block_rows = []
    for block in (1, 2, 5, 10, 20, 30, 50):
        mean, lo, hi = block_bootstrap_ci(diff_vals, block=block, n_boot=1000)
        rmean, rlo, rhi = block_bootstrap_ci(per_landmark_rel, block=block, n_boot=1000)
        block_rows.append(
            {
                "block_length": block,
                "n_landmarks": int(len(diff_vals)),
                "mean_difference": mean,
                "difference_ci_low": lo,
                "difference_ci_high": hi,
                "mean_per_landmark_relative": rmean,
                "relative_ci_low": rlo,
                "relative_ci_high": rhi,
            }
        )
    blocks = pd.DataFrame(block_rows)
    blocks.to_csv(OUT / "block_length_sensitivity.csv", index=False)

    sessions = (
        frame.groupby("worm_id")
        .agg(
            n_landmarks=("landmark", "size"),
            frozen_mean_mse=("frozen_mse", "mean"),
            rolling_mean_mse=("rolling_w500_mse", "mean"),
            mean_paired_difference=("frozen_mse", "mean"),
        )
        .reset_index()
    )
    sessions["mean_paired_difference"] = [
        float((frame[frame.worm_id == w].frozen_mse - frame[frame.worm_id == w].rolling_w500_mse).mean())
        for w in sessions.worm_id
    ]
    sessions["aggregate_relative_improvement"] = 1 - sessions.rolling_mean_mse / sessions.frozen_mean_mse
    sessions.to_csv(OUT / "reproduction_by_session.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(OUT / "forecast_overlap.csv", index=False)

    payload = {
        "stage": "A_reproduction_and_metric_reconciliation",
        "historical_artifact": str(HISTORICAL.relative_to(PROJECT_ROOT)),
        "git": git_state(),
        "evaluation_landmarks": "split_starts(test, history=2, horizon=100)[::10]",
        "frozen_baseline": "train-split-only weighted ridge VAR, ridge=1.0, spectral cap=0.995",
        "primary_model": "prespecified Rolling VAR w500; not selected on test",
        "primary_metric": "paired H20 free-running MSE improvement versus frozen VAR",
        "relative_definitions": {
            "aggregate": "1 - mean(adaptive MSE) / mean(frozen MSE) = 0.411078",
            "per_landmark_mean": "mean[(frozen MSE - adaptive MSE)/frozen MSE] = 0.338626",
            "difference_over_frozen_mean": "mean(frozen - adaptive)/mean(frozen) = 0.411078",
        },
        "uncertainty": "moving-block bootstrap over overlapping landmarks; blocks 1-50 reported",
        "evidence_tier": "exploratory repeated-session reanalysis; not animal-level inference",
        "overlap_note": "H20 forecasts at stride 10 overlap 10 frames; landmarks are dependent.",
    }
    (OUT / "reproduction_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(blocks.to_string(index=False))
    print(sessions.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
