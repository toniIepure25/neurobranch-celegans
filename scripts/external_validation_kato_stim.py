"""Pre-registered external pilot on the constant-oxygen Kato Stim segment.

This is a per-animal refit test (protocol B), not a frozen-model transfer.  The
model family and hyperparameters are fixed from Phase 6; no external test result
is used to choose a model.  Oxygen is held at 21% and is never used as an input.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT
from neurobranch.data.external import build_kato_stim_worm_data, load_kato_stim
from neurobranch.dynamics.models import VARModel
from neurobranch.preprocessing import split_starts
from neurobranch.simulator import NeuralSimulator
from neurobranch.simulator.metrics import one_step_metrics, rollout_metrics

FROZEN_CONFIG = {"lag": 2, "ridge": 10.0, "spectral_cap": 0.995, "support_quantile": 0.99}
WARNING = (
    "EXPLORATORY EXTERNAL AUTONOMOUS-DYNAMICS PILOT — NOT A SENSORY-RESPONSE, "
    "DIGITAL-TWIN OR CAUSAL-INTERVENTION RESULT."
)


def _cluster_bootstrap(values: np.ndarray, *, n_boot: int = 2000, seed: int = 17) -> list[float] | None:
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if values.size < 2:
        return None
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def _subset_starts(starts: np.ndarray, max_starts: int) -> np.ndarray:
    starts = np.asarray(starts, dtype=np.int64)
    if len(starts) <= max_starts:
        return starts
    idx = np.linspace(0, len(starts) - 1, max_starts).round().astype(int)
    return starts[np.unique(idx)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stim", type=Path, default=PROJECT_ROOT / "data/raw/external/kato2015/WT_Stim.mat")
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports/external_validation")
    parser.add_argument("--max-seconds", type=float, default=360.0)
    parser.add_argument("--starts", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)

    recordings = load_kato_stim(args.stim, verify=True)
    per_worm: list[dict[str, Any]] = []
    per_start: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for rec in recordings:
        try:
            data = build_kato_stim_worm_data(rec, max_seconds=args.max_seconds)
            train_starts = split_starts(data, "train", history=FROZEN_CONFIG["lag"], horizon=1)
            model = VARModel.fit(
                data.X,
                data.y_macro,
                train_starts,
                history=FROZEN_CONFIG["lag"],
                ridge=FROZEN_CONFIG["ridge"],
                seed=args.seed,
                spectral_cap=FROZEN_CONFIG["spectral_cap"],
            )
            sim = NeuralSimulator(model, data, support_quantile=FROZEN_CONFIG["support_quantile"])
            starts = _subset_starts(
                split_starts(data, "test", history=FROZEN_CONFIG["lag"], horizon=args.horizon),
                args.starts,
            )
            one_step = one_step_metrics(
                sim,
                split_starts(data, "test", history=FROZEN_CONFIG["lag"], horizon=1),
                history=FROZEN_CONFIG["lag"],
            )
            summary, rows = rollout_metrics(
                sim, starts, horizon=args.horizon, noise_std=0.0, base_seed=args.seed
            )
            for row in rows:
                row["worm_id"] = int(rec.worm_index)
                row["dataset_name"] = rec.dataset_name
                row["skill_vs_persistence"] = (
                    1.0 - row["mse"] / row["persistence_mse"]
                    if row.get("finite") and row.get("persistence_mse", 0) > 0
                    else float("nan")
                )
                per_start.append(row)
            test_labels, counts = np.unique(data.y_macro[data.masks["test"]], return_counts=True)
            behavior = {
                str(int(label)): int(count) / int(data.masks["test"].sum())
                for label, count in zip(test_labels, counts)
            }
            per_worm.append(
                {
                    "worm_id": int(rec.worm_index),
                    "dataset_name": rec.dataset_name,
                    "n_traces_raw": int(rec.n_traces),
                    "n_traces_used": int(data.n_neurons),
                    "n_behavior_traces_excluded": int(rec.behavior_mask.sum()),
                    "n_frames_used": int(data.n_frames),
                    "n_train_frames": int(data.masks["train"].sum()),
                    "n_val_frames": int(data.masks["val"].sum()),
                    "n_test_frames": int(data.masks["test"].sum()),
                    "n_test_starts": int(len(starts)),
                    "fps": float(rec.fps),
                    "spectral_radius": float(model.spectral_radius),
                    "one_step": one_step,
                    **summary,
                    "test_behavior_fraction": behavior,
                }
            )
            print(
                f"worm {rec.worm_index}: neurons={data.n_neurons} starts={len(starts)} "
                f"skill={summary['skill_vs_persistence']:.4f}",
                flush=True,
            )
        except Exception as exc:  # keep every failure visible
            failures.append({"worm_id": int(rec.worm_index), "error": repr(exc)})
            print(f"worm {rec.worm_index}: FAILED {exc!r}", flush=True)

    skills = np.array([float(row["skill_vs_persistence"]) for row in per_worm], dtype=float)
    summary_payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "warning": WARNING,
        "evidence_tier": "exploratory_external_pilot",
        "protocol": "B_per_animal_refit",
        "frozen_config": FROZEN_CONFIG,
        "model_selection": "none; Phase 6 VAR hyperparameters frozen before this run",
        "input_used": False,
        "oxygen": {
            "identity": "O2",
            "segment": f"0 <= t < {args.max_seconds} s",
            "concentration_percent": 21.0,
            "switch_in_segment": False,
            "interpretation": "constant environmental condition, not targeted neuronal stimulation",
        },
        "split": "chronological 2/3 train, 1/6 validation, purge 12, remainder test",
        "baseline": "constant initial state held across the rollout horizon",
        "primary_horizon": int(args.horizon),
        "n_sessions_attempted": len(recordings),
        "n_sessions_ok": len(per_worm),
        "failures": failures,
        "n_positive_skill": int(np.sum(skills > 0)) if skills.size else 0,
        "median_skill_vs_persistence": float(np.median(skills)) if skills.size else None,
        "mean_skill_vs_persistence": float(np.mean(skills)) if skills.size else None,
        "bootstrap_95ci_mean_skill": _cluster_bootstrap(skills),
        "mean_energy_distance": float(np.nanmean([r["energy_distance"] for r in per_worm]))
        if per_worm
        else None,
        "mean_autocorr_error": float(np.nanmean([r["autocorr_error"] for r in per_worm]))
        if per_worm
        else None,
        "mean_covariance_error": float(np.nanmean([r["covariance_error"] for r in per_worm]))
        if per_worm
        else None,
        "mean_ood_fraction": float(np.nanmean([r["ood_fraction"] for r in per_worm])) if per_worm else None,
        "frozen_phase5_var_applicable": False,
        "frozen_phase5_var_reason": (
            "Phase 5/6 VAR checkpoints are tied to each NoStim worm's ordered 103-127-dimensional "
            "trace panel; external worms have different panels and trace-level multi-labels. "
            "No valid frozen-transfer comparison exists."
        ),
        "comparison_to_phase6": (
            "Descriptive only. Phase 6 used five different NoStim sessions, a blocked split and a "
            "reused test block; these are unpaired observations from different animals."
        ),
        "limitations": [
            "Seven recording sessions; animal independence is not proven by OSF metadata.",
            "Constant-oxygen first half only; no oxygen-response or sensory-input test was run.",
            "Behaviour labels are inferred from calcium signals, not independent behaviour.",
            "Different coarse behaviour vocabulary and trace panels from NoStim.",
            "Small n; bootstrap interval is descriptive and does not rescue a failed effect.",
        ],
    }

    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / "kato_stim_pilot_summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(per_worm).to_csv(args.reports / "kato_stim_pilot_per_worm.csv", index=False)
    pd.DataFrame(per_start).to_csv(args.reports / "kato_stim_pilot_starts.csv", index=False)
    print(
        json.dumps(
            {
                k: summary_payload[k]
                for k in (
                    "n_sessions_ok",
                    "n_positive_skill",
                    "median_skill_vs_persistence",
                    "mean_skill_vs_persistence",
                    "bootstrap_95ci_mean_skill",
                )
            },
            indent=2,
        )
    )
    return 0 if per_worm else 1


if __name__ == "__main__":
    raise SystemExit(main())
