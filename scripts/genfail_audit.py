"""GENFAIL Stage A: reproduce prior metrics and audit data/split provenance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT, git_state, sha256_file
from neurobranch.diagnostics.common import load_nostim
from neurobranch.dynamics.context_models import StableLatentLDS
from neurobranch.dynamics.stable_var import StableVARModel
from neurobranch.preprocessing import split_starts

OUT = PROJECT_ROOT / "reports/simulator/failure_diagnostics"
CONTEXT = PROJECT_ROOT / "reports/simulator/context_dynamics"
PHASE6 = PROJECT_ROOT / "reports/simulator/phase6"


def _skill(mse: float, persistence: float) -> float:
    return float(1.0 - mse / persistence) if persistence > 0 else float("nan")


def _prior_metric_reproduction() -> tuple[pd.DataFrame, dict]:
    rows = []
    phase6 = pd.read_csv(PHASE6 / "test_per_start.csv")
    phase6_summary = pd.read_csv(PHASE6 / "test_summary.csv")
    keys = ["family", "worm_id", "horizon", "seed"]
    recomputed = (
        phase6.groupby(keys, as_index=False)
        .agg(mse=("mse", "mean"), persistence_mse=("persistence_mse", "mean"))
        .assign(recomputed_skill=lambda d: 1.0 - d.mse / d.persistence_mse)
    )
    merged = phase6_summary.merge(recomputed, on=keys, suffixes=("_reported", "_recomputed"))
    for _, row in merged.iterrows():
        rows.append(
            {
                "source": "phase6_test_summary",
                "key": "|".join(str(row[k]) for k in keys),
                "metric": "skill_vs_persistence",
                "reported": float(row["skill_vs_persistence"]),
                "recomputed": float(row["recomputed_skill"]),
                "abs_difference": abs(float(row["skill_vs_persistence"]) - float(row["recomputed_skill"])),
            }
        )

    context = pd.read_csv(CONTEXT / "free_running_latent.csv")
    context_per_start_note = {
        "source": "context_free_running_latent_per_start",
        "status": "UNRESOLVED_SCHEMA",
        "reason": "the stored per-start table has model and worm_id but no horizon column; horizon-specific recomputation is impossible without guessing row groups",
    }

    documented = [
        ("context_latent_H20_mean", -1.616),
        ("context_latent_H20_median", -0.9107),
        ("context_latent_H100_mean", -0.7147),
        ("context_latent_H100_median", -0.1639),
        ("stable_var_H20_mean", -1.4813),
        ("stable_var_H20_median", -0.7699),
        ("stable_var_H100_mean", -0.6124),
        ("stable_var_H100_median", -0.0606),
    ]
    for label, value in documented:
        model = "latent_lds" if "latent" in label else "stable_var"
        horizon = 20 if "H20" in label else 100
        subset = context[(context.model == model) & (context.horizon == horizon)]
        recomputed = (
            float(subset.skill_vs_persistence.mean())
            if "mean" in label
            else float(subset.skill_vs_persistence.median())
        )
        rows.append(
            {
                "source": "documented_context_headline",
                "key": label,
                "metric": "skill_vs_persistence",
                "reported": float(value),
                "recomputed": recomputed,
                "abs_difference": abs(float(value) - recomputed),
            }
        )

    table = pd.DataFrame(rows)
    table["tolerance"] = np.where(table.source == "documented_context_headline", 1e-3, 1e-6)
    table["within_tolerance"] = table.abs_difference <= table.tolerance
    discrepancies = table[~table.within_tolerance].to_dict(orient="records")
    return table, {
        "n_checks": len(table),
        "n_discrepancies": len(discrepancies),
        "discrepancies": discrepancies,
        "unresolved": [context_per_start_note],
    }


def _refit_one() -> list[dict]:
    _, datasets, _ = load_nostim()
    data = datasets[0]
    train = split_starts(data, "train", history=1, horizon=1)
    test = split_starts(data, "test", history=1, horizon=20)
    lat = StableLatentLDS.fit(
        data.X, data.y_macro, train, train_mask=data.masks["train"], latent_dim=15, ridge=1e-2
    )
    var = StableVARModel.fit(data.X, data.y_macro, train, history=1, ridge=1.0)
    truth = np.stack([data.X[t + 1 : t + 21] for t in test])
    persistence = np.stack([np.repeat(data.X[t][None], 20, axis=0) for t in test])
    lat_pred = np.stack(
        [lat.rollout(np.zeros(data.n_neurons), 20, initial_history=data.X[t - 0 : t + 1])[1:] for t in test]
    )
    var_pred = np.empty_like(truth)
    for i, t in enumerate(test):
        current = data.X[t]
        for step in range(20):
            current = np.asarray(
                var.predict_next(current[None, None, :]), dtype=np.float64
            ).reshape(data.n_neurons)
            var_pred[i, step] = current
    base = float(np.mean((persistence - truth) ** 2))
    rows = []
    for model, pred in [("latent_lds", lat_pred), ("stable_var", var_pred)]:
        rows.append(
            {
                "source": "independent_refit_session0",
                "model": model,
                "worm_id": 0,
                "horizon": 20,
                "n_starts": len(test),
                "mse": float(np.mean((pred - truth) ** 2)),
                "persistence_mse": base,
                "skill_vs_persistence": _skill(float(np.mean((pred - truth) ** 2)), base),
                "realized_spectral_radius": float(getattr(lat if model == "latent_lds" else var, "realized_spectral_radius", np.nan)),
            }
        )
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg, datasets, recordings = load_nostim()
    data_rows = []
    split_rows = []
    for data in datasets:
        rec = data.recording
        dt = np.diff(rec.time_seconds)
        data_rows.append(
            {
                "worm_id": int(data.worm_id),
                "n_frames": int(data.n_frames),
                "n_neurons_total": int(rec.n_neurons),
                "n_predictors": int(data.n_neurons),
                "n_excluded_behavior_neurons": int(len(data.excluded_neurons)),
                "fps": float(rec.fps),
                "median_dt_seconds": float(np.median(dt)),
                "min_dt_seconds": float(dt.min()),
                "max_dt_seconds": float(dt.max()),
                "time_monotonic": bool(np.all(dt > 0)),
                "nan_fraction_X": float(np.isnan(data.X).mean()),
                "nan_fraction_X_raw": float(np.isnan(data.X_raw).mean()),
                "finite_X": bool(np.isfinite(data.X).all()),
                "duration_seconds": float(rec.time_seconds[-1] - rec.time_seconds[0]),
                "source_sha256": rec.source_sha256,
            }
        )
        for split, (lo, hi) in data.split_bounds.items():
            mask = data.masks[split]
            split_rows.append(
                {
                    "worm_id": int(data.worm_id),
                    "split": split,
                    "frame_start": int(lo),
                    "frame_stop": int(hi),
                    "n_frames": int(hi - lo),
                    "mask_count": int(mask.sum()),
                    "first_mask_frame": int(np.flatnonzero(mask)[0]),
                    "last_mask_frame": int(np.flatnonzero(mask)[-1]),
                    "n_test_start_h20": int(len(split_starts(data, split, history=1, horizon=20))),
                }
            )
    pd.DataFrame(data_rows).to_csv(OUT / "data_audit.csv", index=False)
    pd.DataFrame(split_rows).to_csv(OUT / "split_audit.csv", index=False)
    reproduction, repro_summary = _prior_metric_reproduction()
    reproduction.to_csv(OUT / "prior_metric_reproduction.csv", index=False)
    refit = pd.DataFrame(_refit_one())
    refit.to_csv(OUT / "independent_refit.csv", index=False)

    checkpoint_missing = []
    manifest = PHASE6 / "test_summary.csv"
    if manifest.exists():
        summary = pd.read_csv(manifest)
        for path in summary["checkpoint"].dropna().unique():
            if not Path(path).exists():
                checkpoint_missing.append(str(path))
    payload = {
        "stage": "A_reproduction_audit",
        "evidence_tier": "exploratory; prior test tails already inspected",
        "source_file": str(cfg.data.path),
        "source_sha256": sha256_file(PROJECT_ROOT / cfg.data.path),
        "n_recordings": len(recordings),
        "n_sessions": len(datasets),
        "prior_metric_reproduction": repro_summary,
        "independent_refit": refit.to_dict(orient="records"),
        "missing_phase6_checkpoints": sorted(set(checkpoint_missing)),
        "missing_checkpoint_count": len(set(checkpoint_missing)),
        "discrepancy_policy": "prior reports are not modified; differences are recorded in this file",
        "git": git_state(),
    }
    (OUT / "reproduction_audit.json").write_text(
        json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["prior_metric_reproduction"], indent=2))
    print(refit.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
