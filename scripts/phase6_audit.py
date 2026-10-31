"""Phase 6 audit: reproduce Phase 5 free-running results and quantify fixes.

This script deliberately compares the corrected evaluation contract with the
legacy behaviours that Phase 5 used:

* true observed history vs a single repeated initial state;
* per-signal temporal autocorrelation vs concatenated-block correlation;
* free-running skill vs the one-step R2 difference used by the old Gate 2;
* validation-calibrated support vs a training-only 99th-percentile envelope.

It writes machine-readable evidence and never fits a new model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT, ExperimentConfig, sha256_file
from neurobranch.simulator.metrics import (
    _lag_correlation,
    one_step_metrics,
    rollout_metrics,
)
from neurobranch.simulator.train import (
    checkpoint_root,
    load_datasets,
    load_simulator_for,
)

AUDIT_KINDS = ("mlp_residual", "var", "gru")
AUDIT_SEEDS = (17, 23, 31)


def legacy_autocorr_error(sim: np.ndarray, real: np.ndarray) -> float:
    """The Phase 5 formula: correlate concatenated simulated and real blocks."""
    if len(sim) <= 11:
        return float("nan")
    errs = []
    for lag in (1, 2, 5, 10):
        a = np.corrcoef(sim[:-lag].T, sim[lag:].T)
        b = np.corrcoef(real[:-lag].T, real[lag:].T)
        n = sim.shape[1]
        errs.append(float(np.mean(np.abs(a[:n, n:] - b[:n, n:]))))
    return float(np.nanmean(errs))


def corrected_autocorr_error(sim: np.ndarray, real: np.ndarray) -> float:
    errs = []
    for lag in (1, 2, 5, 10):
        a = _lag_correlation(sim, lag)
        b = _lag_correlation(real, lag)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.any():
            errs.append(float(np.mean(np.abs(a[ok] - b[ok]))))
    return float(np.nanmean(errs)) if errs else float("nan")


def _legacy_rollout(sim, t: int, horizon: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rollout = sim.rollout(sim.data.X[int(t)], horizon, seed=seed)
    return rollout.states[1:], sim.data.X[int(t) : int(t) + horizon + 1]


def _free_skill_ci(rows: list[dict[str, Any]], *, seed: int, n_boot: int = 1000) -> dict[str, float]:
    valid = [r for r in rows if r.get("finite")]
    if not valid:
        return {"skill": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    mse = np.asarray([r["mse"] for r in valid])
    base = np.asarray([r["persistence_mse"] for r in valid])
    point = 1.0 - mse.mean() / base.mean()
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(valid), len(valid))
        denom = base[idx].mean()
        if denom > 0:
            vals.append(float(1.0 - mse[idx].mean() / denom))
    return {
        "skill": float(point),
        "ci_low": float(np.quantile(vals, 0.025)),
        "ci_high": float(np.quantile(vals, 0.975)),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    cfg = ExperimentConfig.load(args.config)
    root = checkpoint_root(cfg)
    datasets = load_datasets(cfg)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    stored_path = PROJECT_ROOT / "reports" / "simulator" / "rollout_metrics.csv"
    stored = pd.read_csv(stored_path) if stored_path.exists() else pd.DataFrame()

    for kind in args.kinds:
        for data in datasets:
            for seed in args.seeds:
                sim = load_simulator_for(data, kind, seed, root=root)
                starts = sim.test_starts(horizon=args.horizon)[: args.starts]
                summary, per_start = rollout_metrics(
                    sim,
                    starts,
                    horizon=args.horizon,
                    noise_std=0.0,
                    base_seed=1000 + seed,
                )
                one = one_step_metrics(sim)
                corrected_ci = _free_skill_ci(per_start, seed=seed)

                legacy_sim, legacy_real, legacy_baseline = [], [], []
                for i, t in enumerate(starts):
                    pred, real = _legacy_rollout(sim, int(t), args.horizon, 1000 + seed + i)
                    legacy_sim.append(pred)
                    legacy_real.append(real[1:])
                    legacy_baseline.append(np.repeat(sim.data.X[int(t)][None], len(real) - 1, axis=0))
                legacy_arr = np.concatenate(legacy_sim, axis=0)
                legacy_real_arr = np.concatenate(legacy_real, axis=0)
                legacy_autocorr = legacy_autocorr_error(legacy_arr, legacy_real_arr)
                corrected_autocorr = corrected_autocorr_error(legacy_arr, legacy_real_arr)

                history = int(getattr(sim.model, "history_len", 1))
                stored_mse = float("nan")
                if not stored.empty:
                    hit = stored[
                        (stored["kind"] == kind)
                        & (stored["worm_id"] == int(data.worm_id))
                        & (stored["seed"] == seed)
                    ]
                    if len(hit):
                        stored_mse = float(hit.iloc[0]["mse"])
                legacy_skill = 1.0 - float(
                    np.mean([(p - r) ** 2 for p, r in zip(legacy_sim, legacy_real)])
                ) / float(np.mean([(b - r) ** 2 for b, r in zip(legacy_baseline, legacy_real)]))
                free_skill = summary["skill_vs_persistence"]

                rows.append(
                    {
                        "kind": kind,
                        "worm_id": int(data.worm_id),
                        "seed": seed,
                        "history": history,
                        "n_starts": len(starts),
                        "one_step_r2": one["r2"],
                        "one_step_delta_r2_vs_persistence": one["delta_r2_vs_persistence"],
                        "free_running_skill": free_skill,
                        "free_running_mse": summary["mse"],
                        "persistence_mse": summary["persistence_mse"],
                        "stored_phase5_mse": stored_mse,
                        "delta_mse_vs_stored_phase5": float(summary["mse"] - stored_mse)
                        if np.isfinite(stored_mse)
                        else float("nan"),
                        "free_running_skill_corrected_ci_low": corrected_ci["ci_low"],
                        "legacy_x0_skill": float(legacy_skill),
                        "delta_skill_history_correction": float(free_skill - legacy_skill),
                        "legacy_autocorr_error": legacy_autocorr,
                        "corrected_autocorr_error": corrected_autocorr,
                        "delta_autocorr": float(corrected_autocorr - legacy_autocorr),
                        "ood_fraction_calibrated": summary["ood_fraction"],
                        "support_threshold_calibrated": float(sim.support.threshold),
                        "support_threshold_training": float(sim.support.training_threshold),
                        "readout_supported_fraction": summary["readout_supported_fraction"],
                        "finite_fraction": summary["finite_fraction"],
                        "beats_persistence": bool(corrected_ci["ci_low"] > 0),
                    }
                )
                if len(rows) % 9 == 0:
                    print(f"audited {len(rows)} model-worm-seed combinations", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(reports / "reproduction.csv", index=False)
    by_kind = (
        frame.groupby("kind", as_index=False)
        .agg(
            mean_one_step_delta_r2=("one_step_delta_r2_vs_persistence", "mean"),
            mean_free_skill=("free_running_skill", "mean"),
            mean_legacy_skill=("legacy_x0_skill", "mean"),
            mean_legacy_autocorr=("legacy_autocorr_error", "mean"),
            mean_corrected_autocorr=("corrected_autocorr_error", "mean"),
            mean_ood=("ood_fraction_calibrated", "mean"),
            n_rows=("kind", "size"),
        )
        .to_dict(orient="records")
    )
    payload = {
        "schema_version": 1,
        "config": str(args.config),
        "data_sha256": sha256_file(cfg.data.path),
        "checkpoint_root": str(root),
        "horizon": int(args.horizon),
        "starts": int(args.starts),
        "kinds": list(args.kinds),
        "seeds": list(args.seeds),
        "n_rows": len(rows),
        "by_kind": by_kind,
        "findings": {
            "history": "evaluation now uses the real observed history window; x0-only is retained as legacy",
            "autocorrelation": "corrected metric compares separate per-neuron lag correlations",
            "gate2": "free-running skill is now the gate quantity; one-step delta R2 is descriptive",
            "support": "threshold is the 99th percentile of held-out validation distances by default",
        },
        "start_manifest_sha256": None,
    }
    (reports / "summary.json").write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, default=float))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/blocked_suite.yaml")
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports/simulator/phase6/audit")
    parser.add_argument("--kinds", nargs="+", default=list(AUDIT_KINDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(AUDIT_SEEDS))
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--starts", type=int, default=20)
    args = parser.parse_args(argv)
    audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
