"""Validation-only model selection for Phase 6 autonomous dynamics.

Fits the predeclared compact families on training blocks, evaluates only on the
validation block, and freezes one configuration per family.  The test block is
not read here.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT, ExperimentConfig, sha256_file
from neurobranch.dynamics.checkpointing import save_model
from neurobranch.dynamics.models import VARModel
from neurobranch.dynamics.switching import SwitchingVARModel
from neurobranch.preprocessing import split_starts
from neurobranch.simulator import NeuralSimulator
from neurobranch.simulator.metrics import rollout_metrics
from neurobranch.simulator.train import checkpoint_root, load_datasets

HORIZON = 20


@dataclass(frozen=True)
class VarConfig:
    lag: int
    ridge: float
    spectral_cap: float | None = 0.995

    @property
    def label(self) -> str:
        cap = "none" if self.spectral_cap is None else f"{self.spectral_cap:g}"
        return f"var_p{self.lag}_r{self.ridge:g}_c{cap}"


@dataclass(frozen=True)
class NoiseConfig:
    noise_model: str
    bootstrap_block: int = 5

    @property
    def label(self) -> str:
        return f"stochastic_{self.noise_model}"


VAR_GRID = tuple(
    VarConfig(lag=lag, ridge=ridge)
    for lag in (1, 2, 3)
    for ridge in (1e-3, 1e-2, 1e-1, 1.0, 10.0)
)
NOISE_GRID = (
    NoiseConfig("gaussian"),
    NoiseConfig("bootstrap_iid"),
    NoiseConfig("bootstrap_block", 5),
)
SWITCHING_KS = (2, 3)


def _fixed_starts(data, split: str, *, n: int = 50, history: int = 3) -> np.ndarray:
    starts = split_starts(data, split, history=history, horizon=HORIZON)
    if len(starts) > n:
        idx = np.linspace(0, len(starts) - 1, n).round().astype(int)
        starts = starts[np.unique(idx)]
    return np.asarray(starts, dtype=np.int64)


def _metrics(model, data, starts: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    sim = NeuralSimulator(model, data)
    summary, per_start = rollout_metrics(
        sim,
        starts,
        horizon=HORIZON,
        noise_std=0.0,
        base_seed=1700,
    )
    frame = pd.DataFrame(per_start)
    return summary, frame


def _row(
    *,
    family: str,
    config: str,
    data,
    summary: dict[str, Any],
    n_starts: int,
) -> dict[str, Any]:
    return {
        "family": family,
        "config": config,
        "worm_id": int(data.worm_id),
        "n_starts": n_starts,
        "mse": summary["mse"],
        "persistence_mse": summary["persistence_mse"],
        "skill": summary["skill_vs_persistence"],
        "r2": summary["r2"],
        "energy_distance": summary["energy_distance"],
        "autocorr_error": summary["autocorr_error"],
        "covariance_error": summary["covariance_error"],
        "ood_fraction": summary["ood_fraction"],
        "readout_supported_fraction": summary["readout_supported_fraction"],
        "finite_fraction": summary["finite_fraction"],
    }


def _save(model, path: Path, metadata: dict[str, Any]) -> str:
    save_model(model, path, metadata=metadata)
    return sha256_file(path)


def select_var(datasets, root: Path, reports: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    starts_by_worm = {int(d.worm_id): _fixed_starts(d, "val") for d in datasets}
    for cfg in VAR_GRID:
        for data in datasets:
            starts = starts_by_worm[int(data.worm_id)]
            train_starts = split_starts(data, "train", history=cfg.lag, horizon=1)
            model = VARModel.fit(
                data.X,
                data.y_macro,
                train_starts,
                history=cfg.lag,
                ridge=cfg.ridge,
                seed=17,
                spectral_cap=cfg.spectral_cap,
                noise_model="none",
            )
            summary, _ = _metrics(model, data, starts)
            rows.append(_row(family="var", config=cfg.label, data=data, summary=summary, n_starts=len(starts)))
            print(f"var {cfg.label} worm {data.worm_id}: skill={summary['skill_vs_persistence']:.4f}", flush=True)
    frame = pd.DataFrame(rows)
    med = frame.groupby("config", as_index=False).agg(
        median_skill=("skill", "median"),
        mean_skill=("skill", "mean"),
        positive_worms=("skill", lambda s: int((s > 0).sum())),
        mean_energy=("energy_distance", "mean"),
        mean_ood=("ood_fraction", "mean"),
    )
    med["complexity"] = med["config"].map(
        {c.label: (c.lag, c.ridge) for c in VAR_GRID}
    )
    med = med.sort_values(
        ["median_skill", "mean_energy", "complexity"],
        ascending=[False, True, True],
    )
    best_label = str(med.iloc[0]["config"])
    best = next(c for c in VAR_GRID if c.label == best_label)
    frozen = {
        "family": "var",
        "config": asdict(best),
        "label": best.label,
        "validation_median_skill": float(med.iloc[0]["median_skill"]),
        "validation_positive_worms": int(med.iloc[0]["positive_worms"]),
        "selection_table": med.to_dict(orient="records"),
    }
    model_dir = root / "phase6" / "var"
    for data in datasets:
        starts = split_starts(data, "train", history=best.lag, horizon=1)
        model = VARModel.fit(
            data.X,
            data.y_macro,
            starts,
            history=best.lag,
            ridge=best.ridge,
            seed=17,
            spectral_cap=best.spectral_cap,
            noise_model="none",
        )
        meta = {
            "phase": 6,
            "family": "var",
            "config": asdict(best),
            "worm_id": int(data.worm_id),
            "selection": "validation only; test block not read",
            "history": best.lag,
            "training_blocks": [0, 1, 3],
            "validation_block": 4,
            "test_block": 2,
        }
        frozen.setdefault("checkpoints", {})[str(data.worm_id)] = _save(
            model, model_dir / f"worm{data.worm_id}.pt", meta
        )
    (reports / "var_selection.json").write_text(json.dumps(frozen, indent=2, default=float) + "\n", encoding="utf-8")
    frame.to_csv(reports / "var_validation_grid.csv", index=False)
    return frozen, rows


def select_noise(datasets, selected: dict[str, Any], root: Path, reports: Path) -> dict[str, Any]:
    rows = []
    starts_by_worm = {int(d.worm_id): _fixed_starts(d, "val") for d in datasets}
    cfg = selected["config"]
    for noise in NOISE_GRID:
        for data in datasets:
            starts = starts_by_worm[int(data.worm_id)]
            train_starts = split_starts(data, "train", history=int(cfg["lag"]), horizon=1)
            model = VARModel.fit(
                data.X,
                data.y_macro,
                train_starts,
                history=int(cfg["lag"]),
                ridge=float(cfg["ridge"]),
                seed=17,
                spectral_cap=cfg["spectral_cap"],
                noise_model=noise.noise_model,
                bootstrap_block=noise.bootstrap_block,
            )
            summary, _ = _metrics(model, data, starts)
            rows.append(
                _row(
                    family="stochastic_var",
                    config=noise.label,
                    data=data,
                    summary=summary,
                    n_starts=len(starts),
                )
            )
            print(
                f"noise {noise.label} worm {data.worm_id}: skill={summary['skill_vs_persistence']:.4f} "
                f"energy={summary['energy_distance']:.4f}",
                flush=True,
            )
    frame = pd.DataFrame(rows)
    med = frame.groupby("config", as_index=False).agg(
        median_skill=("skill", "median"),
        mean_energy=("energy_distance", "mean"),
        mean_ood=("ood_fraction", "mean"),
        positive_worms=("skill", lambda s: int((s > 0).sum())),
    )
    # Distributional fidelity is primary for stochastic models, with skill as
    # a guardrail: do not select a noise model that destroys trajectory skill.
    med = med.sort_values(["mean_energy", "median_skill"], ascending=[True, False])
    best_label = str(med.iloc[0]["config"])
    frozen = {
        "family": "stochastic_var",
        "base_var": selected["config"],
        "config": next(asdict(n) for n in NOISE_GRID if n.label == best_label),
        "label": best_label,
        "validation_positive_worms": int(med.iloc[0]["positive_worms"]),
        "validation_table": med.to_dict(orient="records"),
    }
    model_dir = root / "phase6" / "stochastic_var"
    for data in datasets:
        starts = split_starts(data, "train", history=int(cfg["lag"]), horizon=1)
        noise = next(n for n in NOISE_GRID if n.label == best_label)
        model = VARModel.fit(
            data.X,
            data.y_macro,
            starts,
            history=int(cfg["lag"]),
            ridge=float(cfg["ridge"]),
            seed=17,
            spectral_cap=cfg["spectral_cap"],
            noise_model=noise.noise_model,
            bootstrap_block=noise.bootstrap_block,
        )
        meta = {
            "phase": 6,
            "family": "stochastic_var",
            "config": asdict(noise),
            "base_var": selected["config"],
            "worm_id": int(data.worm_id),
            "selection": "validation only; test block not read",
            "history": int(cfg["lag"]),
        }
        frozen.setdefault("checkpoints", {})[str(data.worm_id)] = _save(
            model, model_dir / f"worm{data.worm_id}.pt", meta
        )
    (reports / "stochastic_selection.json").write_text(
        json.dumps(frozen, indent=2, default=float) + "\n", encoding="utf-8"
    )
    frame.to_csv(reports / "stochastic_validation_grid.csv", index=False)
    return frozen


def select_switching(datasets, root: Path, reports: Path) -> dict[str, Any]:
    rows = []
    starts_by_worm = {int(d.worm_id): _fixed_starts(d, "val") for d in datasets}
    fits: dict[tuple[int, int], SwitchingVARModel] = {}
    for K in SWITCHING_KS:
        for data in datasets:
            starts = starts_by_worm[int(data.worm_id)]
            train_starts = split_starts(data, "train", history=1, horizon=1)
            model = SwitchingVARModel.fit(
                data.X,
                data.y_macro,
                train_starts,
                K=K,
                ridge=1e-2,
                seed=17,
                restarts=5,
                n_iter=40,
            )
            fits[(K, int(data.worm_id))] = model
            summary, _ = _metrics(model, data, starts)
            row = _row(
                family="switching_var",
                config=f"switching_K{K}",
                data=data,
                summary=summary,
                n_starts=len(starts),
            )
            row["min_occupancy"] = float(model.stationary.min())
            rows.append(row)
            print(
                f"switching K={K} worm {data.worm_id}: skill={summary['skill_vs_persistence']:.4f} "
                f"min_occ={model.stationary.min():.3f}",
                flush=True,
            )
    frame = pd.DataFrame(rows)
    med = frame.groupby("config", as_index=False).agg(
        median_skill=("skill", "median"),
        mean_skill=("skill", "mean"),
        positive_worms=("skill", lambda s: int((s > 0).sum())),
        mean_energy=("energy_distance", "mean"),
        min_occupancy=("min_occupancy", "min"),
    )
    med["K"] = med["config"].str.extract(r"K(\d+)").astype(int)
    med = med.sort_values(["median_skill", "mean_energy", "K"], ascending=[False, True, True])
    k2 = med[med["K"] == 2].iloc[0]
    eligible = [r for r in med.to_dict(orient="records") if r["min_occupancy"] >= 0.05]
    best = eligible[0] if eligible else med.iloc[0].to_dict()
    # K=2 is primary; K=3 must improve median skill by at least 2%.
    if int(best["K"]) == 3 and float(best["median_skill"]) < float(k2["median_skill"]) * 1.02:
        best = k2.to_dict()
    frozen = {
        "family": "switching_var",
        "K": int(best["K"]),
        "label": str(best["config"]),
        "validation_median_skill": float(best["median_skill"]),
        "validation_positive_worms": int(best["positive_worms"]),
        "validation_table": med.to_dict(orient="records"),
    }
    model_dir = root / "phase6" / "switching_var"
    for data in datasets:
        model = fits[(int(best["K"]), int(data.worm_id))]
        meta = {
            "phase": 6,
            "family": "switching_var",
            "K": int(best["K"]),
            "worm_id": int(data.worm_id),
            "selection": "validation only; test block not read",
            "diagnostics": model.diagnostics,
        }
        frozen.setdefault("checkpoints", {})[str(data.worm_id)] = _save(
            model, model_dir / f"worm{data.worm_id}.pt", meta
        )
    (reports / "switching_selection.json").write_text(
        json.dumps(frozen, indent=2, default=float) + "\n", encoding="utf-8"
    )
    frame.to_csv(reports / "switching_validation_grid.csv", index=False)
    return frozen


def trigger_b4(var: dict[str, Any], stochastic: dict[str, Any], switching: dict[str, Any]) -> dict[str, Any]:
    positive = max(
        int(var.get("validation_positive_worms", 0)),
        int(stochastic.get("validation_positive_worms", 0)),
        int(switching.get("validation_positive_worms", 0)),
    )
    return {
        "trigger": positive >= 3,
        "positive_worms": positive,
        "rule": "at least one B1-B3 configuration has positive validation skill in >=3/5 worms",
        "action": "fit compact nonlinear B4" if positive >= 3 else "B4 not pursued",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/blocked_suite.yaml")
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports/simulator/phase6")
    parser.add_argument("--families", nargs="+", default=["var", "stochastic", "switching"])
    args = parser.parse_args(argv)
    cfg = ExperimentConfig.load(args.config)
    root = checkpoint_root(cfg)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    datasets = load_datasets(cfg)
    manifest = {
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "data_sha256": sha256_file(cfg.data.path),
        "horizon": HORIZON,
        "validation_starts": {
            str(d.worm_id): _fixed_starts(d, "val").tolist() for d in datasets
        },
    }
    (reports / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    var = None
    if "var" in args.families:
        var, _ = select_var(datasets, root, reports)
    if var is None:
        var = json.loads((reports / "var_selection.json").read_text(encoding="utf-8"))
    stochastic = None
    if "stochastic" in args.families:
        stochastic = select_noise(datasets, var, root, reports)
    if stochastic is None:
        stochastic = json.loads((reports / "stochastic_selection.json").read_text(encoding="utf-8"))
    switching = None
    if "switching" in args.families:
        switching = select_switching(datasets, root, reports)
    if switching is None:
        switching = json.loads((reports / "switching_selection.json").read_text(encoding="utf-8"))
    payload = {
        "families": {"var": var, "stochastic_var": stochastic, "switching_var": switching},
        "b4": trigger_b4(var, stochastic, switching),
        "test_block_read": False,
        "frozen_at": "validation selection; test evaluation must be exploratory",
    }
    (reports / "selection_frozen.json").write_text(
        json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(json.dumps({"b4": payload["b4"], "var": var["label"], "stochastic": stochastic["label"], "switching": switching["label"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
