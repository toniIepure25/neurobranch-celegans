"""Stage D: oxygen-input ablation on Kato WT_Stim, with explicit low power.

All forecasts here are *known-schedule conditioned*: the measured oxygen series
for the rollout window is supplied as an external input.  This is not a
free-running forecast of a future stimulus and not targeted neuronal stimulation.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT, git_state
from neurobranch.data.external import build_kato_stim_context_worm_data, load_kato_stim
from neurobranch.dynamics.context_models import CompositionalLatentLDS, StableLatentLDS
from neurobranch.preprocessing import split_starts
from neurobranch.simulator.adapters import NeuralSimulator
from neurobranch.simulator.metrics import rollout_metrics


def _copy_with_schedule(data, values: np.ndarray, name: str):
    out = copy.copy(data)
    out.external_input = np.asarray(values, dtype=np.float64)
    out.input_name = name
    return out


def _time_since_transition(oxygen: np.ndarray, fps: float) -> np.ndarray:
    changes = np.flatnonzero(np.diff(oxygen) != 0) + 1
    elapsed = np.zeros(len(oxygen), dtype=np.float64)
    last = -np.inf
    change_set = set(int(i) for i in changes)
    for t in range(len(oxygen)):
        if t in change_set:
            last = float(t)
        elapsed[t] = 0.0 if not np.isfinite(last) else (t - last) / max(fps, 1e-9)
    return elapsed


def _lagged(oxygen: np.ndarray, lag: int) -> np.ndarray:
    out = np.empty_like(oxygen)
    out[:lag] = oxygen[0]
    out[lag:] = oxygen[:-lag]
    return out


def _one_step(data, model, schedule: np.ndarray | None) -> dict[str, float]:
    starts = split_starts(data, "test", history=1, horizon=1)
    H = np.stack([data.X[t : t + 1] for t in starts])
    preds = []
    for i, t in enumerate(starts):
        u = None if schedule is None else schedule[int(t) : int(t) + 1]
        preds.append(model.predict_next(H[i], inputs=u))
    pred = np.asarray(preds)
    truth = data.X[starts + 1]
    persistence = data.X[starts]
    mse = float(np.mean((pred - truth) ** 2))
    pers = float(np.mean((persistence - truth) ** 2))
    ss_res = float(np.sum((truth - pred) ** 2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "n_starts": int(len(starts)),
        "mse": mse,
        "persistence_mse": pers,
        "skill_vs_persistence": float(1 - mse / pers) if pers > 0 else float("nan"),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


@dataclass
class StimulusOnly:
    """E: current oxygen alone, with no neural history (deliberately weak)."""

    mean: np.ndarray
    coefficient: np.ndarray
    prior: np.ndarray
    n_neurons: int
    input_mean: float
    input_std: float
    input_dim: int = 1
    supports_input: bool = True
    history_len: int = 1
    native_stochastic: bool = False
    name: str = "stimulus_only"

    @classmethod
    def fit(cls, X, y, starts, train_mask, input_values):
        U = np.asarray(input_values, dtype=np.float64).reshape(-1)
        mu_u, sd_u = float(U[train_mask].mean()), float(U[train_mask].std() or 1.0)
        z = (U - mu_u) / sd_u
        mean = X[train_mask].mean(axis=0)
        target = X[starts + 1] - mean
        design = np.column_stack([z[starts], np.ones(len(starts))])
        coef = np.linalg.lstsq(design, target, rcond=None)[0]
        coefficient = coef[0]
        prior = np.bincount(y[train_mask], minlength=4).astype(float)
        prior /= max(1.0, prior.sum())
        return cls(mean, coefficient, prior, X.shape[1], mu_u, sd_u)

    def _u(self, inputs):
        if inputs is None:
            return 0.0
        return (float(np.asarray(inputs).reshape(-1)[0]) - self.input_mean) / self.input_std

    def predict_next(self, history, *, inputs=None):
        arr = np.asarray(history)
        if arr.ndim == 3:
            return np.stack([self.predict_next(h, inputs=inputs) for h in arr])
        return self.mean + self.coefficient * self._u(inputs)

    def behavior_proba(self, history):
        arr = np.asarray(history)
        n = arr.shape[0] if arr.ndim >= 2 else 1
        return np.tile(self.prior, (n, 1))

    def rollout(self, x0, steps, *, seed=0, noise_std=0.0, perturb=None,
                initial_history=None, stochastic=False, inputs=None):
        U = np.asarray(inputs, dtype=np.float64).reshape(steps) if inputs is not None else np.zeros(steps)
        states = np.empty((steps + 1, self.n_neurons))
        states[0] = np.asarray(x0).reshape(self.n_neurons)
        for s in range(steps):
            x = self.mean + self.coefficient * self._u(U[s])
            if perturb is not None:
                x = perturb(x, s)
            states[s + 1] = x
        return states


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stim", type=Path, default=Path(os.environ.get("TEMP", ".")) / "neurobranch_external" / "WT_Stim.mat")
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports/simulator/context_dynamics")
    args = parser.parse_args(argv)
    if not args.stim.exists():
        raise SystemExit(f"Kato WT_Stim not found: {args.stim}")
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    selection = json.loads((reports / "model_selection.json").read_text(encoding="utf-8"))["selected_latent"]
    recordings = load_kato_stim(args.stim, verify=True)

    summary_rows: list[dict[str, Any]] = []
    one_rows: list[dict[str, Any]] = []
    per_start: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for rec in recordings:
        data = build_kato_stim_context_worm_data(rec)
        oxygen = np.asarray(data.external_input, dtype=np.float64)
        fps = float(rec.fps)
        schedules = {
            "A_no_input": None,
            "B_observed_o2": oxygen,
            "D_lagged_o2_12f": _lagged(oxygen, 12),
            "D_time_since_transition": _time_since_transition(oxygen, fps),
            "D_time_index": np.linspace(0, 1, len(oxygen)),
        }
        train_starts = split_starts(data, "train", history=1, horizon=1)
        models: dict[str, Any] = {}
        for name, schedule in schedules.items():
            if name == "A_no_input":
                model = StableLatentLDS.fit(
                    data.X, data.y_macro, train_starts, train_mask=data.masks["train"],
                    latent_dim=int(selection["latent_dim"]), ridge=float(selection["ridge"]),
                    spectral_cap=float(selection["spectral_cap"]),
                )
            else:
                model = StableLatentLDS.fit(
                    data.X, data.y_macro, train_starts, train_mask=data.masks["train"],
                    latent_dim=int(selection["latent_dim"]), ridge=float(selection["ridge"]),
                    spectral_cap=float(selection["spectral_cap"]), input_values=schedule,
                )
            models[name] = model
        models["C_compositional_o2"] = CompositionalLatentLDS.fit(
            data.X, data.y_macro, train_starts, train_mask=data.masks["train"],
            latent_dim=int(selection["latent_dim"]), ridge=float(selection["ridge"]),
            spectral_cap=float(selection["spectral_cap"]), input_values=oxygen,
            n_operators=3, alternations=5,
        )
        models["E_stimulus_only_o2"] = StimulusOnly.fit(
            data.X, data.y_macro, train_starts, data.masks["train"], oxygen
        )
        schedule_by_model = {
            **schedules,
            "C_compositional_o2": oxygen,
            "E_stimulus_only_o2": oxygen,
        }
        for name, model in models.items():
            schedule = schedule_by_model[name]
            view = _copy_with_schedule(data, schedule, name) if schedule is not None else data
            horizons = (20,)
            for horizon in horizons:
                sim = NeuralSimulator(model, view)
                starts = sim.test_starts(horizon=horizon)
                if len(starts) > 40:
                    # The oxygen analysis is low power; a fixed, evenly spaced
                    # 40-start subset keeps the kNN support audit tractable.
                    idx = np.linspace(0, len(starts) - 1, 40).round().astype(int)
                    starts = starts[np.unique(idx)]
                summ, rows = rollout_metrics(
                    sim, starts=starts, horizon=horizon,
                    use_external_input=schedule is not None,
                )
                summary_rows.append(
                    {
                        "model": name,
                        "session_id": rec.session_id,
                        "worm_id": int(data.worm_id),
                        "horizon": int(horizon),
                        "n_starts": int(summ["n_starts"]),
                        "mse": summ["mse"],
                        "persistence_mse": summ["persistence_mse"],
                        "skill_vs_persistence": summ["skill_vs_persistence"],
                        "r2": summ["r2"],
                        "energy_distance": summ["energy_distance"],
                        "support_q90_fraction": summ["support_q90_fraction"],
                        "support_q95_fraction": summ["support_q95_fraction"],
                        "support_q99_fraction": summ["support_q99_fraction"],
                        "forecast_type": "known_schedule_conditioned" if schedule is not None else "autonomous",
                    }
                )
                per_start[(rec.session_id, name, horizon)] = rows
            one = _one_step(data, model, schedule)
            one_rows.append({"model": name, "session_id": rec.session_id, **one})
        print(f"oxygen models {rec.session_id} done", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(reports / "oxygen_model_comparison.csv", index=False)
    pd.DataFrame(one_rows).to_csv(reports / "oxygen_one_step.csv", index=False)

    # Paired per-start bootstrap of B - A and C - B, within session.
    paired = []
    for sid in summary["session_id"].unique():
        for horizon in (20, 100):
            for candidate in ("B_observed_o2", "C_compositional_o2"):
                base = "A_no_input" if candidate == "B_observed_o2" else "B_observed_o2"
                a = per_start.get((sid, base, horizon), [])
                b = per_start.get((sid, candidate, horizon), [])
                if not a or len(a) != len(b):
                    continue
                diff = np.array([r["mse"] - q["mse"] for r, q in zip(b, a)])
                rng = np.random.default_rng(17)
                boots = [float(np.mean(rng.choice(diff, len(diff), replace=True))) for _ in range(1000)]
                paired.append(
                    {
                        "session_id": sid,
                        "horizon": horizon,
                        "candidate": candidate,
                        "baseline": base,
                        "n_starts": int(len(diff)),
                        "mean_mse_difference": float(diff.mean()),
                        "ci_low": float(np.percentile(boots, 2.5)),
                        "ci_high": float(np.percentile(boots, 97.5)),
                        "candidate_better_fraction": float(np.mean(diff < 0)),
                    }
                )
    pd.DataFrame(paired).to_csv(reports / "oxygen_pairwise.csv", index=False)
    agg = (
        summary.groupby(["model", "horizon"], as_index=False)
        .agg(
            sessions=("session_id", "nunique"),
            mean_skill=("skill_vs_persistence", "mean"),
            median_skill=("skill_vs_persistence", "median"),
            mean_mse=("mse", "mean"),
            mean_persistence=("persistence_mse", "mean"),
            mean_support_q95=("support_q95_fraction", "mean"),
        )
    )
    agg.to_csv(reports / "oxygen_model_aggregate.csv", index=False)
    payload = {
        "schema_version": 1,
        "stage": "D_oxygen_input_models",
        "evidence_tier": "exploratory_low_power",
        "n_sessions": int(summary["session_id"].nunique()),
        "held_out_transitions_per_session": 3,
        "total_held_out_transition_events": int(3 * summary["session_id"].nunique()),
        "forecast_semantics": "known measured oxygen schedule supplied; not a stimulus forecast and not targeted neuronal stimulation",
        "horizons": "H20 only, on an evenly spaced subset of up to 40 held-out starts per session; H100 is covered in Stage B",
        "models": {
            "A_no_input": "autonomous latent LDS",
            "B_observed_o2": "latent LDS with measured oxygen input",
            "C_compositional_o2": "compositional operators with causal coefficient policy and oxygen input",
            "D_lagged_o2_12f": "oxygen delayed by 12 frames (~4 s) as a timing control",
            "D_time_since_transition": "elapsed time since last oxygen switch as a non-specific control",
            "D_time_index": "linear time as a non-specific control",
            "E_stimulus_only_o2": "current oxygen alone with no neural history",
        },
        "warning": "Oxygen is an environmental sensory input. This is not targeted neuronal stimulation or optogenetics.",
        "power_warning": "Seven sessions share a fixed schedule; transition events are not independent biological replicates. Any positive difference is low-power and exploratory.",
        "git": git_state(),
    }
    (reports / "oxygen_model_summary.json").write_text(
        json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
