"""Stage G: bounded named-neuron sensitivity probes with controls.

These are mathematical interventions on a model.  They are not optogenetic
doses, not experimental stimulation, and not causal circuit claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT, ExperimentConfig, git_state
from neurobranch.dynamics.context_models import StableLatentLDS
from neurobranch.dynamics.training import load_worm_data
from neurobranch.preprocessing import split_starts
from neurobranch.simulator.adapters import NeuralSimulator
from neurobranch.simulator.metrics import first_outcome
from neurobranch.simulator.types import PerturbationSpec

TARGETS = ("AVBL", "AVBR", "AVEL", "AVER", "RIVL", "RIVR")
WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION - NOT CAUSALLY VALIDATED. Not an experimental dose."


def _starts(data, n: int = 5) -> np.ndarray:
    sim_starts = split_starts(data, "test", history=1, horizon=20)
    if len(sim_starts) > n:
        idx = np.linspace(0, len(sim_starts) - 1, n).round().astype(int)
        sim_starts = sim_starts[np.unique(idx)]
    return sim_starts


def _amplitude(data, target: str) -> tuple[float, float]:
    j = data.predictor_names.index(target)
    starts = split_starts(data, "train", history=1, horizon=1)
    delta = data.X[starts + 1, j] - data.X[starts, j]
    sd = float(np.std(delta))
    lo, hi = np.quantile(data.X[:, j], [0.05, 0.95])
    return sd, float(0.5 * (hi - lo))


def _rollout(sim, frame, spec, seed):
    return sim.rollout(
        None,
        20,
        initial_history=sim.initial_history(int(frame), history=1),
        seed=int(seed),
        perturbation=spec,
        stochastic=False,
    )


def _effect(base, perturbed, sim) -> dict[str, Any]:
    shift = perturbed.states - base.states
    final = float(np.linalg.norm(shift[-1]))
    maximum = float(np.max(np.linalg.norm(shift, axis=1)))
    d = perturbed.support_distance
    th = sim.support.thresholds
    return {
        "final_shift": final,
        "max_shift": maximum,
        "mse_shift": float(np.mean(shift**2)),
        "support_q90_fraction": float(np.mean(d <= th["q90"])),
        "support_q95_fraction": float(np.mean(d <= th["q95"])),
        "support_q99_fraction": float(np.mean(d <= th["q99"])),
        "max_support_distance": float(np.max(d)),
        "outcome": int(first_outcome(int(base.behavior_proba.argmax(axis=1)[0]), perturbed.behavior_proba.argmax(axis=1))),
        "finite": bool(np.isfinite(perturbed.states).all()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/forward_stress.yaml")
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports/simulator/context_dynamics")
    args = parser.parse_args(argv)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    selected = json.loads((reports / "model_selection.json").read_text(encoding="utf-8"))["selected_latent"]
    datasets = load_worm_data(ExperimentConfig.load(args.config))
    rows: list[dict[str, Any]] = []
    for data in datasets:
        model = StableLatentLDS.fit(
            data.X,
            data.y_macro,
            split_starts(data, "train", history=1, horizon=1),
            train_mask=data.masks["train"],
            latent_dim=int(selected["latent_dim"]),
            ridge=float(selected["ridge"]),
            spectral_cap=float(selected["spectral_cap"]),
        )
        sim = NeuralSimulator(model, data)
        available = [t for t in TARGETS if t in data.predictor_names]
        controls = [
            n
            for n in data.predictor_names
            if n.isalpha() and n.isupper() and n not in available
        ]
        if not available or not controls:
            continue
        control = controls[len(controls) // 2]
        frames = _starts(data)
        for frame in frames:
            base = _rollout(sim, int(frame), None, 17000 + int(frame))
            for target in available:
                sd, cap = _amplitude(data, target)
                for protocol, duration in (("initial_state", 0), ("dynamical_input", 5)):
                    for direction in (1, -1):
                        amp = float(direction * min(sd, cap))
                        spec = PerturbationSpec(
                            target=target,
                            protocol=protocol,  # type: ignore[arg-type]
                            mode="additive",
                            amplitude=amp,
                            onset=0,
                            duration=duration,
                        )
                        pert = _rollout(sim, int(frame), spec, 17000 + int(frame))
                        eff = _effect(base, pert, sim)
                        ctrl_spec = PerturbationSpec(
                            target=control,
                            protocol=protocol,  # type: ignore[arg-type]
                            mode="additive",
                            amplitude=amp,
                            onset=0,
                            duration=duration,
                        )
                        ctrl = _rollout(sim, int(frame), ctrl_spec, 17000 + int(frame))
                        ceff = _effect(base, ctrl, sim)
                        rows.append(
                            {
                                "worm_id": int(data.worm_id),
                                "frame": int(frame),
                                "target": target,
                                "control_target": control,
                                "protocol": protocol,
                                "duration": duration,
                                "amplitude": amp,
                                "direction": direction,
                                **{f"target_{k}": v for k, v in eff.items()},
                                **{f"control_{k}": v for k, v in ceff.items()},
                                "shift_excess_over_control": eff["final_shift"] - ceff["final_shift"],
                                "warning": WARNING,
                            }
                        )
        print(f"perturbations worm {data.worm_id} done", flush=True)
    frame_df = pd.DataFrame(rows)
    frame_df.to_csv(reports / "perturbation_events.csv", index=False)
    summary = (
        frame_df.groupby(["target", "protocol", "direction"], as_index=False)
        .agg(
            n=("target_final_shift", "size"),
            mean_target_shift=("target_final_shift", "mean"),
            mean_control_shift=("control_final_shift", "mean"),
            mean_excess=("shift_excess_over_control", "mean"),
            mean_support_q95=("target_support_q95_fraction", "mean"),
            mean_support_q99=("target_support_q99_fraction", "mean"),
            worms=("worm_id", "nunique"),
            worms_positive_excess=("shift_excess_over_control", lambda s: int((s > 0).sum())),
        )
        .sort_values("mean_excess", ascending=False)
    )
    summary.to_csv(reports / "perturbation_summary.csv", index=False)
    payload = {
        "schema_version": 1,
        "stage": "G_perturbations",
        "evidence_tier": "exploratory",
        "targets_requested": list(TARGETS),
        "targets_present_by_worm": {
            str(data.worm_id): [t for t in TARGETS if t in data.predictor_names] for data in datasets
        },
        "control_target_rule": "deterministic alphabetic non-target neuron; same signed amplitude/protocol",
        "amplitude_rule": "min(one-step SD of target, half central 90% range); standardised model units",
        "warning": WARNING,
        "gate7": "not_evaluable: no matched experimental intervention",
        "gate8": "not_evaluable: no compatible intervention dataset",
        "git": git_state(),
    }
    (reports / "perturbation_summary.json").write_text(
        json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(summary.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
