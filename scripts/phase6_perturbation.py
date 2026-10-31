"""Support-constrained Phase 6 virtual perturbations.

Amplitudes are derived from each target's observed one-step change and capped to
the central 90% marginal range.  Outputs are model-dependent sensitivity probes,
not experimental stimulation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT, ExperimentConfig, sha256_file
from neurobranch.preprocessing import split_starts
from neurobranch.simulator import NeuralSimulator
from neurobranch.simulator.perturbation import CANDIDATES, outcome_distribution, tv
from neurobranch.simulator.train import checkpoint_root, load_datasets
from neurobranch.simulator.types import PerturbationSpec

WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED."
Q_GRID = (1.0,)
DURATIONS = (1, 5)


def _amplitudes(data, target: str) -> tuple[float, float, float]:
    j = data.predictor_names.index(target)
    starts = split_starts(data, "train", history=1, horizon=1)
    delta = data.X[starts + 1, j] - data.X[starts, j]
    sd = float(np.std(delta))
    lo, hi = np.quantile(data.X[:, j], [0.05, 0.95])
    return sd, float(lo), float(hi)


def _target(name: str, amp: float, protocol: str, duration: int = 0) -> PerturbationSpec:
    return PerturbationSpec(
        target=name,
        protocol=protocol,
        mode="additive",
        amplitude=float(amp),
        onset=0,
        duration=int(duration),
    )


def _frames(data, n: int = 8) -> np.ndarray:
    starts = split_starts(data, "test", history=3, horizon=20)
    if len(starts) > n:
        idx = np.linspace(0, len(starts) - 1, n).round().astype(int)
        starts = starts[np.unique(idx)]
    return starts


def _ensemble(sim, frame: int, n: int, spec: PerturbationSpec | None = None):
    initial = int(sim.data.y_macro[int(frame)])
    h = int(getattr(sim.model, "history_len", 1))
    return sim.rollout_ensemble(
        None,
        n,
        20,
        initial_history=sim.initial_history(int(frame), history=h),
        initial_macro=initial,
        base_seed=17000 + int(frame),
        perturbation=spec,
        include_latent=False,
    )


def _effect(sim, frame: int, n: int, spec: PerturbationSpec | None) -> dict[str, Any]:
    ens = _ensemble(sim, frame, n, spec)
    initial = int(sim.data.y_macro[int(frame)])
    dist = outcome_distribution(ens.rollouts, initial)
    stack = ens.stack
    return {
        "distribution": dist,
        "states": stack,
        "ood_fraction": ens.ood_fraction,
        "readout_supported_fraction": float(
            np.mean([r.metadata["readout_supported_fraction"] for r in ens.rollouts])
        ),
        "max_abs": float(np.nanmax(np.abs(stack))),
    }


def run_family(data, sim, family: str, *, n_rollouts: int, frames: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    random_name = next(
        n for n in data.predictor_names if n.isalpha() and n not in CANDIDATES and n.isupper()
    )
    for frame in frames:
        base = _effect(sim, int(frame), n_rollouts, None)
        control_cache: dict[tuple[float, str, int], dict[str, Any]] = {}
        for target in CANDIDATES:
            sd, lo, hi = _amplitudes(data, target)
            cap = max(0.0, 0.5 * (hi - lo))
            for q in Q_GRID:
                amp = min(float(q * sd), cap)
                for direction in (1, -1):
                    signed_amp = direction * amp
                    for protocol, duration in (
                        ("initial_state", 0),
                        *[(d, d) for d in DURATIONS],
                    ):
                        spec = _target(
                            target,
                            signed_amp,
                            "initial_state" if protocol == "initial_state" else "dynamical_input",
                            duration,
                        )
                        eff = _effect(sim, int(frame), n_rollouts, spec)
                        tv_effect = tv(base["distribution"], eff["distribution"])
                        ctrl_spec = _target(random_name, signed_amp, spec.protocol, duration)
                        ctrl_key = (round(signed_amp, 10), spec.protocol, duration)
                        if ctrl_key not in control_cache:
                            control_cache[ctrl_key] = _effect(
                                sim, int(frame), n_rollouts, ctrl_spec
                            )
                        ctrl = control_cache[ctrl_key]
                        tv_ctrl = tv(base["distribution"], ctrl["distribution"])
                        rows.append(
                            {
                                "family": family,
                                "worm_id": int(data.worm_id),
                                "frame": int(frame),
                                "target": target,
                                "control_target": random_name,
                                "protocol": spec.protocol,
                                "duration": int(duration),
                                "direction": int(direction),
                                "q": float(q),
                                "amplitude": signed_amp,
                                "tv_effect": tv_effect,
                                "tv_control": tv_ctrl,
                                "specificity_ratio": float(tv_effect / tv_ctrl) if tv_ctrl > 1e-9 else float("nan"),
                                "in_support_fraction": float(1.0 - eff["ood_fraction"]),
                                "readout_supported_fraction": eff["readout_supported_fraction"],
                                "baseline_ood_fraction": float(base["ood_fraction"]),
                                "mean_final_shift": float(
                                    np.mean(
                                        np.linalg.norm(
                                            eff["states"][:, -1] - base["states"][:, -1], axis=1
                                        )
                                    )
                                ),
                                "max_abs": eff["max_abs"],
                                "warning": WARNING,
                            }
                        )
        print(f"{family} worm {data.worm_id} frame {frame}: {len(rows)} rows", flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/blocked_suite.yaml")
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports/simulator/phase6")
    parser.add_argument("--family", default="stochastic_var")
    parser.add_argument("--rollouts", type=int, default=15)
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args(argv)
    cfg = ExperimentConfig.load(args.config)
    root = checkpoint_root(cfg)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    datasets = load_datasets(cfg)
    rows: list[dict[str, Any]] = []
    for data in datasets:
        sim = NeuralSimulator.from_checkpoint(
            root / "phase6" / args.family / f"worm{data.worm_id}.pt", data
        )
        rows.extend(run_family(data, sim, args.family, n_rollouts=args.rollouts, frames=_frames(data, n=args.frames)))
    frame = pd.DataFrame(rows)
    frame.to_csv(reports / "perturbation_events.csv", index=False)
    grouped = frame.groupby(["family", "target", "protocol", "duration", "direction", "q"], as_index=False).agg(
        n=("tv_effect", "size"),
        mean_tv=("tv_effect", "mean"),
        mean_control_tv=("tv_control", "mean"),
        mean_specificity=("specificity_ratio", "mean"),
        mean_in_support=("in_support_fraction", "mean"),
        mean_readout_support=("readout_supported_fraction", "mean"),
        mean_final_shift=("mean_final_shift", "mean"),
        max_abs=("max_abs", "max"),
    )
    grouped.to_csv(reports / "perturbation_summary.csv", index=False)
    # Gate G5 uses the highest-support amplitude/protocol per target.
    support = (
        frame.groupby(["family", "target", "protocol", "duration", "direction", "q"], as_index=False)
        .agg(in_support=("in_support_fraction", "mean"), tv=("tv_effect", "mean"))
    )
    eligible = support[support["in_support"] >= 0.90]
    payload = {
        "warning": WARNING,
        "evidence_tier": "exploratory",
        "family": args.family,
        "n_events": len(frame),
        "n_targets": len(CANDIDATES),
        "in_support_fraction_mean": float(frame["in_support_fraction"].mean()),
        "n_conditions_in_support": int(len(eligible)),
        "n_conditions": int(len(support)),
        "support_rule": ">=90% perturbed states inside validation-calibrated support",
        "interpretation": "model-dependent sensitivity; no biological stimulation equivalence",
        "summary": grouped.to_dict(orient="records"),
        "sha256_config": sha256_file(args.config),
    }
    (reports / "perturbation_overview.json").write_text(
        json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: payload[k] for k in ("n_events", "in_support_fraction_mean", "n_conditions_in_support", "n_conditions")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
