"""Named-neuron virtual perturbations on a free-running simulator.

Every number here is model-dependent.  The module deliberately exposes the
protocol (initial state vs dynamical input), amplitude, onset and model seed so
that a ranking cannot be mistaken for a biological causal effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from neurobranch.data.kato import MACRO_ORDER
from neurobranch.simulator.adapters import NeuralSimulator
from neurobranch.simulator.branching import entropy, js, tv
from neurobranch.simulator.types import WARNING, PerturbationSpec

CANDIDATES = ("AVBL", "AVBR", "AVEL", "AVER", "RIVL", "RIVR")


def outcome_distribution(rollouts, initial: int) -> np.ndarray:
    counts = np.zeros(4, dtype=np.float64)
    for rollout in rollouts:
        labels = rollout.behavior_proba.argmax(axis=1)
        outcome = int(initial)
        for label in labels:
            if int(label) != int(initial):
                outcome = int(label)
                break
        counts[outcome] += 1.0
    return counts / max(1.0, counts.sum())


@dataclass
class Baseline:
    distribution: np.ndarray
    states: np.ndarray  # (n, horizon+1, neurons)
    ood_fraction: float


def baseline_for(
    sim: NeuralSimulator,
    frame: int,
    *,
    horizon: int,
    n_rollouts: int,
    noise_std: float,
    base_seed: int,
) -> Baseline:
    ensemble = sim.rollout_ensemble(
        sim.data.X[int(frame)],
        n_rollouts,
        horizon,
        base_seed=base_seed,
        noise_std=noise_std,
    )
    initial = int(sim.data.y_macro[int(frame)])
    return Baseline(
        distribution=outcome_distribution(ensemble.rollouts, initial),
        states=ensemble.stack,
        ood_fraction=ensemble.ood_fraction,
    )


def perturbed_for(
    sim: NeuralSimulator,
    frame: int,
    spec: PerturbationSpec,
    *,
    horizon: int,
    n_rollouts: int,
    noise_std: float,
    base_seed: int,
):
    return sim.rollout_ensemble(
        sim.data.X[int(frame)],
        n_rollouts,
        horizon,
        base_seed=base_seed,
        noise_std=noise_std,
        perturbation=spec,
    )


def effect_row(
    sim: NeuralSimulator,
    frame: int,
    spec: PerturbationSpec,
    baseline: Baseline,
    *,
    horizon: int,
    n_rollouts: int,
    noise_std: float,
    base_seed: int,
    control_kind: str = "candidate",
    model_seed: int = 17,
) -> dict[str, Any]:
    ensemble = perturbed_for(
        sim,
        frame,
        spec,
        horizon=horizon,
        n_rollouts=n_rollouts,
        noise_std=noise_std,
        base_seed=base_seed,
    )
    initial = int(sim.data.y_macro[int(frame)])
    p_base = baseline.distribution
    p_pert = outcome_distribution(ensemble.rollouts, initial)
    states = ensemble.stack
    n = min(states.shape[0], baseline.states.shape[0])
    diff = states[:n] - baseline.states[:n]
    scale = float(np.maximum(sim.data.X[sim.data.masks["train"]].std(axis=0).mean(), 1e-8))
    shift = float(np.mean(np.linalg.norm(diff, axis=2)))
    terminal = float(np.mean(np.linalg.norm(diff[:, -1], axis=1)))
    return {
        "worm_id": int(sim.data.worm_id),
        "frame": int(frame),
        "target": spec.target,
        "protocol": spec.protocol,
        "mode": spec.mode,
        "amplitude": float(spec.amplitude),
        "onset": int(spec.onset),
        "duration": int(spec.duration),
        "model_seed": int(model_seed),
        "control_kind": control_kind,
        "baseline_entropy": entropy(p_base),
        "perturbed_entropy": entropy(p_pert),
        "delta_tv": tv(p_pert, p_base),
        "delta_js": js(p_pert, p_base),
        "delta_turn": float(p_pert[MACRO_ORDER.index("TURN")] - p_base[MACRO_ORDER.index("TURN")]),
        "delta_forward": float(p_pert[MACRO_ORDER.index("FWD")] - p_base[MACRO_ORDER.index("FWD")]),
        "delta_rev": float(p_pert[MACRO_ORDER.index("REV")] - p_base[MACRO_ORDER.index("REV")]),
        "trajectory_shift": shift,
        "trajectory_shift_sd": shift / scale,
        "terminal_shift": terminal,
        "terminal_shift_sd": terminal / scale,
        "perturbed_ood_fraction": ensemble.ood_fraction,
        "baseline_ood_fraction": baseline.ood_fraction,
        "finite": bool(all(r.finite for r in ensemble.rollouts)),
        "baseline_distribution": ";".join(f"{s}:{p_base[i]:.4f}" for i, s in enumerate(MACRO_ORDER)),
        "perturbed_distribution": ";".join(f"{s}:{p_pert[i]:.4f}" for i, s in enumerate(MACRO_ORDER)),
        "n_rollouts": int(n_rollouts),
        "warning": WARNING,
    }


def candidate_specs(
    directions: tuple[int, ...] = (1, -1),
    *,
    amplitude: float = 1.0,
    onset: int = 0,
    duration: int = 10,
) -> list[PerturbationSpec]:
    return [
        PerturbationSpec(
            target=name,
            protocol="dynamical_input",
            mode="additive",
            amplitude=float(direction * amplitude),
            onset=int(onset),
            duration=int(duration),
        )
        for name in CANDIDATES
        for direction in directions
    ]


def initial_state_specs(amplitude: float = 1.0) -> list[PerturbationSpec]:
    return [
        PerturbationSpec(
            target=name,
            protocol="initial_state",
            mode="additive",
            amplitude=float(amplitude),
            onset=0,
            duration=0,
        )
        for name in CANDIDATES
    ]
