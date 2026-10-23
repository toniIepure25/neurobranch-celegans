"""Counterfactual simulation of virtual neuronal perturbations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from neurobranch.behavior import conditional_entropy
from neurobranch.data.kato import MACRO_ORDER
from neurobranch.interventions.safety import OBSERVATIONAL_STATUS, CausalStatus
from neurobranch.interventions.spec import InterventionSpec, make_perturbation
from neurobranch.preprocessing import WormData, split_starts


@dataclass
class CounterfactualResult:
    spec: InterventionSpec
    worm_id: int
    causal_status: CausalStatus
    baseline_distribution: dict[str, float]
    intervention_distribution: dict[str, float]
    delta: dict[str, float]
    delta_target: float
    delta_ci: tuple[float, float]
    n_initial: int
    n_seeds: int
    mean_trajectory_displacement: float
    max_trajectory_displacement: float
    ood_fraction: float | None
    baseline_turn_rate: float
    per_initial: pd.DataFrame
    notes: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "intervention": self.spec.to_dict(),
            "worm_id": self.worm_id,
            "causal_status": {
                "level": self.causal_status.level,
                "validated": self.causal_status.validated,
                "label": self.causal_status.label,
                "reasons": list(self.causal_status.reasons),
            },
            "baseline_distribution": self.baseline_distribution,
            "intervention_distribution": self.intervention_distribution,
            "delta": self.delta,
            "delta_target": self.delta_target,
            "delta_target_ci": list(self.delta_ci),
            "n_initial": self.n_initial,
            "n_seeds": self.n_seeds,
            "mean_trajectory_displacement": self.mean_trajectory_displacement,
            "max_trajectory_displacement": self.max_trajectory_displacement,
            "ood_fraction": self.ood_fraction,
            "baseline_turn_rate": self.baseline_turn_rate,
            "notes": list(self.notes),
        }


def _outcome_from_labels(initial_label: str, labels: np.ndarray, steps: int) -> str:
    for lab in labels[:steps]:
        nxt = MACRO_ORDER[int(lab)]
        if nxt != initial_label:
            return nxt
    return initial_label


def _initial_frames(data: WormData, behavior: str, horizon: int, n: int, seed: int) -> np.ndarray:
    starts = split_starts(data, "test", history=1, horizon=1)
    starts = starts[starts + horizon < data.n_frames]
    starts = starts[data.y_macro[starts] == MACRO_ORDER.index(behavior)]
    rng = np.random.default_rng(seed)
    if len(starts) > n:
        starts = rng.choice(starts, size=n, replace=False)
    return np.sort(starts)


def run_counterfactual(
    model: Any,
    data: WormData,
    spec: InterventionSpec,
    *,
    seed: int = 17,
    noise_std: float = 0.05,
    status: CausalStatus = OBSERVATIONAL_STATUS,
) -> CounterfactualResult:
    """Paired baseline/intervention rollouts with common random numbers."""
    spec.validate()
    frames = _initial_frames(data, spec.initial_behavior, spec.horizon, spec.n_initial, seed)
    if len(frames) == 0:
        raise ValueError(f"No test frames with behaviour {spec.initial_behavior}")
    perturb, inds = make_perturbation(data, spec)
    base_counts = {s: 0 for s in MACRO_ORDER}
    int_counts = {s: 0 for s in MACRO_ORDER}
    rows = []
    displacements = []
    ood_hits, ood_total = 0, 0
    train_X = data.X[data.masks["train"]]
    ood_nn = None
    ood_threshold = None
    if hasattr(model, "ood_score"):
        ood_threshold = float(np.quantile(model.ood_score(train_X), 0.99))
    else:
        from sklearn.neighbors import NearestNeighbors

        ood_nn = NearestNeighbors(n_neighbors=1).fit(train_X)
        train_d = ood_nn.kneighbors(train_X, n_neighbors=2, return_distance=True)[0][:, 1]
        ood_threshold = float(np.quantile(train_d, 0.99))
    for t in frames:
        x0 = data.X[int(t)]
        current = MACRO_ORDER[data.y_macro[int(t)]]
        for s in spec.seeds:
            base = model.rollout(x0, spec.horizon, seed=int(seed + s), noise_std=noise_std)
            inter = model.rollout(
                x0, spec.horizon, seed=int(seed + s), noise_std=noise_std, perturb=perturb
            )
            base_lab = model.behavior_proba(base[:-1]).argmax(axis=1)
            int_lab = model.behavior_proba(inter[:-1]).argmax(axis=1)
            ob = _outcome_from_labels(current, base_lab, spec.horizon)
            oi = _outcome_from_labels(current, int_lab, spec.horizon)
            base_counts[ob] += 1
            int_counts[oi] += 1
            disp = np.sqrt(np.mean((base - inter) ** 2, axis=1))
            displacements.extend(disp.tolist())
            if ood_threshold is not None:
                if ood_nn is not None:
                    d = ood_nn.kneighbors(inter[1:], n_neighbors=1, return_distance=True)[0][:, 0]
                    ood_hits += int(np.sum(d > ood_threshold))
                else:
                    ood_hits += int(np.sum(model.ood_score(inter[1:]) > ood_threshold))
                ood_total += len(inter) - 1
            rows.append(
                {
                    "frame": int(t),
                    "sim_seed": int(s),
                    "current_behavior": current,
                    "baseline_outcome": ob,
                    "intervention_outcome": oi,
                    "changed": ob != oi,
                    "mean_displacement": float(disp.mean()),
                    "target_indices": ",".join(map(str, inds)),
                }
            )
    n = max(1, sum(base_counts.values()))
    b = {s: base_counts[s] / n for s in MACRO_ORDER}
    q = {s: int_counts[s] / n for s in MACRO_ORDER}
    delta = {s: q[s] - b[s] for s in MACRO_ORDER}
    target = spec.target_behavior
    per_initial = pd.DataFrame(rows)
    # Cluster bootstrap over initial frames (not simulation seeds).
    rng = np.random.default_rng(seed)
    frame_deltas = []
    for t in frames:
        sub = per_initial[per_initial.frame == t]
        if len(sub):
            frame_deltas.append(float(np.mean(sub.intervention_outcome == target) - np.mean(sub.baseline_outcome == target)))
    frame_deltas = np.asarray(frame_deltas)
    if len(frame_deltas):
        boots = [
            float(np.mean(rng.choice(frame_deltas, size=len(frame_deltas), replace=True)))
            for _ in range(500)
        ]
        ci = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
    else:
        ci = (float("nan"), float("nan"))
    notes = []
    if ood_threshold is not None and ood_total:
        frac = ood_hits / ood_total
        if frac > 0.1:
            notes.append(
                f"{frac:.1%} of intervention states lie beyond the 99th percentile of training latent support"
            )
    if abs(delta[target]) < 0.1:
        notes.append("Target transition-probability change is small (<0.10) and may not be robust")
    if len(spec.targets) == 1 and spec.amplitude >= 2.0:
        notes.append("Amplitude is large in standardized model units; interpret as a stress test, not a dose")
    return CounterfactualResult(
        spec=spec,
        worm_id=data.worm_id,
        causal_status=status,
        baseline_distribution=b,
        intervention_distribution=q,
        delta=delta,
        delta_target=float(delta[target]),
        delta_ci=ci,
        n_initial=int(len(frames)),
        n_seeds=len(spec.seeds),
        mean_trajectory_displacement=float(np.mean(displacements)) if displacements else float("nan"),
        max_trajectory_displacement=float(np.max(displacements)) if displacements else float("nan"),
        ood_fraction=float(ood_hits / ood_total) if ood_total else None,
        baseline_turn_rate=float(b[target]),
        per_initial=per_initial,
        notes=tuple(notes),
    )


def compare_random_effects(
    results: list[CounterfactualResult],
    target: str = "TURN",
) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "name": r.spec.name,
                "targets": ",".join(r.spec.targets),
                "mode": r.spec.mode,
                "amplitude": r.spec.amplitude,
                "delta_target": r.delta[target],
                "ci_low": r.delta_ci[0],
                "ci_high": r.delta_ci[1],
                "ood_fraction": r.ood_fraction,
                "displacement": r.mean_trajectory_displacement,
            }
        )
    return pd.DataFrame(rows)


def transition_entropy(distribution: dict[str, float]) -> float:
    return conditional_entropy(distribution)
