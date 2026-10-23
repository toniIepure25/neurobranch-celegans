"""Sparse virtual-perturbation search.

This is hypothesis generation over a fitted simulator.  Ranking neurons by a
simulated effect is not a biological discovery and does not identify a causal
effect.  Random-neuron and amplitude controls are mandatory companions.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from neurobranch.interventions.engine import CounterfactualResult, run_counterfactual
from neurobranch.interventions.spec import InterventionSpec
from neurobranch.preprocessing import WormData

# Literature-motivated candidates, restricted at runtime to cells present in
# the predictor set.  AVA/SMD/RIB are absent by design (behaviour-label leak).
CANDIDATE_CELLS = (
    "AVEL",
    "AVER",
    "AVBL",
    "AVBR",
    "AIBL",
    "AIBR",
    "RIML",
    "RIMR",
    "RIVL",
    "RIVR",
    "AIAL",
    "AIAR",
    "AVHL",
    "AVHR",
    "AVFR",
    "VA01",
    "VB01",
    "DB01",
)


def available_candidates(data: WormData, preferred: Iterable[str] = CANDIDATE_CELLS) -> list[str]:
    return [n for n in preferred if n in data.predictor_names]


def random_control_pool(data: WormData) -> list[str]:
    """Named predictor cells that are not in the hypothesis-driven candidate set.

    Sampling controls from the full pool would include candidates and make the
    specificity comparison circular.  This helper is the single definition used
    by the control runner and by the regression test.
    """
    candidates = set(available_candidates(data))
    return [
        n
        for n in data.predictor_names
        if any(ch.isalpha() for ch in n) and n not in candidates
    ]


def run_sparse_search(
    model,
    data: WormData,
    *,
    seed: int = 17,
    amplitudes: tuple[float, ...] = (-2.0, 2.0),
    n_initial: int = 24,
    sim_seeds: tuple[int, ...] = (0, 1),
    horizon: int = 12,
    max_candidates: int = 12,
    target: str = "TURN",
) -> tuple[pd.DataFrame, list[CounterfactualResult]]:
    cands = available_candidates(data)[:max_candidates]
    results: list[CounterfactualResult] = []
    for name in cands:
        for amp in amplitudes:
            spec = InterventionSpec(
                name=f"{name}_{amp:+.1f}",
                targets=(name,),
                mode="additive",
                amplitude=amp,
                onset=0,
                duration=horizon,
                horizon=horizon,
                n_initial=n_initial,
                seeds=sim_seeds,
                initial_behavior="REV",
                target_behavior=target,
            )
            results.append(run_counterfactual(model, data, spec, seed=seed))
    rows = []
    for r in results:
        rows.append(
            {
                "candidate": r.spec.targets[0],
                "amplitude": r.spec.amplitude,
                "target": target,
                "baseline_prob": r.baseline_distribution[target],
                "intervention_prob": r.intervention_distribution[target],
                "delta_prob": r.delta[target],
                "ci_low": r.delta_ci[0],
                "ci_high": r.delta_ci[1],
                "mean_displacement": r.mean_trajectory_displacement,
                "ood_fraction": r.ood_fraction,
                "n_initial": r.n_initial,
                "n_seeds": r.n_seeds,
            }
        )
    df = pd.DataFrame(rows)
    if len(df):
        df["abs_delta"] = df.delta_prob.abs()
        df = df.sort_values("abs_delta", ascending=False).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
    return df, results


def random_control(
    model,
    data: WormData,
    *,
    n_random: int = 10,
    amplitude: float = 2.0,
    seed: int = 17,
    n_initial: int = 24,
    sim_seeds: tuple[int, ...] = (0, 1),
    horizon: int = 12,
    target: str = "TURN",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Controls must be disjoint from the hypothesis-driven candidate set.  Sampling
    # from the same list and calling it "random" would make the comparison circular.
    pool = random_control_pool(data)
    if not pool:
        raise ValueError("No non-candidate named predictor cells available for controls")
    picks = rng.choice(pool, size=min(n_random, len(pool)), replace=False)
    rows = []
    for name in picks:
        spec = InterventionSpec(
            name=f"random_{name}",
            targets=(name,),
            mode="additive",
            amplitude=amplitude,
            duration=horizon,
            horizon=horizon,
            n_initial=n_initial,
            seeds=sim_seeds,
            initial_behavior="REV",
            target_behavior=target,
            notes="random control",
        )
        r = run_counterfactual(model, data, spec, seed=seed)
        rows.append(
            {
                "neuron": name,
                "delta_prob": r.delta[target],
                "ci_low": r.delta_ci[0],
                "ci_high": r.delta_ci[1],
                "ood_fraction": r.ood_fraction,
                "mean_displacement": r.mean_trajectory_displacement,
            }
        )
    return pd.DataFrame(rows)
