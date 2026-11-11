"""Approximate power for a cluster-randomized binary intervention.

The simulation is deliberately modest: animal-level random effects, a fixed
number of trials per animal, and a two-sample test on animal-level proportions.
It is a design tool, not a substitute for the lab's own pilot variance.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit
from scipy.stats import ttest_ind


def _check_probability(value: float, name: str) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be a probability strictly between 0 and 1")


def simulate_power(
    *,
    animals_per_arm: int,
    trials_per_animal: int,
    p_sham: float,
    p_target: float,
    icc: float,
    reps: int = 500,
    alpha: float = 0.05,
    seed: int = 20261111,
) -> float:
    """Return simulated power for a two-sided animal-level test.

    Animal effects are drawn on the logit scale with ``sigma =
    sqrt(ICC / (1 - ICC))``.  This is an approximation to the intended
    intraclass correlation, not an exact beta-binomial model.
    """
    if animals_per_arm < 2:
        raise ValueError("animals_per_arm must be at least 2")
    if trials_per_animal < 1:
        raise ValueError("trials_per_animal must be positive")
    if reps < 1:
        raise ValueError("reps must be positive")
    if not 0.0 < icc < 1.0:
        raise ValueError("icc must be in (0, 1)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    _check_probability(p_sham, "p_sham")
    _check_probability(p_target, "p_target")

    rng = np.random.default_rng(seed)
    sigma = float(np.sqrt(icc / (1.0 - icc)))
    effects = rng.normal(0.0, sigma, size=(reps, 2, animals_per_arm))
    base = np.log(np.array([p_sham, p_target]) / (1.0 - np.array([p_sham, p_target])))
    probabilities = expit(base[None, :, None] + effects)
    counts = rng.binomial(trials_per_animal, probabilities)
    proportions = counts / trials_per_animal
    result = ttest_ind(
        proportions[:, 0, :],
        proportions[:, 1, :],
        axis=1,
        equal_var=False,
    )
    return float(np.mean(result.pvalue < alpha))


def minimum_detectable_effect(
    *,
    animals_per_arm: int,
    trials_per_animal: int,
    p_sham: float,
    icc: float,
    target_power: float = 0.8,
    reps: int = 300,
    alpha: float = 0.05,
    seed: int = 20261111,
    grid: tuple[float, ...] = tuple(float(x) for x in np.arange(0.05, 0.61, 0.025)),
) -> float:
    """Smallest absolute risk difference reaching ``target_power``."""
    if not 0.0 < target_power < 1.0:
        raise ValueError("target_power must be in (0, 1)")
    for effect in grid:
        p_target = min(p_sham + effect, 0.95)
        if p_target <= p_sham:
            continue
        power = simulate_power(
            animals_per_arm=animals_per_arm,
            trials_per_animal=trials_per_animal,
            p_sham=p_sham,
            p_target=p_target,
            icc=icc,
            reps=reps,
            alpha=alpha,
            seed=seed,
        )
        if power >= target_power:
            return float(effect)
    return float("nan")


def build_power_report(
    *,
    animals_per_arm: tuple[int, ...] = (6, 12),
    iccs: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4),
    design_icc: float = 0.2,
    trials_per_animal: int = 30,
    p_sham: float = 0.30,
    reps: int = 300,
    alpha: float = 0.05,
    target_power: float = 0.8,
    seed: int = 20261111,
) -> dict[str, Any]:
    """Build a reproducible screening/confirmation power report."""
    if design_icc not in iccs:
        raise ValueError("design_icc must be one of iccs")
    designs: list[dict[str, Any]] = []
    for animals in animals_per_arm:
        mde = minimum_detectable_effect(
            animals_per_arm=animals,
            trials_per_animal=trials_per_animal,
            p_sham=p_sham,
            icc=design_icc,
            target_power=target_power,
            reps=reps,
            alpha=alpha,
            seed=seed,
        )
        designs.append(
            {
                "animals_per_arm": animals,
                "trials_per_animal": trials_per_animal,
                "mde": mde,
                "power": (
                    simulate_power(
                        animals_per_arm=animals,
                        trials_per_animal=trials_per_animal,
                        p_sham=p_sham,
                        p_target=min(p_sham + mde, 0.95),
                        icc=design_icc,
                        reps=reps,
                        alpha=alpha,
                        seed=seed,
                    )
                    if np.isfinite(mde)
                    else 0.0
                ),
            }
        )
    sensitivity = [
        {
            "icc": icc,
            "animals_per_arm": animals_per_arm[0],
            "power_at_0.20_effect": simulate_power(
                animals_per_arm=animals_per_arm[0],
                trials_per_animal=trials_per_animal,
                p_sham=p_sham,
                p_target=min(p_sham + 0.20, 0.95),
                icc=icc,
                reps=reps,
                alpha=alpha,
                seed=seed,
            ),
        }
        for icc in iccs
    ]
    return {
        "format": "vienna-intervention-power-v1",
        "seed": seed,
        "assumptions": {
            "p_sham": p_sham,
            "design_icc": design_icc,
            "trials_per_animal": trials_per_animal,
            "alpha": alpha,
            "target_power": target_power,
            "reps": reps,
            "icc_model": "logit-normal animal effect, sigma=sqrt(ICC/(1-ICC))",
            "test": "two-sided Welch t-test on animal-level proportions",
        },
        "designs": designs,
        "sensitivity": sensitivity,
    }
