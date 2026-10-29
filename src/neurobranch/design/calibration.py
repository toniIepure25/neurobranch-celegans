"""Simulation-based calibration of the reliability interval.

This is a frequentist coverage study, not a Bayesian SBC rank test: synthetic
atlases are generated from known variance components and the animal jackknife
interval is checked against the known fold-level reliability.  It answers one
question only: does the reported interval cover the value it claims to cover?

The simulator and estimator share the additive model stated in
``neurobranch.design.ceiling``; coverage under model misspecification is not
tested here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from neurobranch.benchmark.robustness import animal_jackknife


def simulate_atlas(
    *,
    n_animals: int = 12,
    n_targets: int = 3,
    n_responders: int = 6,
    n_events: int = 3,
    signal_sd: float = 1.0,
    noise_sd: float = 1.0,
    interaction_sd: float = 0.5,
    seed: int = 17,
) -> pd.DataFrame:
    """Generate a synthetic atlas with known additive components."""
    rng = np.random.default_rng(seed)
    profiles = rng.normal(0.0, signal_sd, size=(n_targets, n_responders))
    rows: list[dict[str, float | str]] = []
    for target_index in range(n_targets):
        for animal in range(n_animals):
            deviations = rng.normal(0.0, interaction_sd, size=n_responders)
            for responder in range(n_responders):
                for event in range(n_events):
                    value = float(
                        profiles[target_index, responder]
                        + deviations[responder]
                        + rng.normal(0.0, noise_sd)
                    )
                    rows.append(
                        {
                            "dataset_id": f"a{animal}",
                            "target": f"T{target_index}",
                            "neuron": f"N{responder}",
                            "event_id": f"{animal}:{target_index}:{responder}:{event}",
                            "response_mean": value,
                        }
                    )
    return pd.DataFrame(rows)


def true_reliability(
    *,
    signal_sd: float = 1.0,
    noise_sd: float = 1.0,
    interaction_sd: float = 0.5,
    fold_size: int = 12,
    n_events: int = 3,
) -> float:
    """Population reliability at ``fold_size`` under the simulator.

    The estimator works on cell means, so per-event noise is divided by
    ``n_events`` before it enters the reliability formula.
    """
    signal = float(signal_sd) ** 2
    error = float(noise_sd) ** 2 / float(n_events) + float(interaction_sd) ** 2
    return signal / (signal + error / float(fold_size))


def coverage_study(
    *,
    n_reps: int = 30,
    fold_size: int = 12,
    n_animals: int = 12,
    n_targets: int = 3,
    n_responders: int = 6,
    n_events: int = 3,
    signal_sd: float = 1.0,
    noise_sd: float = 1.0,
    interaction_sd: float = 0.5,
    seed: int = 23,
) -> dict[str, float]:
    """Run the coverage study and return coverage, width and bias."""
    if n_reps < 5:
        raise ValueError("n_reps must be >= 5")
    truth = true_reliability(
        signal_sd=signal_sd,
        noise_sd=noise_sd,
        interaction_sd=interaction_sd,
        fold_size=fold_size,
        n_events=n_events,
    )
    covered = 0
    widths: list[float] = []
    estimates: list[float] = []
    for rep in range(n_reps):
        frame = simulate_atlas(
            n_animals=n_animals,
            n_targets=n_targets,
            n_responders=n_responders,
            n_events=n_events,
            signal_sd=signal_sd,
            noise_sd=noise_sd,
            interaction_sd=interaction_sd,
            seed=seed + rep,
        )
        result = animal_jackknife(frame, fold_size=fold_size, min_animals=6)
        if not np.isfinite(result["estimate"]):
            continue
        estimates.append(result["estimate"])
        widths.append(result["ci_high"] - result["ci_low"])
        if result["ci_low"] <= truth <= result["ci_high"]:
            covered += 1
    if not estimates:
        return {
            "truth": truth,
            "coverage": float("nan"),
            "mean_width": float("nan"),
            "bias": float("nan"),
            "n_reps": 0.0,
        }
    values = np.asarray(estimates, dtype=float)
    return {
        "truth": truth,
        "coverage": covered / values.size,
        "mean_width": float(np.mean(widths)),
        "bias": float(values.mean() - truth),
        "n_reps": float(values.size),
    }


__all__ = ["coverage_study", "simulate_atlas", "true_reliability"]
