"""Synthetic ground-truth scenarios for auditing branching statistics.

These are *not* recordings and are never presented as data.  They are unit-test
fixtures with known generative structure, used to check that the detector
responds sensibly to deterministic outcomes, stochastic branches, projection
artefacts, unobserved state, and rare transitions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurobranch.data.kato import MACRO_ORDER


@dataclass
class BranchingScenario:
    name: str
    Z: np.ndarray
    Z_high: np.ndarray
    behavior: np.ndarray
    candidate_frames: np.ndarray
    hidden_state: np.ndarray | None
    ground_truth: str


def _scenario_arrays(
    kind: str,
    *,
    n_candidates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, str]:
    rng = np.random.default_rng(seed)
    fwd = MACRO_ORDER.index("FWD")
    rev = MACRO_ORDER.index("REV")
    turn = MACRO_ORDER.index("TURN")
    n_frames = 20 + 40 * n_candidates + 20
    behavior = np.full(n_frames, rev, dtype=np.int64)
    candidates = np.arange(20, 20 + 40 * n_candidates, 40, dtype=int)
    hidden = rng.integers(0, 2, size=len(candidates))

    if kind == "deterministic":
        outcomes = np.full(len(candidates), fwd)
        truth = "one deterministic outcome"
    elif kind == "stochastic":
        outcomes = rng.choice([fwd, rev, turn], size=len(candidates), p=[0.4, 0.4, 0.2])
        truth = "three-way stochastic branch"
    elif kind == "rare":
        outcomes = np.full(len(candidates), fwd)
        outcomes[rng.choice(len(candidates), max(1, len(candidates) // 15), replace=False)] = rev
        truth = "one dominant outcome with rare alternative"
    elif kind == "hidden":
        outcomes = np.where(hidden == 0, fwd, rev)
        truth = "outcome determined by an unobserved binary state"
    elif kind == "projection":
        outcomes = np.where(hidden == 0, fwd, rev)
        truth = "two high-dimensional states collapsed by a 3-D projection"
    else:
        raise ValueError(f"unknown scenario {kind!r}")

    # Write one outcome after a short REV lead-in per candidate segment.
    for t, out in zip(candidates, outcomes):
        behavior[t : t + 4] = rev
        behavior[t + 4 : t + 20] = out
        behavior[t + 20 : t + 40] = rev

    # Candidate geometry.  Deterministic/rare outcomes share one location; the
    # stochastic and hidden-state cases deliberately mix outcomes at one point.
    if kind == "projection":
        # High-dimensional clusters are well separated, but the first 3
        # coordinates are identical for both clusters.  A 3-D projection
        # therefore cannot separate them.
        high = rng.normal(scale=0.15, size=(len(candidates), 8))
        high[hidden == 0, 0] = -4.0
        high[hidden == 1, 0] = 4.0
        cand_high = high
        # The projection uses coordinates that carry no cluster information.
        cand_z = high[:, 3:6]
    else:
        centers = rng.normal(scale=0.35, size=(len(candidates), 3))
        if kind in {"deterministic", "rare", "stochastic", "hidden"}:
            centers[:] = np.array([0.0, 0.0, 0.0])
        cand_high = centers + rng.normal(scale=0.05, size=centers.shape)
        cand_z = cand_high

    # local_entropies indexes a full-length latent matrix, so scatter candidate
    # embeddings back into frame space.  Non-candidate frames are irrelevant.
    Z = np.zeros((n_frames, cand_z.shape[1]), dtype=float)
    Z_high = np.zeros((n_frames, cand_high.shape[1]), dtype=float)
    Z[candidates] = cand_z
    Z_high[candidates] = cand_high
    return Z, Z_high, behavior, candidates, hidden, truth


def make_scenario(kind: str, *, n_candidates: int = 60, seed: int = 0) -> BranchingScenario:
    """Create a labelled synthetic branching scenario.

    Parameters
    ----------
    kind:
        ``deterministic``, ``stochastic``, ``rare``, ``hidden``, or
        ``projection``.
    n_candidates:
        Number of candidate states (each separated by 40 frames).
    seed:
        Random seed for outcomes and geometry.
    """
    Z, Z_high, behavior, cand, hidden, truth = _scenario_arrays(
        kind, n_candidates=n_candidates, seed=seed
    )
    return BranchingScenario(
        name=kind,
        Z=Z,
        Z_high=Z_high,
        behavior=behavior,
        candidate_frames=cand,
        hidden_state=hidden,
        ground_truth=truth,
    )


def outcomes_for(scenario: BranchingScenario, horizon: int = 20) -> np.ndarray:
    """First macro state different from the current state within the horizon."""
    from neurobranch.behavior import next_state_after_leaving

    labels = [MACRO_ORDER[i] for i in scenario.behavior]
    out = []
    for t in scenario.candidate_frames:
        nxt = next_state_after_leaving(labels, int(t), horizon)
        out.append(nxt[0] if nxt is not None else labels[int(t)])
    return np.asarray(out)
