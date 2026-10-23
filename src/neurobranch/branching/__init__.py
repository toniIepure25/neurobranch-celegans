"""Empirical and model-based branching analysis.

Terminology is deliberate.  A visible split in a projected trajectory is *not*
a bifurcation.  We estimate conditional distributions of the next behavioural
state reached after the current state, and quantify neighbourhood diversity.
Only a parameter-dependent change in the vector field would justify the word
bifurcation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from neurobranch.behavior import conditional_entropy, next_state_after_leaving
from neurobranch.data.kato import MACRO_ORDER
from neurobranch.preprocessing import WormData, split_starts


@dataclass
class LocalOutcome:
    frame: int
    entropy: float
    majority_probability: float
    counts: dict[str, int]
    probabilities: dict[str, float]
    n_neighbors: int
    latent_distance: float


def _outcome(data: WormData, t: int, horizon: int) -> str:
    out = next_state_after_leaving([MACRO_ORDER[i] for i in data.y_macro], int(t), horizon)
    if out is None:
        return MACRO_ORDER[int(data.y_macro[int(t)])]
    return out[0]


def _entropy_from_counts(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    p = {k: v / total for k, v in counts.items()}
    return conditional_entropy(p)


def local_outcome_diversity(
    Z: np.ndarray,
    data: WormData,
    candidate_frames: np.ndarray,
    *,
    k: int = 40,
    horizon: int = 20,
    exclude_temporal_radius: int = 12,
    rng_seed: int = 0,
) -> list[LocalOutcome]:
    """kNN outcome entropy among behaviourally matched test states.

    Neighbours within +/- ``exclude_temporal_radius`` frames are removed so a
    query cannot retrieve its own continuation and manufacture certainty.
    """
    candidate_frames = np.asarray(candidate_frames, dtype=int)
    if len(candidate_frames) == 0:
        return []
    Zc = Z[candidate_frames]
    outcomes = np.array([_outcome(data, int(t), horizon) for t in candidate_frames])
    results: list[LocalOutcome] = []
    for qi, t in enumerate(candidate_frames):
        dist = np.linalg.norm(Zc - Zc[qi], axis=1)
        allowed = np.abs(candidate_frames - t) > exclude_temporal_radius
        allowed[qi] = False
        if allowed.sum() < 3:
            continue
        idx = np.flatnonzero(allowed)
        kk = min(k, len(idx))
        idx = idx[np.argpartition(dist[idx], kk - 1)[:kk]]
        counts = {s: int(np.sum(outcomes[idx] == s)) for s in MACRO_ORDER}
        total = max(1, sum(counts.values()))
        probs = {s: c / total for s, c in counts.items()}
        ent = _entropy_from_counts(counts)
        results.append(
            LocalOutcome(
                frame=int(t),
                entropy=float(ent),
                majority_probability=float(max(probs.values())),
                counts=counts,
                probabilities=probs,
                n_neighbors=int(total),
                latent_distance=float(dist[idx].mean()),
            )
        )
    return results


def permutation_branching_test(
    Z: np.ndarray,
    data: WormData,
    candidate_frames: np.ndarray,
    *,
    k: int = 40,
    horizon: int = 20,
    n_permutations: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Test whether local outcome entropy exceeds outcome-shuffled chance."""
    obs = local_outcome_diversity(Z, data, candidate_frames, k=k, horizon=horizon, rng_seed=seed)
    if not obs:
        return {"mean_entropy": float("nan"), "null_mean": float("nan"), "p_value": float("nan")}
    outcomes = np.array([_outcome(data, int(t), horizon) for t in candidate_frames])
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(outcomes)
        # Recompute with labels permuted, preserving geometry.
        vals = []
        Zc = Z[np.asarray(candidate_frames)]
        for qi, t in enumerate(candidate_frames):
            dist = np.linalg.norm(Zc - Zc[qi], axis=1)
            allowed = np.abs(np.asarray(candidate_frames) - t) > 12
            allowed[qi] = False
            idx = np.flatnonzero(allowed)
            if len(idx) < 3:
                continue
            kk = min(k, len(idx))
            idx = idx[np.argpartition(dist[idx], kk - 1)[:kk]]
            counts = {s: int(np.sum(shuffled[idx] == s)) for s in MACRO_ORDER}
            vals.append(_entropy_from_counts(counts))
        null.append(float(np.mean(vals)))
    observed_mean = float(np.mean([o.entropy for o in obs]))
    p = float((1 + np.sum(np.array(null) >= observed_mean)) / (1 + len(null)))
    return {
        "mean_entropy": observed_mean,
        "null_mean": float(np.mean(null)),
        "null_sd": float(np.std(null)),
        "excess_entropy_bits": observed_mean - float(np.mean(null)),
        "p_value": p,
        "n_candidate_states": int(len(obs)),
    }


def candidate_frames(data: WormData, *, behavior: str = "REV", split: str = "test", horizon: int = 20) -> np.ndarray:
    idx = split_starts(data, split, history=1, horizon=1)
    idx = idx[(idx + horizon < data.n_frames)]
    target = MACRO_ORDER.index(behavior)
    return idx[data.y_macro[idx] == target]


def entropy_across_latent_dims(
    data: WormData,
    *,
    dims: tuple[int, ...] = (3, 8, 16, 32),
    k: int = 40,
    horizon: int = 20,
) -> dict[str, Any]:
    """Check whether branching score survives a change of projection width."""
    from sklearn.decomposition import PCA

    starts = split_starts(data, "train", history=1, horizon=1)
    cand = candidate_frames(data, horizon=horizon)
    train_X = data.X[starts]
    scores = {}
    for d in dims:
        pca = PCA(n_components=min(d, train_X.shape[1]), random_state=0, svd_solver="randomized")
        pca.fit(train_X)
        Z = pca.transform(data.X)
        res = local_outcome_diversity(Z, data, cand, k=k, horizon=horizon)
        scores[f"dim_{d}"] = {int(r.frame): float(r.entropy) for r in res}
    common = sorted(set.intersection(*(set(v) for v in scores.values()))) if scores else []
    correlations = {}
    base = f"dim_{dims[0]}"
    for name, vals in scores.items():
        if name == base or not common:
            continue
        a = np.array([scores[base][t] for t in common])
        b = np.array([vals[t] for t in common])
        if np.std(a) == 0 or np.std(b) == 0:
            correlations[name] = float("nan")
        else:
            correlations[name] = float(spearmanr(a, b).statistic)
    return {"scores": scores, "spearman_vs_base": correlations, "dims": list(dims)}


def model_branching(
    model: Any,
    data: WormData,
    candidate_frames_arr: np.ndarray,
    *,
    steps: int = 20,
    n_seeds: int = 5,
    behavior: str = "REV",
) -> dict[str, Any]:
    """Outcome distribution from open-loop model rollouts from the same states."""
    frames = candidate_frames_arr
    counts = {s: 0 for s in MACRO_ORDER}
    per_frame = {}
    for t in frames:
        local = {s: 0 for s in MACRO_ORDER}
        for seed in range(n_seeds):
            seq = model.rollout(data.X[int(t)], steps, seed=int(seed))
            labels = model.behavior_proba(seq[:-1]).argmax(axis=1)
            current = MACRO_ORDER[data.y_macro[int(t)]]
            outcome = current
            for lab in labels:
                nxt = MACRO_ORDER[int(lab)]
                if nxt != current:
                    outcome = nxt
                    break
            counts[outcome] += 1
            local[outcome] += 1
        per_frame[int(t)] = {s: c / n_seeds for s, c in local.items()}
    total = max(1, sum(counts.values()))
    probs = {s: c / total for s, c in counts.items()}
    return {
        "probabilities": probs,
        "entropy": conditional_entropy(probs),
        "per_frame": per_frame,
        "n_frames": int(len(frames)),
        "n_seeds": n_seeds,
    }


def summarize_branching(results: list[LocalOutcome], top_n: int = 10) -> dict[str, Any]:
    if not results:
        return {"n": 0}
    ent = np.array([r.entropy for r in results])
    maj = np.array([r.majority_probability for r in results])
    top = sorted(results, key=lambda r: r.entropy, reverse=True)[:top_n]
    return {
        "n": len(results),
        "mean_entropy_bits": float(ent.mean()),
        "median_entropy_bits": float(np.median(ent)),
        "fraction_entropy_gt_0_5": float(np.mean(ent > 0.5)),
        "mean_majority_probability": float(maj.mean()),
        "top_frames": [
            {
                "frame": r.frame,
                "entropy": r.entropy,
                "majority_probability": r.majority_probability,
                "counts": r.counts,
                "probabilities": r.probabilities,
            }
            for r in top
        ],
    }
