"""Corrected statistics for local trajectory-outcome diversity.

The original Phase 1 test shuffled outcome labels *i.i.d.* across candidate
frames and used a one-sided test for excess entropy.  That answers a narrow
question ("is local diversity greater than expected if outcomes were unrelated
to geometry?") but not the reverse question, and it destroys temporal
autocorrelation.  This module keeps the original statistic but adds:

* explicit one-sided excess, one-sided deficit, and two-sided p-values;
* an i.i.d. null, a contiguous-block null, and a circular-shift null;
* a moving-block bootstrap interval for the mean local entropy.

All tests use the same observed geometry and neighbourhood construction.  The
null changes only the assignment of outcomes to candidate states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from neurobranch.behavior import conditional_entropy
from neurobranch.data.kato import MACRO_ORDER


@dataclass(frozen=True)
class NullTestResult:
    observed_mean_entropy: float
    null_mean: float
    null_sd: float
    null_q025: float
    null_q975: float
    excess_entropy_bits: float
    p_excess: float
    p_deficit: float
    p_two_sided: float
    method: str
    block_size: int | None
    n_candidates: int
    n_permutations: int
    n_valid_null: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_entropy": self.observed_mean_entropy,
            "null_mean": self.null_mean,
            "null_sd": self.null_sd,
            "null_q025": self.null_q025,
            "null_q975": self.null_q975,
            "excess_entropy_bits": self.excess_entropy_bits,
            "p_value": self.p_excess,
            "p_excess": self.p_excess,
            "p_deficit": self.p_deficit,
            "p_two_sided": self.p_two_sided,
            "null_method": self.method,
            "null_block_size": self.block_size,
            "n_candidate_states": self.n_candidates,
            "n_permutations": self.n_permutations,
            "n_valid_null": self.n_valid_null,
            "test_direction": "excess = observed > null; deficit = observed < null",
        }


def entropy_from_counts(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return conditional_entropy({k: v / total for k, v in counts.items()})


def local_entropies(
    Z: np.ndarray,
    candidate_frames: np.ndarray,
    outcomes: np.ndarray,
    *,
    k: int = 40,
    exclude_temporal_radius: int = 12,
) -> np.ndarray:
    """Mean-free per-query entropy helper shared by observed and null tests."""
    candidate_frames = np.asarray(candidate_frames, dtype=int)
    outcomes = np.asarray(outcomes)
    Zc = np.asarray(Z)[candidate_frames]
    vals: list[float] = []
    for qi, t in enumerate(candidate_frames):
        dist = np.linalg.norm(Zc - Zc[qi], axis=1)
        allowed = np.abs(candidate_frames - t) > exclude_temporal_radius
        allowed[qi] = False
        idx = np.flatnonzero(allowed)
        if len(idx) < 3:
            continue
        kk = min(k, len(idx))
        idx = idx[np.argpartition(dist[idx], kk - 1)[:kk]]
        counts = {s: int(np.sum(outcomes[idx] == s)) for s in MACRO_ORDER}
        vals.append(entropy_from_counts(counts))
    return np.asarray(vals, dtype=float)


def block_permute(labels: np.ndarray, rng: np.random.Generator, block_size: int) -> np.ndarray:
    """Permute contiguous blocks of a label sequence.

    Blocks are shuffled without replacement, preserving short-range order
    *within* each block while destroying its relation to the candidate geometry.
    """
    labels = np.asarray(labels)
    n = len(labels)
    if block_size < 1 or block_size >= n:
        return rng.permutation(labels)
    blocks = [labels[i : i + block_size] for i in range(0, n, block_size)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order])


def circular_shift(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    labels = np.asarray(labels)
    n = len(labels)
    if n < 2:
        return labels.copy()
    shift = int(rng.integers(1, n))
    return np.roll(labels, shift)


def moving_block_bootstrap_mean(
    values: np.ndarray,
    *,
    block_size: int = 10,
    n_bootstrap: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile interval for a mean of ordered observations.

    Candidate frames are ordered in time but overlap in their outcome horizons,
    so the i.i.d. bootstrap understates uncertainty.  A moving-block bootstrap
    keeps local dependence within blocks.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    if n == 1 or block_size >= n:
        return {"mean": float(x.mean()), "ci_low": float(x.mean()), "ci_high": float(x.mean())}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=(n_bootstrap, n_blocks))
    offsets = np.arange(block_size)
    means = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = ((starts[b, :, None] + offsets) % n).ravel()[:n]
        means[b] = x[idx].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(x.mean()), "ci_low": float(lo), "ci_high": float(hi), "n_bootstrap": n_bootstrap}


def null_branching_test(
    Z: np.ndarray,
    candidate_frames: np.ndarray,
    outcomes: np.ndarray,
    *,
    k: int = 40,
    exclude_temporal_radius: int = 12,
    n_permutations: int = 1000,
    method: str = "block",
    block_size: int | None = None,
    seed: int = 0,
    bootstrap: int = 1000,
) -> dict[str, Any]:
    """Test local outcome entropy against a geometry-unrelated null.

    ``method`` is ``"iid"`` (Phase 1 behaviour), ``"block"`` (contiguous block
    permutation), or ``"shift"`` (circular shift of the outcome sequence).  The
    latter two retain short-range temporal structure, so they are more
    conservative nulls than i.i.d. shuffling.
    """
    cand = np.asarray(candidate_frames, dtype=int)
    y = np.asarray(outcomes)
    if len(cand) == 0:
        raise ValueError("no candidate frames")
    if method not in {"iid", "block", "shift"}:
        raise ValueError(f"unknown null method {method!r}")
    if block_size is None:
        block_size = max(10, int(round(len(cand) ** 0.5)))
    obs_vals = local_entropies(
        Z, cand, y, k=k, exclude_temporal_radius=exclude_temporal_radius
    )
    observed = float(obs_vals.mean()) if len(obs_vals) else float("nan")
    rng = np.random.default_rng(seed)
    null: list[float] = []
    for _ in range(n_permutations):
        if method == "iid":
            shuffled = rng.permutation(y)
        elif method == "block":
            shuffled = block_permute(y, rng, block_size)
        else:
            shuffled = circular_shift(y, rng)
        vals = local_entropies(
            Z, cand, shuffled, k=k, exclude_temporal_radius=exclude_temporal_radius
        )
        if len(vals):
            null.append(float(vals.mean()))
    if not null:
        raise ValueError("null test produced no valid resamples")
    arr = np.asarray(null)
    p_excess = float((1 + np.sum(arr >= observed)) / (1 + len(arr)))
    p_deficit = float((1 + np.sum(arr <= observed)) / (1 + len(arr)))
    p_two = float(min(1.0, 2.0 * min(p_excess, p_deficit)))
    ci = moving_block_bootstrap_mean(
        obs_vals,
        block_size=max(2, min(block_size, len(obs_vals) // 2 or 2)),
        n_bootstrap=bootstrap,
        seed=seed,
    )
    # The interval is kept separate from the null test in the calling code; it
    # is returned here as attributes on the object's dict for reporting.
    result = NullTestResult(
        observed_mean_entropy=observed,
        null_mean=float(arr.mean()),
        null_sd=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        null_q025=float(np.quantile(arr, 0.025)),
        null_q975=float(np.quantile(arr, 0.975)),
        excess_entropy_bits=observed - float(arr.mean()),
        p_excess=p_excess,
        p_deficit=p_deficit,
        p_two_sided=p_two,
        method=method,
        block_size=int(block_size),
        n_candidates=int(len(cand)),
        n_permutations=int(n_permutations),
        n_valid_null=int(len(arr)),
    )
    out = result.to_dict()
    out.update({"mean_entropy_ci_low": ci["ci_low"], "mean_entropy_ci_high": ci["ci_high"]})
    return out


def compare_null_specs(
    Z: np.ndarray,
    candidate_frames: np.ndarray,
    outcomes: np.ndarray,
    *,
    k: int = 40,
    n_permutations: int = 500,
    block_sizes: tuple[int, ...] = (10, 30, 60),
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Run i.i.d., block, and circular-shift nulls as a sensitivity analysis."""
    specs: list[tuple[str, int | None]] = [("iid", None)]
    specs += [("block", b) for b in block_sizes]
    specs += [("shift", None)]
    return [
        null_branching_test(
            Z,
            candidate_frames,
            outcomes,
            k=k,
            n_permutations=n_permutations,
            method=m,
            block_size=b,
            seed=seed,
            bootstrap=500,
        )
        for m, b in specs
    ]
