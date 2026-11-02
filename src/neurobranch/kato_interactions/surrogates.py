"""Split-safe temporal surrogates for cross-neuron alignment controls.

Surrogates permute or circularly shift each neuron independently inside the
causal fitting window ending at the prediction landmark.  Every value used to
fit a surrogate is therefore observed no later than that landmark.  Individual
marginals and local autocorrelation are retained as closely as the finite window
allows, while cross-neuron temporal alignment is destroyed.
"""

from __future__ import annotations

import numpy as np

from neurobranch.kato_interactions.models import FrozenLinearModel, fit_var


def _block_permutation(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    block = max(2, min(int(block), n))
    edges = np.arange(0, n, block)
    blocks = [np.arange(lo, min(lo + block, n)) for lo in edges]
    order = rng.permutation(len(blocks))
    if len(blocks) > 1 and np.array_equal(order, np.arange(len(blocks))):
        order = np.roll(order, 1)
    return np.concatenate([blocks[i] for i in order])


def shuffle_window(
    values: np.ndarray,
    *,
    method: str = "circular",
    block: int = 25,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return a column-wise temporal surrogate of a causal window."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("window must be two-dimensional")
    n = arr.shape[0]
    rng = np.random.default_rng(int(seed))
    out = np.empty_like(arr)
    shifts: list[int] = []
    for j in range(arr.shape[1]):
        if method == "circular":
            lo = max(1, n // 10)
            hi = max(lo + 1, n - lo)
            shift = int(rng.integers(lo, hi))
            out[:, j] = np.roll(arr[:, j], shift)
            shifts.append(shift)
        elif method == "block":
            out[:, j] = arr[_block_permutation(n, block, rng), j]
        else:
            raise ValueError(f"unknown surrogate method {method!r}")
    return out, {
        "n": int(n),
        "min_shift_fraction": float(min(shifts) / n) if shifts else float("nan"),
        "max_shift_fraction": float(max(shifts) / n) if shifts else float("nan"),
    }


def fit_shuffled_window(
    X: np.ndarray,
    t: int,
    *,
    window: int = 500,
    method: str = "circular",
    block: int = 25,
    seed: int = 0,
    ridge: float = 1.0,
    spectral_cap: float | None = 0.995,
    min_transitions: int = 50,
) -> FrozenLinearModel | None:
    t = int(t)
    lo = max(1, t - int(window) + 1)
    if t - lo < int(min_transitions):
        return None
    values, diagnostics = shuffle_window(
        X[lo : t + 1],
        method=method,
        block=block,
        seed=seed,
    )
    starts = np.arange(0, len(values) - 1, dtype=np.int64)
    model = fit_var(values, starts, ridge=ridge, spectral_cap=spectral_cap)
    if model is None:
        return None
    model.fit_record.update(
        {
            "window": int(window),
            "landmark": t,
            "split": "causal_window_surrogate",
            "surrogate_method": method,
            "surrogate_block": int(block),
            "surrogate_seed": int(seed),
            **diagnostics,
        }
    )
    return model
