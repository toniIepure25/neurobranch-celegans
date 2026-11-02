"""Eigenmode, timescale and non-normal transient diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from neurobranch.adaptive.var import spectral_radius


def transient_amplification(A: np.ndarray, horizon: int = 100) -> tuple[float, int]:
    """Maximum spectral norm of ``A**h`` for h=1..horizon."""
    arr = np.asarray(A, dtype=np.float64)
    power = np.eye(arr.shape[0], dtype=np.float64)
    peak, peak_step = 0.0, 0
    for step in range(1, int(horizon) + 1):
        power = power @ arr
        value = float(np.linalg.norm(power, 2))
        if value > peak:
            peak, peak_step = value, step
    return peak, peak_step


def nonnormality(A: np.ndarray) -> float:
    """Henrici departure from normality, scale-free."""
    arr = np.asarray(A, dtype=np.float64)
    comm = arr @ arr.T - arr.T @ arr
    denom = float(np.linalg.norm(arr, "fro"))
    return float(np.linalg.norm(comm, "fro") / denom) if denom > 0 else float("nan")


def eigenvector_condition(A: np.ndarray) -> float:
    vals, vecs = np.linalg.eig(np.asarray(A, dtype=np.float64))
    del vals
    if vecs.shape[0] != vecs.shape[1]:
        return float("nan")
    try:
        return float(np.linalg.cond(vecs))
    except np.linalg.LinAlgError:
        return float("nan")


def operator_summary(A: np.ndarray, *, fps: float = 2.9, horizon: int = 100) -> dict[str, float]:
    arr = np.asarray(A, dtype=np.float64)
    vals = np.linalg.eigvals(arr)
    mags = np.abs(vals)
    sv = np.linalg.svd(arr, compute_uv=False)
    peak, peak_step = transient_amplification(arr, horizon)
    complex_modes = vals[np.imag(vals) > 1e-8]
    return {
        "spectral_radius": spectral_radius(arr),
        "max_eigenvalue_magnitude": float(mags.max()) if len(mags) else float("nan"),
        "min_eigenvalue_magnitude": float(mags.min()) if len(mags) else float("nan"),
        "max_singular_value": float(sv.max()) if len(sv) else float("nan"),
        "min_singular_value": float(sv.min()) if len(sv) else float("nan"),
        "condition_number": float(sv.max() / max(sv.min(), 1e-12)) if len(sv) else float("nan"),
        "nonnormality": nonnormality(arr),
        "eigenvector_condition": eigenvector_condition(arr),
        "transient_peak": peak,
        "transient_peak_step": float(peak_step),
        "complex_pair_count": float(len(complex_modes)),
        "oscillatory_fraction": float(len(complex_modes) / max(len(vals), 1)),
        "leading_period_frames": float(
            2.0 * np.pi / abs(np.angle(complex_modes[np.argmax(np.abs(complex_modes))]))
        )
        if len(complex_modes)
        else float("nan"),
        "leading_period_seconds": float(
            (2.0 * np.pi / abs(np.angle(complex_modes[np.argmax(np.abs(complex_modes))]))) / fps
        )
        if len(complex_modes)
        else float("nan"),
    }


def mode_table(A: np.ndarray, *, fps: float = 2.9) -> pd.DataFrame:
    vals = np.linalg.eigvals(np.asarray(A, dtype=np.float64))
    rows = []
    for j, value in enumerate(vals):
        mag = float(abs(value))
        angle = float(np.angle(value))
        rows.append(
            {
                "mode": j,
                "real": float(np.real(value)),
                "imag": float(np.imag(value)),
                "magnitude": mag,
                "angle_rad": angle,
                "frequency_hz": float(abs(angle) / (2.0 * np.pi) * fps),
                "period_frames": float(2.0 * np.pi / abs(angle)) if abs(angle) > 1e-12 else float("inf"),
                "decay_frames": float(-1.0 / np.log(mag)) if 0 < mag < 1 else float("inf"),
                "decay_seconds": float((-1.0 / np.log(mag)) / fps) if 0 < mag < 1 else float("inf"),
                "is_complex": bool(abs(np.imag(value)) > 1e-8),
            }
        )
    return pd.DataFrame(rows)


def leading_decay_frames(A: np.ndarray) -> float:
    mags = np.abs(np.linalg.eigvals(np.asarray(A, dtype=np.float64)))
    if not len(mags):
        return float("nan")
    rho = float(mags.max())
    return float(-1.0 / np.log(rho)) if 0 < rho < 1 else float("inf")


def complex_pair_stability(table: pd.DataFrame) -> dict[str, float]:
    """Summarize positive-imaginary modes and their period/decay spread."""
    pair = table[table["is_complex"] & (table["imag"] > 0)].copy()
    if pair.empty:
        return {
            "n_positive_imaginary": 0,
            "median_period_frames": float("nan"),
            "median_decay_frames": float("nan"),
            "max_magnitude": float("nan"),
        }
    return {
        "n_positive_imaginary": int(len(pair)),
        "median_period_frames": float(pair["period_frames"].median()),
        "median_decay_frames": float(pair["decay_frames"].median()),
        "max_magnitude": float(pair["magnitude"].max()),
    }
