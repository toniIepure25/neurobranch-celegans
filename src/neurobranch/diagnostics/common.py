"""Shared utilities for the generative-failure diagnostics branch.

Everything here is deliberately small and explicit.  These utilities do not
change the simulator's models or the frozen Phase 6-15 reports.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from neurobranch.config import PROJECT_ROOT, ExperimentConfig
from neurobranch.data.kato import WormRecording, load_kato
from neurobranch.preprocessing import WormData, make_worm_data
from neurobranch.simulator.metrics import _safe_r2


def load_nostim(
    config_path: str | None = None,
) -> tuple[ExperimentConfig, list[WormData], list[WormRecording]]:
    cfg = ExperimentConfig.load(PROJECT_ROOT / (config_path or "configs/experiments/forward_stress.yaml"))
    dataset = load_kato(cfg.data.path)
    data = [make_worm_data(rec, cfg.split) for rec in dataset]
    return cfg, data, list(dataset)


def skill_vs_persistence(mse: float, persistence_mse: float) -> float:
    if not np.isfinite(mse) or not np.isfinite(persistence_mse) or persistence_mse <= 0:
        return float("nan")
    return float(1.0 - mse / persistence_mse)


def block_bootstrap_skill(
    mse: np.ndarray,
    persistence_mse: np.ndarray,
    *,
    block: int = 20,
    n_boot: int = 1000,
    seed: int = 17,
) -> dict[str, float]:
    """Moving-block bootstrap over dependent held-out start indices."""
    mse = np.asarray(mse, dtype=np.float64)
    persistence_mse = np.asarray(persistence_mse, dtype=np.float64)
    n = len(mse)
    if n < 5 or n != len(persistence_mse):
        return {"lo": float("nan"), "hi": float("nan"), "n": int(n)}
    rng = np.random.default_rng(seed)
    block = max(1, min(int(block), n))
    n_blocks = int(np.ceil(n / block))
    values = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        denom = float(persistence_mse[idx].mean())
        values[b] = 1.0 - float(mse[idx].mean()) / denom if denom > 0 else np.nan
    return {
        "lo": float(np.nanpercentile(values, 2.5)),
        "hi": float(np.nanpercentile(values, 97.5)),
        "n": int(n),
        "block": int(block),
    }


def per_neuron_skill(
    pred: np.ndarray,
    truth: np.ndarray,
    persistence: np.ndarray,
) -> np.ndarray:
    """Skill for each neuron, pooling starts and horizons."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    persistence = np.asarray(persistence, dtype=np.float64)
    mse = np.nanmean((pred - truth) ** 2, axis=(0, 1))
    base = np.nanmean((persistence - truth) ** 2, axis=(0, 1))
    out = np.full(mse.shape, np.nan, dtype=np.float64)
    ok = base > 1e-12
    out[ok] = 1.0 - mse[ok] / base[ok]
    return out


def per_neuron_r2(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    return np.array([_safe_r2(truth[:, :, j], pred[:, :, j]) for j in range(truth.shape[2])])


def forecast_matrix(
    starts: Iterable[int],
    X: np.ndarray,
    *,
    history: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return persistence history and true future for identical starts."""
    starts = np.asarray(list(starts), dtype=np.int64)
    hist = np.stack([X[t - history + 1 : t + 1] for t in starts])
    truth = np.stack([X[t + 1 : t + horizon + 1] for t in starts])
    return hist, truth


def make_forecast_starts(data: WormData, horizon: int, history: int = 5) -> np.ndarray:
    from neurobranch.preprocessing import split_starts

    return split_starts(data, "test", history=history, horizon=horizon)


def dominant_frequency_and_power(X: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    """Per-neuron dominant frequency and low-frequency power fraction."""
    x = np.asarray(X, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    n = x.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    power = np.abs(np.fft.rfft(x, axis=0)) ** 2
    total = power.sum(axis=0)
    total[total <= 1e-12] = np.nan
    low = power[(freqs > 0) & (freqs < 0.05)].sum(axis=0) / total
    dom = freqs[np.argmax(power[1:], axis=0) + 1]
    return {"dominant_hz": dom, "low_fraction": low}


@dataclass
class SynthSystem:
    """Known linear Gaussian system with optional calcium filtering."""

    A: np.ndarray
    D: np.ndarray
    mean: np.ndarray
    obs_cov: np.ndarray
    process_cov: np.ndarray
    alpha: float
    seed: int

    @property
    def n_neurons(self) -> int:
        return int(self.D.shape[0])

    @property
    def latent_dim(self) -> int:
        return int(self.A.shape[0])

    def simulate(self, n_frames: int, *, burn_in: int = 200) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        k = self.latent_dim
        z = rng.normal(size=k)
        c = rng.normal(size=k)
        zs, ys = [], []
        for t in range(n_frames + burn_in):
            z = self.A @ z + rng.multivariate_normal(np.zeros(k), self.process_cov)
            c = z.copy() if self.alpha >= 1.0 else (1.0 - self.alpha) * c + self.alpha * z
            y = self.mean + self.D @ c + rng.multivariate_normal(
                np.zeros(self.n_neurons), self.obs_cov
            )
            if t >= burn_in:
                zs.append(z.copy())
                ys.append(y)
        return np.asarray(ys, dtype=np.float64), np.asarray(zs, dtype=np.float64)


def random_stable_system(
    *,
    latent_dim: int = 6,
    n_neurons: int = 40,
    alpha: float = 1.0,
    radius: float = 0.95,
    obs_sd: float = 0.2,
    process_sd: float = 0.2,
    seed: int = 17,
    oscillatory: bool = False,
) -> SynthSystem:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(latent_dim, latent_dim)))
    if oscillatory:
        if latent_dim % 2:
            raise ValueError("oscillatory systems need an even latent_dim")
        blocks = []
        for i in range(latent_dim // 2):
            theta = 0.2 + 1.0 * i / max(1, latent_dim // 2)
            c, s = np.cos(theta), np.sin(theta)
            blocks.append(radius * np.array([[c, -s], [s, c]]))
        A = Q @ np.block(
            [
                [blocks[i] if i == j else np.zeros((2, 2)) for j in range(len(blocks))]
                for i in range(len(blocks))
            ]
        ) @ Q.T
    else:
        eig = np.linspace(0.3, radius, latent_dim)
        A = Q @ np.diag(eig) @ Q.T
    D = rng.normal(scale=1.0 / np.sqrt(latent_dim), size=(n_neurons, latent_dim))
    mean = rng.normal(scale=0.2, size=n_neurons)
    return SynthSystem(
        A=A,
        D=D,
        mean=mean,
        obs_cov=np.eye(n_neurons) * obs_sd**2,
        process_cov=np.eye(latent_dim) * process_sd**2,
        alpha=float(alpha),
        seed=int(seed),
    )


def git_state(repo: str | None = None) -> dict[str, Any]:
    from neurobranch.config import git_state as _git_state

    return _git_state()
