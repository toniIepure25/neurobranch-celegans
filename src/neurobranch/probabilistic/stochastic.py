"""Small stochastic extensions of the adaptive conditional-mean models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurobranch.probabilistic.residual import center_residuals, regularized_covariance


@dataclass
class Innovation:
    kind: str
    pool: np.ndarray | None = None
    covariance: np.ndarray | None = None
    cholesky: np.ndarray | None = None
    scale: np.ndarray | None = None
    block: int = 1
    metadata: dict | None = None

    def sample(self, n_members: int, horizon: int, rng: np.random.Generator) -> np.ndarray:
        n_members = int(n_members)
        horizon = int(horizon)
        if self.kind == "gauss":
            z = rng.standard_normal((n_members * horizon, self.cholesky.shape[0]))
            flat = z @ self.cholesky.T
        elif self.kind == "gauss_diag":
            flat = rng.standard_normal((n_members * horizon, self.scale.size)) * self.scale
        elif self.kind in {"bootstrap", "block"}:
            pool = self.pool
            if pool is None or len(pool) == 0:
                raise ValueError("empty residual pool")
            if self.kind == "bootstrap":
                idx = rng.integers(0, len(pool), size=n_members * horizon)
                flat = pool[idx]
            else:
                block = max(1, min(int(self.block), len(pool)))
                n_blocks = int(np.ceil(horizon / block))
                pieces = []
                for _ in range(n_members):
                    starts = rng.integers(0, len(pool) - block + 1, size=n_blocks)
                    sample = np.concatenate([pool[s : s + block] for s in starts], axis=0)
                    pieces.append(sample[:horizon])
                flat = np.concatenate(pieces, axis=0)
        else:
            raise ValueError(f"unknown innovation kind {self.kind}")
        return flat.reshape(n_members, horizon, -1)


def build_innovation(
    residuals: np.ndarray,
    kind: str,
    *,
    block: int = 5,
    covariance_method: str = "ledoit",
) -> Innovation:
    centered, mean = center_residuals(residuals)
    meta = {"n_residuals": int(len(centered)), "innovation_mean_norm": float(np.linalg.norm(mean))}
    if kind == "gauss":
        cov, cov_meta = regularized_covariance(centered, method=covariance_method)
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            eigval, eigvec = np.linalg.eigh(cov)
            chol = eigvec @ np.diag(np.sqrt(np.clip(eigval, 1e-10, None)))
        meta.update(cov_meta)
        return Innovation("gauss", covariance=cov, cholesky=chol, metadata=meta)
    if kind == "gauss_diag":
        scale = np.sqrt(np.maximum(centered.var(axis=0, ddof=1), 1e-10))
        meta.update({"method": "diagonal", "scale_median": float(np.median(scale))})
        return Innovation("gauss_diag", scale=scale, metadata=meta)
    if kind == "bootstrap":
        return Innovation("bootstrap", pool=centered, metadata={**meta, "method": "iid_bootstrap"})
    if kind == "block":
        return Innovation(
            "block", pool=centered, block=int(block), metadata={**meta, "method": "moving_block", "block": block}
        )
    raise ValueError(f"unknown innovation kind {kind}")


def ensemble_rollout(
    mean_kind: str,
    fitted,
    x0: np.ndarray,
    horizon: int,
    n_members: int,
    innovation: Innovation,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate independent stochastic trajectories; shape (members, horizon, dim)."""
    horizon = int(horizon)
    n_members = int(n_members)
    x0 = np.asarray(x0, dtype=np.float64)
    eps = innovation.sample(n_members, horizon, rng)
    out = np.empty((n_members, horizon, x0.size), dtype=np.float64)
    if mean_kind == "var":
        x = np.repeat(x0[None, :], n_members, axis=0)
        for step in range(horizon):
            x = x @ fitted.A.T + fitted.b + eps[:, step]
            out[:, step] = x
    elif mean_kind == "persistence":
        x = np.repeat(x0[None, :], n_members, axis=0)
        for step in range(horizon):
            x = x + eps[:, step]
            out[:, step] = x
    elif mean_kind == "local_mean":
        mu = np.asarray(fitted.value, dtype=np.float64)
        for step in range(horizon):
            out[:, step] = mu + eps[:, step]
    elif mean_kind == "independent_ar":
        x = np.repeat(x0[None, :], n_members, axis=0)
        for step in range(horizon):
            x = fitted.a * x + fitted.b + eps[:, step]
            out[:, step] = x
    else:
        raise ValueError(f"unknown mean kind {mean_kind}")
    return out


def require_finite(ensemble: np.ndarray) -> float:
    arr = np.asarray(ensemble)
    return float(np.isfinite(arr).all())
