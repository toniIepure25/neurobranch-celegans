"""Train-only latent projections and real-vs-simulated trajectory validation.

Two representations are deliberately kept separate:

* PCA, which has an exact linear inverse and is therefore used for latent-space
  dynamics;
* BunDLe-Net, which is a representation/transition model without a published
  inverse decoder and is therefore used as an encoder and visualisation, not
  as proof that a named-neuron intervention maps to one latent coordinate.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from torch import nn

from neurobranch.dynamics.torch_utils import set_seed
from neurobranch.preprocessing import WormData, split_starts
from neurobranch.representation.bundlenet import BunDLeLite, causal_lowpass


@dataclass
class PCAEncoder:
    pca: PCA

    @classmethod
    def fit(cls, data: WormData, *, n_components: int = 3) -> PCAEncoder:
        X = data.X[data.masks["train"]]
        d = min(n_components, X.shape[1], len(X) - 1)
        return cls(PCA(n_components=d, random_state=0, svd_solver="randomized").fit(X))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(np.asarray(X, dtype=np.float64))

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        return self.pca.inverse_transform(np.asarray(Z, dtype=np.float64))

    @property
    def explained_variance(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_


def _pad_windows(X: np.ndarray, window: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n = len(X)
    if n < window:
        X = np.concatenate([np.repeat(X[:1], window - n, axis=0), X], axis=0)
        n = window
    padded = np.concatenate([np.repeat(X[:1], window - 1, axis=0), X], axis=0)
    return np.stack([padded[t : t + window] for t in range(n)]).astype(np.float32)


@dataclass
class BunDLeEncoder:
    """A train-only BunDLeLite encoder with its causal preprocessing."""

    net: BunDLeLite
    mu: np.ndarray
    sd: np.ndarray
    window: int
    latent_dim: int
    fps: float
    seed: int
    history: dict[str, list[float]]
    metadata: dict[str, Any]

    def transform(self, states: np.ndarray) -> np.ndarray:
        X = np.asarray(states, dtype=np.float64)
        Xf = causal_lowpass(X, self.fps)
        Xf = (Xf - self.mu) / np.maximum(self.sd, 1e-8)
        windows = _pad_windows(Xf, self.window)
        return self.net.encode(windows)

    @classmethod
    def fit(
        cls,
        data: WormData,
        *,
        window: int = 15,
        latent_dim: int = 3,
        epochs: int = 80,
        seed: int = 17,
        batch_size: int = 128,
    ) -> BunDLeEncoder:
        set_seed(seed, num_threads=4)
        rec = data.recording
        X = np.asarray(data.X, dtype=np.float64)
        Xf = causal_lowpass(X, rec.fps)
        mu = Xf[data.masks["train"]].mean(0)
        sd = np.maximum(Xf[data.masks["train"]].std(0), 1e-8)
        Xf = (Xf - mu) / sd
        fine_names = list(dict.fromkeys(rec.states))
        yfine = np.asarray([fine_names.index(s) for s in rec.states], dtype=np.int64)
        T = len(Xf)
        candidates = np.arange(window - 1, T - 1)
        train_end = np.asarray(
            [t for t in candidates if data.masks["train"][t - window + 1 : t + 2].all()]
        )
        val_end = np.asarray([t for t in candidates if data.masks["val"][t - window + 1 : t + 2].all()])
        if len(train_end) < 100 or len(val_end) < 10:
            raise ValueError("Insufficient BunDLe windows in train/validation")

        net = BunDLeLite(Xf.shape[1], window, latent_dim, len(fine_names))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        rng = np.random.default_rng(seed)
        best = (float("inf"), -1, copy.deepcopy(net.state_dict()))
        history = {"train_loss": [], "val_loss": []}

        def make_pair(endpoints: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
            X0 = np.stack([Xf[t - window + 1 : t + 1] for t in endpoints])
            X1 = np.stack([Xf[t - window + 2 : t + 2] for t in endpoints])
            batch = np.stack([X0, X1], axis=1).astype(np.float32)
            return torch.from_numpy(batch), torch.from_numpy(yfine[endpoints + 1])

        for _ in range(epochs):
            net.train()
            order = rng.permutation(len(train_end))
            losses = []
            for lo in range(0, len(order), batch_size):
                idx = order[lo : lo + batch_size]
                bx, by = make_pair(train_end[idx])
                up, low, logits = net(bx)
                loss = 0.9 * nn.functional.mse_loss(up, low) + 0.1 * nn.functional.cross_entropy(logits, by)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(float(loss))
            net.eval()
            with torch.no_grad():
                vx, vy = make_pair(val_end)
                up, low, logits = net(vx)
                val = float(
                    0.9 * nn.functional.mse_loss(up, low) + 0.1 * nn.functional.cross_entropy(logits, vy)
                )
            history["train_loss"].append(float(np.mean(losses)))
            history["val_loss"].append(val)
            if val < best[0]:
                best = (val, len(history["val_loss"]) - 1, copy.deepcopy(net.state_dict()))
        net.load_state_dict(best[2])
        net.eval()
        return cls(
            net=net,
            mu=mu,
            sd=sd,
            window=window,
            latent_dim=latent_dim,
            fps=float(rec.fps),
            seed=int(seed),
            history=history,
            metadata={
                "model": "bundlenet_lite_pytorch",
                "best_epoch": best[1],
                "best_val_loss": best[0],
                "train_windows": int(len(train_end)),
                "val_windows": int(len(val_end)),
                "train_only": True,
                "causal_lowpass": True,
                "deviations": [
                    "PyTorch reimplementation",
                    "causal one-pass low-pass rather than upstream filtfilt",
                    "blocked split rather than upstream random KFold",
                    "best validation epoch rather than fixed final epoch",
                ],
            },
        )

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "mu": self.mu,
                "sd": self.sd,
                "window": self.window,
                "latent_dim": self.latent_dim,
                "fps": self.fps,
                "seed": self.seed,
                "history": self.history,
                "metadata": self.metadata,
                "n_neurons": int(self.net.tau[1].in_features // self.window),
            },
            path,
        )


def path_geometry(Z: np.ndarray) -> dict[str, float]:
    Z = np.asarray(Z, dtype=np.float64)
    if len(Z) < 3:
        return {"path_length": float("nan"), "mean_step": float("nan"), "curvature": float("nan")}
    steps = np.diff(Z, axis=0)
    norms = np.linalg.norm(steps, axis=1)
    denom = np.maximum(np.linalg.norm(steps[:-1], axis=1) * np.linalg.norm(steps[1:], axis=1), 1e-12)
    cosines = np.clip((steps[:-1] * steps[1:]).sum(1) / denom, -1.0, 1.0)
    return {
        "path_length": float(norms.sum()),
        "mean_step": float(norms.mean()),
        "curvature": float(np.mean(np.arccos(cosines))),
    }


def latent_energy_distance(a: np.ndarray, b: np.ndarray, *, max_n: int = 300, seed: int = 0) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if len(a) > max_n:
        a = a[rng.choice(len(a), max_n, replace=False)]
    if len(b) > max_n:
        b = b[rng.choice(len(b), max_n, replace=False)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")

    def md(x: np.ndarray, y: np.ndarray) -> float:
        d = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2)
        return float(d.mean())

    return float(2 * md(a, b) - md(a, a) - md(b, b))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    num = np.linalg.norm(X.T @ Y, "fro") ** 2
    den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(num / den) if den > 1e-12 else float("nan")


def procrustes_error(X: np.ndarray, Y: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    R, _ = orthogonal_procrustes(X, Y)
    residual = np.linalg.norm(X @ R - Y) / max(1e-12, np.linalg.norm(Y))
    return float(residual)


def latent_comparison(
    real_states: np.ndarray,
    simulated_states: np.ndarray,
    encoder: PCAEncoder | BunDLeEncoder,
) -> dict[str, float]:
    Zr = encoder.transform(real_states)
    Zs = encoder.transform(simulated_states)
    gr = path_geometry(Zr)
    gs = path_geometry(Zs)
    return {
        "latent_energy_distance": latent_energy_distance(Zr, Zs),
        "real_path_length": gr["path_length"],
        "sim_path_length": gs["path_length"],
        "real_curvature": gr["curvature"],
        "sim_curvature": gs["curvature"],
        "path_length_ratio": gs["path_length"] / gr["path_length"] if gr["path_length"] > 1e-12 else float("nan"),
        "curvature_difference": gs["curvature"] - gr["curvature"],
    }


def real_test_trajectory(data: WormData, start: int, horizon: int) -> np.ndarray:
    return np.asarray(data.X[start : start + horizon + 1], dtype=np.float64)


def pca_latent_starts(data: WormData, horizon: int = 20, n: int = 20) -> np.ndarray:
    starts = split_starts(data, "test", history=1, horizon=1)
    keep = [int(t) for t in starts if data.masks["test"][int(t) : int(t) + horizon + 1].all()]
    return np.asarray(keep[:n], dtype=np.int64)
