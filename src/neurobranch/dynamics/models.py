"""Neuronal- and latent-state dynamics models.

All models are fitted on standardized traces.  Behaviour logits are auxiliary
outputs and never used as dynamics inputs at prediction time.  Every model
supports open-loop rollout from an observed initial state, which is required for
both branching analysis and virtual-perturbation experiments.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from torch import nn

from neurobranch.config import ModelConfig, TrainConfig
from neurobranch.dynamics.torch_utils import (
    GRUNet,
    MLPNet,
    TrainHistory,
    choose_device,
    set_seed,
)

Array = np.ndarray
PerturbFn = Callable[[Array, int], Array]


def _softmax(logits: Array) -> Array:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class BehaviorProbe:
    """Multinomial logistic behaviour decoder with stable serialization."""

    def __init__(self, *, seed: int = 17, max_iter: int = 600):
        self.seed = seed
        self.max_iter = max_iter
        self.clf: LogisticRegression | None = None
        self.classes_: Array | None = None

    def fit(self, Z: Array, y: Array) -> BehaviorProbe:
        self.clf = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.seed,
            class_weight="balanced",
        )
        self.clf.fit(Z, y)
        self.classes_ = self.clf.classes_.astype(np.int64)
        return self

    def predict_proba(self, Z: Array) -> Array:
        if self.clf is None or self.classes_ is None:
            raise RuntimeError("BehaviorProbe is not fitted")
        raw = self.clf.predict_proba(Z)
        # Expand to canonical 4-class order if a class is absent in training.
        out = np.zeros((Z.shape[0], 4), dtype=np.float64)
        for j, c in enumerate(self.classes_):
            out[:, int(c)] = raw[:, j]
        return out

    def predict(self, Z: Array) -> Array:
        return self.predict_proba(Z).argmax(axis=1)

    def state_dict(self) -> dict:
        if self.clf is None:
            raise RuntimeError("BehaviorProbe is not fitted")
        return {
            "coef": self.clf.coef_.copy(),
            "intercept": self.clf.intercept_.copy(),
            "classes": self.classes_.copy(),
        }

    @classmethod
    def from_state_dict(cls, payload: dict) -> BehaviorProbe:
        obj = cls()
        clf = LogisticRegression()
        clf.coef_ = np.asarray(payload["coef"])
        clf.intercept_ = np.asarray(payload["intercept"])
        clf.classes_ = np.asarray(payload["classes"])
        clf.n_features_in_ = clf.coef_.shape[1]
        obj.clf = clf
        obj.classes_ = clf.classes_.astype(np.int64)
        return obj


class DynamicsModel(Protocol):
    name: str
    n_neurons: int
    latent_dim: int | None

    def predict_next(self, history: Array) -> Array: ...
    def behavior_proba(self, history: Array) -> Array: ...
    def rollout(self, x0: Array, steps: int, *, seed: int = 0, noise_std: float = 0.0,
                perturb: PerturbFn | None = None,
                initial_history: Array | None = None) -> Array: ...
    def latent(self, history: Array) -> Array | None: ...


def _history_from_rollout(seq: Array, history_len: int) -> Array:
    """Return last ``history_len`` rows, padding at the start if necessary."""
    if seq.shape[0] >= history_len:
        return seq[-history_len:]
    pad = np.repeat(seq[:1], history_len - seq.shape[0], axis=0)
    return np.concatenate([pad, seq], axis=0)


def _initial_history(
    x0: Array,
    initial_history: Array | None,
    n_neurons: int,
) -> Array:
    """Return a validated ``(history, n_neurons)`` initial window.

    ``x0`` is retained for backwards compatibility.  Models with more than one
    input frame must receive ``initial_history`` during evaluation; padding a
    single frame hides the fact that the real past was discarded.
    """
    if initial_history is not None:
        hist = np.asarray(initial_history, dtype=np.float64)
    else:
        arr = np.asarray(x0, dtype=np.float64)
        hist = arr[None] if arr.ndim == 1 else arr
    if hist.ndim != 2 or hist.shape[1] != n_neurons:
        raise ValueError(f"initial history must be (history, {n_neurons}), got {hist.shape}")
    if not np.isfinite(hist).all():
        raise ValueError("initial history contains non-finite values")
    return hist


def _last_states(history: Array, n_neurons: int) -> Array:
    """Normalize a model input to a (B, N) batch of current states.

    Accepts (B, H, N) history batches or (T, N) rollouts.  The latter is used
    for decoding behaviour along a simulated trajectory.
    """
    arr = np.asarray(history)
    if arr.ndim == 3:
        return arr[:, -1]
    if arr.ndim == 2 and arr.shape[1] == n_neurons:
        return arr
    if arr.ndim == 1 and arr.shape[0] == n_neurons:
        return arr[None]
    raise ValueError(f"Expected (B,H,N) or (T,N) neural input, got {arr.shape}")


def _sequence_windows(history: Array, n_neurons: int, history_len: int) -> Array:
    """Build per-frame histories from a (T, N) sequence, left-padding."""
    arr = np.asarray(history)
    if arr.ndim == 3:
        return arr[:, -history_len:]
    if arr.ndim == 1:
        arr = arr[None]
    if history_len <= 1:
        return arr[:, None, :]
    pad = np.repeat(arr[:1], history_len - 1, axis=0)
    padded = np.concatenate([pad, arr], axis=0)
    return np.stack([padded[i : i + history_len] for i in range(arr.shape[0])])


@dataclass
class PersistenceModel:
    n_neurons: int
    probe: BehaviorProbe
    name: str = "persistence"
    latent_dim: int | None = None

    def predict_next(self, history: Array) -> Array:
        return np.asarray(history)[:, -1].copy()

    def behavior_proba(self, history: Array) -> Array:
        return self.probe.predict_proba(_last_states(history, self.n_neurons))

    def latent(self, history: Array) -> None:
        return None

    def rollout(
        self,
        x0: Array,
        steps: int,
        *,
        seed: int = 0,
        noise_std: float = 0.0,
        perturb: PerturbFn | None = None,
        initial_history: Array | None = None,
        stochastic: bool = False,
    ) -> Array:
        rng = np.random.default_rng(seed)
        hist = _initial_history(x0, initial_history, self.n_neurons)
        seq = [hist[-1].copy()]
        for s in range(steps):
            x = seq[-1].copy()
            if noise_std > 0:
                x = x + rng.normal(scale=noise_std, size=x.shape)
            if perturb is not None:
                x = perturb(x, s)
            seq.append(x)
        return np.stack(seq)


@dataclass
class VARModel:
    n_neurons: int
    A: Array  # (N, N*history)
    b: Array
    history_len: int
    probe: BehaviorProbe
    name: str = "var1"
    latent_dim: int | None = None
    ridge: float = 1e-2
    spectral_cap: float | None = None
    noise_cov: Array | None = None
    residuals: Array | None = None
    noise_model: str = "none"
    bootstrap_block: int = 5
    native_stochastic: bool = False

    @classmethod
    def fit(
        cls,
        X: Array,
        y: Array,
        starts: Array,
        *,
        history: int = 1,
        ridge: float = 1e-2,
        seed: int = 17,
        spectral_cap: float | None = None,
        noise_model: str = "none",
        bootstrap_block: int = 5,
    ) -> VARModel:
        H = np.stack([X[t - history + 1 : t + 1].ravel() for t in starts])
        Y = X[starts + 1]
        # Do not regularize the intercept.
        mu = H.mean(axis=0)
        sd = H.std(axis=0)
        sd[sd < 1e-8] = 1.0
        Hs = (H - mu) / sd
        A_aug = np.linalg.solve(Hs.T @ Hs + ridge * np.eye(Hs.shape[1]), Hs.T @ Y)
        b = Y.mean(axis=0) - (mu / sd) @ A_aug
        A = (A_aug / sd[:, None]).T
        if spectral_cap is not None:
            if spectral_cap <= 0:
                raise ValueError("spectral_cap must be positive")
            companion = np.zeros((X.shape[1] * history, X.shape[1] * history))
            companion[: X.shape[1], :] = A
            if history > 1:
                companion[X.shape[1] :, : -X.shape[1]] = np.eye(X.shape[1] * (history - 1))
            rho = float(np.max(np.abs(np.linalg.eigvals(companion))))
            if rho > spectral_cap:
                A = A * (spectral_cap / rho)
        model = cls(
            n_neurons=X.shape[1],
            A=A,
            b=b,
            history_len=history,
            probe=BehaviorProbe(seed=seed),
            ridge=ridge,
            spectral_cap=spectral_cap,
            noise_model=noise_model,
            bootstrap_block=bootstrap_block,
            native_stochastic=noise_model != "none",
        )
        model.probe.fit(X[starts], y[starts])
        if noise_model != "none":
            residuals = Y - model.predict_next(H)
            model.residuals = np.asarray(residuals, dtype=np.float64)
            cov = LedoitWolf().fit(model.residuals).covariance_
            model.noise_cov = np.asarray(cov, dtype=np.float64)
        return model

    @property
    def spectral_radius(self) -> float:
        size = self.n_neurons * self.history_len
        companion = np.zeros((size, size))
        companion[: self.n_neurons, :] = self.A
        if self.history_len > 1:
            companion[self.n_neurons :, : -self.n_neurons] = np.eye(
                self.n_neurons * (self.history_len - 1)
            )
        return float(np.max(np.abs(np.linalg.eigvals(companion))))

    def _innovation(self, rng: np.random.Generator, buffer: list[Array]) -> Array:
        if self.noise_model == "gaussian" and self.noise_cov is not None:
            return rng.multivariate_normal(np.zeros(self.n_neurons), self.noise_cov)
        if self.noise_model.startswith("bootstrap") and self.residuals is not None:
            if not buffer:
                block = self.bootstrap_block if self.noise_model.endswith("block") else 1
                start = int(rng.integers(0, max(1, len(self.residuals) - block + 1)))
                buffer.extend(self.residuals[start : start + block].copy())
            if not buffer:
                return np.zeros(self.n_neurons)
            return buffer.pop(0)
        return np.zeros(self.n_neurons)

    def predict_next(self, history: Array) -> Array:
        h = np.asarray(history)
        if h.ndim == 2:
            h = h[:, None, :]
        h = h[:, -self.history_len :]
        return h.reshape(h.shape[0], -1) @ self.A.T + self.b

    def behavior_proba(self, history: Array) -> Array:
        return self.probe.predict_proba(_last_states(history, self.n_neurons))

    def latent(self, history: Array) -> None:
        return None

    def rollout(
        self,
        x0: Array,
        steps: int,
        *,
        seed: int = 0,
        noise_std: float = 0.0,
        perturb: PerturbFn | None = None,
        initial_history: Array | None = None,
        stochastic: bool = False,
    ) -> Array:
        rng = np.random.default_rng(seed)
        hist = _initial_history(x0, initial_history, self.n_neurons)
        seq = [hist[-1].copy()]
        buffer = [row.copy() for row in hist]
        residual_buffer: list[Array] = []
        for s in range(steps):
            window = _history_from_rollout(np.stack(buffer), self.history_len)[None]
            x = self.predict_next(window)[0]
            if stochastic:
                x = x + self._innovation(rng, residual_buffer)
            if noise_std > 0:
                x = x + rng.normal(scale=noise_std, size=x.shape)
            if perturb is not None:
                x = perturb(x, s)
            seq.append(x)
            buffer.append(x)
        return np.stack(seq)


@dataclass
class MLPDynamics:
    net: MLPNet
    n_neurons: int
    history_len: int
    probe: BehaviorProbe
    device: str = "cpu"
    name: str = "mlp"
    latent_dim: int | None = None
    residual_std: Array | None = None

    def _tensor(self, history: Array) -> torch.Tensor:
        h = np.asarray(history, dtype=np.float32)[:, -self.history_len :]
        return torch.from_numpy(h).to(self.device)

    def predict_next(self, history: Array) -> Array:
        self.net.eval()
        with torch.no_grad():
            out, _ = self.net(self._tensor(history))
        return out.cpu().numpy()

    def behavior_proba(self, history: Array) -> Array:
        self.net.eval()
        with torch.no_grad():
            windows = _sequence_windows(history, self.n_neurons, self.history_len)
            _, logits = self.net(torch.from_numpy(windows.astype(np.float32)).to(self.device))
        return _softmax(logits.cpu().numpy())

    def latent(self, history: Array) -> None:
        return None

    def rollout(
        self,
        x0: Array,
        steps: int,
        *,
        seed: int = 0,
        noise_std: float = 0.0,
        perturb: PerturbFn | None = None,
        initial_history: Array | None = None,
        stochastic: bool = False,
    ) -> Array:
        rng = np.random.default_rng(seed)
        hist = _initial_history(x0, initial_history, self.n_neurons)
        seq = [hist[-1].copy()]
        buffer = [row.copy() for row in hist]
        self.net.eval()
        with torch.no_grad():
            for s in range(steps):
                window = _history_from_rollout(np.stack(buffer), self.history_len)[None]
                x = self.predict_next(window)[0]
                if noise_std > 0:
                    x = x + rng.normal(scale=noise_std, size=x.shape)
                if perturb is not None:
                    x = perturb(x, s)
                seq.append(x)
                buffer.append(x)
        return np.stack(seq)


def train_torch_model(
    net: nn.Module,
    X: Array,
    y: Array,
    train_starts: Array,
    val_starts: Array,
    *,
    history: int,
    n_classes: int,
    train_cfg: TrainConfig,
    residual: bool = False,
    verbose: bool = False,
) -> TrainHistory:
    device = choose_device(train_cfg.device)
    net = net.to(device)
    opt = torch.optim.AdamW(
        net.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay
    )
    rng = np.random.default_rng(train_cfg.seed)

    def batch(starts: Array, idx: Array) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sel = starts[idx]
        H = np.stack([X[t - history + 1 : t + 1] for t in sel]).astype(np.float32)
        Y = X[sel + 1].astype(np.float32)
        Yb = y[sel + 1].astype(np.int64)
        return (
            torch.from_numpy(H).to(device),
            torch.from_numpy(Y).to(device),
            torch.from_numpy(Yb).to(device),
        )

    def evaluate(starts: Array) -> tuple[float, float]:
        if len(starts) == 0:
            return float("nan"), float("nan")
        net.eval()
        losses, correct = [], []
        with torch.no_grad():
            for lo in range(0, len(starts), 512):
                sel = np.arange(lo, min(len(starts), lo + 512))
                H, Y, Yb = batch(starts, sel)
                pred, logits = net(H)
                mse = torch.mean((pred - Y) ** 2)
                ce = nn.functional.cross_entropy(logits, Yb)
                losses.append(float((mse + 0.2 * ce).item()))
                correct.append((logits.argmax(-1) == Yb).float())
        return float(np.mean(losses)), float(torch.cat(correct).mean().item())

    best = (float("inf"), -1, copy.deepcopy(net.state_dict()))
    hist = TrainHistory([], [], [], -1, float("inf"))
    for epoch in range(train_cfg.epochs):
        net.train()
        order = rng.permutation(len(train_starts))
        ep_losses = []
        for lo in range(0, len(order), train_cfg.batch_size):
            idx = order[lo : lo + train_cfg.batch_size]
            H, Y, Yb = batch(train_starts, idx)
            pred, logits = net(H)
            mse = torch.mean((pred - Y) ** 2)
            ce = nn.functional.cross_entropy(logits, Yb)
            loss = mse + 0.2 * ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if train_cfg.grad_clip:
                nn.utils.clip_grad_norm_(net.parameters(), train_cfg.grad_clip)
            opt.step()
            ep_losses.append(float(loss.item()))
        val_loss, val_acc = evaluate(val_starts)
        hist.train_loss.append(float(np.mean(ep_losses)))
        hist.val_loss.append(val_loss)
        hist.val_behavior_accuracy.append(val_acc)
        if np.isfinite(val_loss) and val_loss < best[0]:
            best = (val_loss, epoch, copy.deepcopy(net.state_dict()))
        if verbose and epoch % 10 == 0:
            print(f"epoch {epoch:3d} train {np.mean(ep_losses):.4f} val {val_loss:.4f} acc {val_acc:.3f}")
        if epoch - best[1] >= train_cfg.patience:
            break
    net.load_state_dict(best[2])
    hist.best_epoch = best[1]
    hist.best_val_loss = best[0]
    return hist


def fit_mlp(
    X: Array,
    y: Array,
    train_starts: Array,
    val_starts: Array,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    *,
    residual: bool = False,
) -> tuple[MLPDynamics, TrainHistory]:
    set_seed(train_cfg.seed, num_threads=train_cfg.num_threads)
    history = model_cfg.history
    net = MLPNet(
        X.shape[1] * history,
        model_cfg.hidden_dim,
        X.shape[1],
        model_cfg.n_behaviors,
        dropout=model_cfg.dropout,
        residual=residual,
    )
    hist = train_torch_model(
        net,
        X,
        y,
        train_starts,
        val_starts,
        history=history,
        n_classes=model_cfg.n_behaviors,
        train_cfg=train_cfg,
        residual=residual,
    )
    probe = BehaviorProbe(seed=train_cfg.seed).fit(X[train_starts], y[train_starts])
    return (
        MLPDynamics(
            net=net,
            n_neurons=X.shape[1],
            history_len=history,
            probe=probe,
            device=str(choose_device(train_cfg.device)),
            name="mlp_residual" if residual else "mlp",
        ),
        hist,
    )


def fit_gru(
    X: Array,
    y: Array,
    train_starts: Array,
    val_starts: Array,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> tuple[MLPDynamics, TrainHistory]:
    set_seed(train_cfg.seed, num_threads=train_cfg.num_threads)
    history = max(2, model_cfg.history)
    net = GRUNet(X.shape[1], model_cfg.hidden_dim, model_cfg.n_behaviors, dropout=model_cfg.dropout)
    hist = train_torch_model(
        net,
        X,
        y,
        train_starts,
        val_starts,
        history=history,
        n_classes=model_cfg.n_behaviors,
        train_cfg=train_cfg,
    )
    probe = BehaviorProbe(seed=train_cfg.seed).fit(X[train_starts], y[train_starts])
    return (
        MLPDynamics(
            net=net,  # type: ignore[arg-type]
            n_neurons=X.shape[1],
            history_len=history,
            probe=probe,
            device=str(choose_device(train_cfg.device)),
            name="gru",
        ),
        hist,
    )


@dataclass
class PCALatent:
    pca: PCA
    z_mean: Array
    z_std: Array

    @classmethod
    def fit(cls, X: Array, latent_dim: int) -> PCALatent:
        d = min(int(latent_dim), X.shape[1], X.shape[0] - 1)
        pca = PCA(n_components=d, random_state=0, svd_solver="randomized")
        Z = pca.fit_transform(X)
        z_mean = Z.mean(axis=0)
        z_std = Z.std(axis=0)
        z_std[z_std < 1e-8] = 1.0
        return cls(pca=pca, z_mean=z_mean, z_std=z_std)

    def encode(self, X: Array) -> Array:
        return (self.pca.transform(X) - self.z_mean) / self.z_std

    def decode(self, Z: Array) -> Array:
        raw = Z * self.z_std + self.z_mean
        return self.pca.inverse_transform(raw)


class _LatentNet(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, n_classes: int, dropout: float):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.out = nn.Linear(hidden_dim, latent_dim)
        self.behavior = nn.Linear(hidden_dim, n_classes)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(z)
        return z + self.out(h), self.behavior(h)


def _train_latent_net(
    net: nn.Module,
    Z: Array,
    y: Array,
    train_starts: Array,
    val_starts: Array,
    train_cfg: TrainConfig,
) -> TrainHistory:
    device = choose_device(train_cfg.device)
    net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    rng = np.random.default_rng(train_cfg.seed)

    def batch(starts: Array, idx: Array):
        sel = starts[idx]
        return (
            torch.from_numpy(Z[sel].astype(np.float32)).to(device),
            torch.from_numpy(Z[sel + 1].astype(np.float32)).to(device),
            torch.from_numpy(y[sel + 1].astype(np.int64)).to(device),
        )

    def evaluate(starts: Array) -> tuple[float, float]:
        if len(starts) == 0:
            return float("nan"), float("nan")
        net.eval()
        losses, correct = [], []
        with torch.no_grad():
            for lo in range(0, len(starts), 512):
                sel = np.arange(lo, min(len(starts), lo + 512))
                z, z1, yy = batch(starts, sel)
                pred, logits = net(z)
                mse = torch.mean((pred - z1) ** 2)
                ce = nn.functional.cross_entropy(logits, yy)
                losses.append(float((mse + 0.2 * ce).item()))
                correct.append(logits.argmax(-1) == yy)
        return float(np.mean(losses)), float(torch.cat(correct).float().mean().item())

    best = (float("inf"), -1, copy.deepcopy(net.state_dict()))
    hist = TrainHistory([], [], [], -1, float("inf"))
    for epoch in range(train_cfg.epochs):
        net.train()
        order = rng.permutation(len(train_starts))
        ep = []
        for lo in range(0, len(order), train_cfg.batch_size):
            idx = order[lo : lo + train_cfg.batch_size]
            z, z1, yy = batch(train_starts, idx)
            pred, logits = net(z)
            loss = torch.mean((pred - z1) ** 2) + 0.2 * nn.functional.cross_entropy(logits, yy)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if train_cfg.grad_clip:
                nn.utils.clip_grad_norm_(net.parameters(), train_cfg.grad_clip)
            opt.step()
            ep.append(float(loss.item()))
        val_loss, val_acc = evaluate(val_starts)
        hist.train_loss.append(float(np.mean(ep)))
        hist.val_loss.append(val_loss)
        hist.val_behavior_accuracy.append(val_acc)
        if np.isfinite(val_loss) and val_loss < best[0]:
            best = (val_loss, epoch, copy.deepcopy(net.state_dict()))
        if epoch - best[1] >= train_cfg.patience:
            break
    net.load_state_dict(best[2])
    hist.best_epoch = best[1]
    hist.best_val_loss = best[0]
    return hist


@dataclass
class LatentMLPDynamics:
    net: _LatentNet
    encoder: PCALatent
    n_neurons: int
    latent_dim: int
    probe: BehaviorProbe
    residual_std: Array
    device: str
    name: str = "latent_mlp"

    def encode(self, X: Array) -> Array:
        return self.encoder.encode(X)

    def decode(self, Z: Array) -> Array:
        return self.encoder.decode(Z)

    def latent(self, history: Array) -> Array:
        return self.encode(np.asarray(history)[:, -1])

    def predict_next_latent(self, Z: Array) -> Array:
        self.net.eval()
        with torch.no_grad():
            out, _ = self.net(torch.from_numpy(np.asarray(Z, dtype=np.float32)).to(self.device))
        return out.cpu().numpy()

    def predict_next(self, history: Array) -> Array:
        Z = self.encode(np.asarray(history)[:, -1])
        return self.decode(self.predict_next_latent(Z))

    def behavior_proba(self, history: Array) -> Array:
        Z = self.encode(_last_states(history, self.n_neurons))
        self.net.eval()
        with torch.no_grad():
            _, logits = self.net(torch.from_numpy(Z.astype(np.float32)).to(self.device))
        return _softmax(logits.cpu().numpy())

    def rollout(
        self,
        x0: Array,
        steps: int,
        *,
        seed: int = 0,
        noise_std: float = 0.0,
        perturb: PerturbFn | None = None,
        initial_history: Array | None = None,
        stochastic: bool = False,
    ) -> Array:
        rng = np.random.default_rng(seed)
        hist = _initial_history(x0, initial_history, self.n_neurons)
        X = [hist[-1].copy()]
        z = self.encode(X[0][None])[0]
        for s in range(steps):
            z = self.predict_next_latent(z[None])[0]
            if noise_std > 0:
                z = z + rng.normal(scale=noise_std, size=z.shape)
            x = self.decode(z[None])[0]
            if perturb is not None:
                x = perturb(x, s)
                z = self.encode(x[None])[0]
            X.append(x)
        return np.stack(X)


@dataclass
class LatentKNNDynamics:
    encoder: PCALatent
    Z_train: Array
    Z_next: Array
    probe: BehaviorProbe
    n_neurons: int
    latent_dim: int
    k: int = 25
    temperature: float = 1.0
    ood_quantile: float = 0.99
    name: str = "latent_knn"

    def latent(self, history: Array) -> Array:
        return self.encoder.encode(_last_states(history, self.n_neurons))

    def _neighbors(self, z: Array, k: int | None = None) -> tuple[Array, Array]:
        kk = min(k or self.k, len(self.Z_train))
        d = np.linalg.norm(self.Z_train - z[None], axis=1)
        idx = np.argpartition(d, kk - 1)[:kk]
        idx = idx[np.argsort(d[idx])]
        return idx, d[idx]

    def predict_next_latent(self, Z: Array, *, deterministic: bool = True) -> Array:
        Z = np.atleast_2d(Z)
        out = np.empty_like(Z)
        for i, z in enumerate(Z):
            idx, dist = self._neighbors(z)
            w = np.exp(-(dist**2) / (2 * self.temperature**2))
            w = (
                np.ones(len(idx)) / len(idx)
                if not np.isfinite(w).all() or w.sum() <= 0
                else w / w.sum()
            )
            if deterministic:
                delta = ((self.Z_next[idx] - self.Z_train[idx]) * w[:, None]).sum(0)
            else:
                j = int(np.random.default_rng(i).choice(len(idx), p=w))
                delta = self.Z_next[idx[j]] - self.Z_train[idx[j]]
            out[i] = z + delta
        return out

    def predict_next(self, history: Array) -> Array:
        Z = self.latent(history)
        return self.decode(self.predict_next_latent(Z))

    def decode(self, Z: Array) -> Array:
        return self.encoder.decode(Z)

    def encode(self, X: Array) -> Array:
        return self.encoder.encode(X)

    def behavior_proba(self, history: Array) -> Array:
        return self.probe.predict_proba(self.latent(history))

    def ood_score(self, X: Array) -> Array:
        """Distance to the k-th training latent state (higher = less supported)."""
        Z = self.encode(X)
        scores = []
        for z in Z:
            _, d = self._neighbors(z)
            scores.append(float(d[-1]))
        return np.asarray(scores)

    def rollout(
        self,
        x0: Array,
        steps: int,
        *,
        seed: int = 0,
        noise_std: float = 0.0,
        perturb: PerturbFn | None = None,
        stochastic: bool = True,
        initial_history: Array | None = None,
    ) -> Array:
        rng = np.random.default_rng(seed)
        hist = _initial_history(x0, initial_history, self.n_neurons)
        X = [hist[-1].copy()]
        z = self.encode(X[0][None])[0]
        for s in range(steps):
            idx, dist = self._neighbors(z)
            w = np.exp(-(dist**2) / (2 * self.temperature**2))
            w = (
                np.ones(len(idx)) / len(idx)
                if not np.isfinite(w).all() or w.sum() <= 0
                else w / w.sum()
            )
            j = int(rng.choice(len(idx), p=w)) if stochastic else int(np.argmax(w))
            delta = self.Z_next[idx[j]] - self.Z_train[idx[j]]
            z = z + delta + (rng.normal(scale=noise_std, size=z.shape) if noise_std > 0 else 0.0)
            x = self.decode(z[None])[0]
            if perturb is not None:
                x = perturb(x, s)
                z = self.encode(x[None])[0]
            X.append(x)
        return np.stack(X)


def fit_pca_latent_model(
    X: Array,
    y: Array,
    train_starts: Array,
    val_starts: Array,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    *,
    kind: str = "latent_mlp",
    k: int = 25,
) -> tuple[DynamicsModel, TrainHistory]:
    set_seed(train_cfg.seed, num_threads=train_cfg.num_threads)
    enc = PCALatent.fit(X[train_starts], model_cfg.latent_dim)
    Z = enc.encode(X)
    probe = BehaviorProbe(seed=train_cfg.seed).fit(Z[train_starts], y[train_starts])
    if kind == "latent_knn":
        hist = TrainHistory([], [], [], 0, float("nan"))
        return (
            LatentKNNDynamics(
                encoder=enc,
                Z_train=Z[train_starts],
                Z_next=Z[train_starts + 1],
                probe=probe,
                n_neurons=X.shape[1],
                latent_dim=enc.pca.n_components_,
                k=k,
            ),
            hist,
        )
    if kind not in {"latent_mlp"}:
        raise ValueError(f"Unknown PCA latent model {kind!r}")
    net = _LatentNet(enc.pca.n_components_, model_cfg.hidden_dim, model_cfg.n_behaviors, model_cfg.dropout)
    hist = _train_latent_net(net, Z, y, train_starts, val_starts, train_cfg)
    resid = Z[train_starts + 1] - Z[train_starts]
    model = LatentMLPDynamics(
        net=net,
        encoder=enc,
        n_neurons=X.shape[1],
        latent_dim=enc.pca.n_components_,
        probe=probe,
        residual_std=resid.std(axis=0),
        device=str(choose_device(train_cfg.device)),
        name="latent_mlp",
    )
    return model, hist


def fit_model(
    kind: str,
    X: Array,
    y: Array,
    train_starts: Array,
    val_starts: Array,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> tuple[DynamicsModel, TrainHistory]:
    if kind == "persistence":
        probe = BehaviorProbe(seed=train_cfg.seed).fit(X[train_starts], y[train_starts])
        return PersistenceModel(n_neurons=X.shape[1], probe=probe), TrainHistory([], [], [], 0, float("nan"))
    if kind == "var":
        model = VARModel.fit(X, y, train_starts, history=model_cfg.history, seed=train_cfg.seed)
        return model, TrainHistory([], [], [], 0, float("nan"))
    if kind in {"mlp", "mlp_residual"}:
        return fit_mlp(
            X, y, train_starts, val_starts, model_cfg, train_cfg, residual=kind.endswith("residual")
        )
    if kind == "gru":
        return fit_gru(X, y, train_starts, val_starts, model_cfg, train_cfg)
    if kind in {"latent_mlp", "latent_knn"}:
        return fit_pca_latent_model(X, y, train_starts, val_starts, model_cfg, train_cfg, kind=kind)
    raise ValueError(f"Unknown model kind {kind!r}")
