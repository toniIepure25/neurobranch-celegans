"""Architecture-compatible small GRU decoder for Phase 5.

The published SAAV decoder is a JAX/Equinox GRUCell with hidden size three,
dropout on every scanned state, LayerNorm over the hidden state, a linear
sigmoid output at every timestep, and class-balanced event sampling.  This
module implements the same architectural contract in PyTorch.  It is an
independent reimplementation, not a copy of the GPL-3.0 reference code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class RNNConfig:
    hidden_size: int = 3
    dropout: float = 0.5
    learning_rate: float = 1e-3
    epochs: int = 60
    batch_size: int = 64
    patience: int = 12
    weight_decay: float = 0.0
    seed: int = 17


class SmallGRUDecoder(nn.Module):
    """GRU sequence decoder with per-timestep sigmoid outputs."""

    def __init__(self, in_size: int, *, hidden_size: int = 3, dropout: float = 0.5) -> None:
        super().__init__()
        if in_size < 1:
            raise ValueError("in_size must be positive")
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.in_size = int(in_size)
        self.hidden_size = int(hidden_size)
        self.gru = nn.GRU(in_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return probabilities with shape ``(batch, time)``."""
        if x.ndim != 3 or x.shape[-1] != self.in_size:
            raise ValueError(f"expected (batch, time, {self.in_size}), got {tuple(x.shape)}")
        hidden, _ = self.gru(x)
        hidden = self.dropout(hidden)
        hidden = self.norm(hidden)
        return torch.sigmoid(self.linear(hidden)).squeeze(-1)


def _event_class_lookup(groups: np.ndarray, y: np.ndarray) -> dict[Any, int]:
    lookup: dict[Any, int] = {}
    for group, label in zip(groups.tolist(), y.tolist()):
        old = lookup.setdefault(group, int(label))
        if old != int(label):
            raise ValueError(f"event {group!r} has inconsistent labels")
    return lookup


def _balanced_event_mask(
    groups: np.ndarray,
    y: np.ndarray,
    *,
    rng: np.random.Generator,
    sample: bool,
) -> np.ndarray:
    lookup = _event_class_lookup(groups, y)
    positives = [g for g, label in lookup.items() if label == 1]
    negatives = [g for g, label in lookup.items() if label == 0]
    n = min(len(positives), len(negatives))
    if n == 0:
        raise ValueError("class-balanced training needs both classes")
    if sample:
        positives = list(rng.choice(positives, size=n, replace=False))
        negatives = list(rng.choice(negatives, size=n, replace=False))
    else:
        positives = positives[:n]
        negatives = negatives[:n]
    keep = set(positives) | set(negatives)
    return np.fromiter((g in keep for g in groups.tolist()), dtype=bool, count=len(groups))


def _as_tensor(x: np.ndarray, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=dtype)


def _loss(prob: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    prob = prob.clamp(1e-6, 1.0 - 1e-6)
    return -(y * torch.log(prob) + (1.0 - y) * torch.log(1.0 - prob)).mean()


def train_rnn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    groups_valid: np.ndarray,
    *,
    sequence_pool: str = "sections",
    config: RNNConfig | None = None,
) -> tuple[SmallGRUDecoder, dict[str, Any]]:
    """Train a class-balanced small GRU.

    ``sequence_pool='sections'`` trains on all four-frame sections with a
    per-timestep loss (the official offline contract).  ``sequence_pool='last'``
    trains only the final timestep, which is the online prediction contract.
    Event groups, not sections, define class balancing.
    """
    cfg = config or RNNConfig()
    if sequence_pool not in {"sections", "last"}:
        raise ValueError("sequence_pool must be 'sections' or 'last'")
    x_train = np.asarray(x_train, dtype=np.float32)
    x_valid = np.asarray(x_valid, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).ravel()
    y_valid = np.asarray(y_valid, dtype=np.float32).ravel()
    groups_train = np.asarray(groups_train)
    groups_valid = np.asarray(groups_valid)
    if len(x_train) != len(y_train) or len(x_valid) != len(y_valid):
        raise ValueError("x/y length mismatch")
    if x_train.ndim != 3 or x_valid.ndim != 3 or x_train.shape[2] != x_valid.shape[2]:
        raise ValueError("train/valid inputs must be 3-D with the same channel count")

    torch.manual_seed(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)
    model = SmallGRUDecoder(
        x_train.shape[2],
        hidden_size=cfg.hidden_size,
        dropout=cfg.dropout,
    )
    optim = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    xt = _as_tensor(x_train)
    yt = _as_tensor(y_train)
    xv = _as_tensor(x_valid)
    yv = _as_tensor(y_valid)

    valid_mask = _balanced_event_mask(groups_valid, y_valid, rng=np_rng, sample=False)
    xv_bal = xv[torch.as_tensor(valid_mask)]
    yv_bal = yv[torch.as_tensor(valid_mask)]
    if sequence_pool == "last":
        xv_bal = xv_bal[:, -1:, :]

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    history: list[dict[str, float]] = []
    stale = 0

    for epoch in range(cfg.epochs):
        model.train()
        mask = _balanced_event_mask(groups_train, y_train, rng=np_rng, sample=True)
        xb = xt[torch.as_tensor(mask)]
        yb = yt[torch.as_tensor(mask)]
        if sequence_pool == "last":
            xb = xb[:, -1:, :]
        perm = torch.as_tensor(np_rng.permutation(len(xb)))
        xb = xb[perm]
        yb = yb[perm]
        total = 0.0
        n_batches = 0
        for start in range(0, len(xb), cfg.batch_size):
            x_batch = xb[start : start + cfg.batch_size]
            y_batch = yb[start : start + cfg.batch_size]
            optim.zero_grad(set_to_none=True)
            prob = model(x_batch)
            if sequence_pool == "sections":
                loss = _loss(prob, y_batch[:, None].expand_as(prob))
            else:
                loss = _loss(prob[:, -1], y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            total += float(loss.detach())
            n_batches += 1

        model.eval()
        with torch.no_grad():
            prob_val = model(xv_bal)
            if sequence_pool == "sections":
                val_loss = float(_loss(prob_val, yv_bal[:, None].expand_as(prob_val)))
            else:
                val_loss = float(_loss(prob_val[:, -1], yv_bal))
        train_loss = total / max(n_batches, 1)
        history.append({"epoch": float(epoch + 1), "train_loss": train_loss, "valid_loss": val_loss})
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_epoch": best_epoch,
        "best_valid_loss": best_loss,
        "epochs_run": len(history),
        "history": history,
        "sequence_pool": sequence_pool,
        "hidden_size": cfg.hidden_size,
        "dropout": cfg.dropout,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
    }


def predict_rnn(
    model: SmallGRUDecoder,
    x: np.ndarray,
    groups: np.ndarray,
    *,
    sequence_pool: str = "sections",
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-section or per-sequence probabilities and row groups.

    For section inputs, the returned array has one probability per section
    (mean over the four timesteps), matching the official section-level
    evaluation.  Event aggregation is the caller's responsibility.
    """
    x = np.asarray(x, dtype=np.float32)
    groups = np.asarray(groups)
    if len(x) != len(groups):
        raise ValueError("x and groups length mismatch")
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = _as_tensor(x[start : start + batch_size])
            prob = model(xb)
            prob = prob[:, -1] if sequence_pool == "last" else prob.mean(dim=1)
            out.append(prob.cpu().numpy())
    if not out:
        return np.empty(0), groups
    return np.concatenate(out), groups
