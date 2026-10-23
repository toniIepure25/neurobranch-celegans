"""Torch utilities shared by dynamics models."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def set_seed(seed: int, *, num_threads: int = 4) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, int(num_threads)))
    torch.use_deterministic_algorithms(False)


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLPNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_classes: int,
        *,
        dropout: float = 0.0,
        residual: bool = False,
    ):
        super().__init__()
        self.residual = residual
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.out = nn.Linear(hidden_dim, out_dim)
        self.behavior = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        h = self.encoder(x)
        delta = self.out(h)
        return (x + delta if self.residual else delta), self.behavior(h)


class GRUNet(nn.Module):
    def __init__(self, n_neurons: int, hidden_dim: int, n_classes: int, *, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(n_neurons, hidden_dim, batch_first=True, num_layers=1)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.out = nn.Linear(hidden_dim, n_neurons)
        self.behavior = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, h = self.gru(x)
        z = self.head(self.dropout(h[-1]))
        return self.out(z), self.behavior(z)


@dataclass
class TrainHistory:
    train_loss: list[float]
    val_loss: list[float]
    val_behavior_accuracy: list[float]
    best_epoch: int
    best_val_loss: float

    def as_dict(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_behavior_accuracy": self.val_behavior_accuracy,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
        }
