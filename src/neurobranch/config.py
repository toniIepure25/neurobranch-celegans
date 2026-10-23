"""Central configuration and path handling.

Configuration files are intentionally plain YAML so that every experiment can
be archived alongside a Git revision.  All paths inside YAML are interpreted
relative to the project root, never to the caller's current directory.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def find_project_root(start: Path | None = None) -> Path:
    """Find the nearest ancestor containing ``pyproject.toml``."""
    env = os.environ.get("NEUROBRANCH_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    cur = (start or Path.cwd()).resolve()
    for candidate in (cur, *cur.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return cur


PROJECT_ROOT = find_project_root()


def resolve_path(value: str | Path, root: Path | None = None) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (root or PROJECT_ROOT) / p


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_state(root: Path | None = None) -> dict[str, Any]:
    """Best-effort machine-readable Git provenance (never fatal)."""
    import subprocess

    cwd = str(root or PROJECT_ROOT)

    def _run(*args: str) -> str | None:
        try:
            return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    return {
        "commit": _run("git", "rev-parse", "HEAD"),
        "dirty": bool(_run("git", "status", "--porcelain")),
        "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
    }


@dataclass
class DataConfig:
    dataset: str = "kato2015"
    path: str = "data/raw/kato2015/WT_NoStim.mat"
    exclude_behavior_neurons: bool = True
    min_identified_neurons: int = 5
    macrostates: bool = True
    smoothing_sigma_frames: float = 0.0
    detrend_window_frames: int = 0
    detrend_warmup_frames: int = 0


@dataclass
class SplitConfig:
    mode: str = "blocked"  # temporal | blocked | worm_holdout | leave_one_worm_out
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    purge_frames: int = 12
    n_blocks: int = 5
    test_block: int = 2
    val_block: int = 4
    holdout_worm: int = 4
    seed: int = 17


@dataclass
class ModelConfig:
    name: str = "latent_mlp"
    latent_dim: int = 16
    hidden_dim: int = 128
    history: int = 1
    dropout: float = 0.0
    behavior_weight: float = 0.2
    n_behaviors: int = 4
    gamma: float = 0.9
    window: int = 15


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 12
    grad_clip: float = 1.0
    seed: int = 17
    device: str = "auto"
    num_threads: int = 4
    noise_std: float = 0.0


@dataclass
class EvaluationConfig:
    horizons: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    seeds: list[int] = field(default_factory=lambda: [17, 23, 31])
    bootstrap_samples: int = 500
    transition_horizon: int = 5
    branching_k: int = 40
    min_neighbors: int = 15


@dataclass
class ExperimentConfig:
    name: str = "baseline"
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output_dir: str = "artifacts/runs"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentConfig:
        payload = dict(payload or {})
        section_types = {
            "data": DataConfig,
            "split": SplitConfig,
            "model": ModelConfig,
            "train": TrainConfig,
            "evaluation": EvaluationConfig,
        }
        sections: dict[str, Any] = {}
        for f in fields(cls):
            if f.name in section_types:
                raw = payload.get(f.name) or {}
                sub = {k: v for k, v in raw.items() if v is not None}
                sections[f.name] = section_types[f.name](**sub)
            elif f.name in payload:
                sections[f.name] = payload[f.name]
        return cls(**sections)

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        with open(resolve_path(path), encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        return cls.from_dict(payload)

    def save(self, path: str | Path) -> Path:
        target = resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)
        return target


def ensure_dirs(root: Path | None = None) -> None:
    root = root or PROJECT_ROOT
    for rel in ("artifacts", "figures", "reports", "data/processed", "data/interim"):
        (root / rel).mkdir(parents=True, exist_ok=True)
