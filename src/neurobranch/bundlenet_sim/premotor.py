"""Biophysical premotor-network sandbox using the published fitted weights.

The Celegans_premotor release exports a 205 x 205 average synaptic matrix ``A``
and a 205-vector ``d`` in ``avg_model_weights.jld2``.  The ODE form and default
parameters below follow the published Julia model (``functions_simulation.jl``
and ``simulations.ipynb``):

    du_j/dt = (1/tau) * (
        a * (u_j - fp1) * (u_j - fp2) * (u_j - fp3)
        + d_j
        + gamma * sum_i A[j, i] * ReLU(u_i)
        + beta  * sum_i W[j, i] * (u_i - u_j)
    )

The public JLD2 export omits the gap-junction matrix ``W`` and the neuron-name
order.  We recover both from the release's own ``df5.csv`` and the notebook's
printed 205-neuron loop order, then verify that every non-zero entry of ``A``
lies inside the chemical-synapse support (913/913 entries).  If those local
derivatives are absent, ``W`` defaults to zero and nodes are labelled by index,
and the UI says so.  This remains a sandbox, not a validated digital twin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from scipy.integrate import solve_ivp


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "processed" / "premotor"


@dataclass(frozen=True)
class PremotorModel:
    A: np.ndarray
    d: np.ndarray
    W: np.ndarray | None = None
    neuron_names: tuple[str, ...] | None = None
    source: str = ""

    @property
    def n_nodes(self) -> int:
        return int(self.A.shape[0])

    def node_strength(self) -> np.ndarray:
        return np.abs(self.A).sum(axis=1) + np.abs(self.d)

    def node_label(self, index: int) -> str:
        if self.neuron_names is not None:
            return self.neuron_names[index]
        return f"node {index}"

    def node_index(self, name: str) -> int:
        if self.neuron_names is None:
            raise ValueError("neuron names are not available")
        return self.neuron_names.index(name)


def load_premotor(data_dir: Path | str | None = None) -> PremotorModel:
    root = Path(data_dir) if data_dir is not None else default_data_dir()
    path = root / "avg_model_weights.jld2"
    if not path.exists():
        raise FileNotFoundError(f"missing premotor weights: {path}")
    with h5py.File(path, "r") as handle:
        A = np.asarray(handle["A_10takes_avg"], dtype=np.float64)
        d = np.asarray(handle["d_10takes_avg"], dtype=np.float64).reshape(-1)
    if A.shape[0] != A.shape[1] or A.shape[0] != len(d):
        raise ValueError(f"inconsistent premotor shapes: A={A.shape}, d={d.shape}")
    W = None
    W_path = root / "W.npy"
    if W_path.exists():
        W = np.asarray(np.load(W_path), dtype=np.float64)
        if W.shape != A.shape:
            raise ValueError(f"W shape {W.shape} does not match A {A.shape}")
    names: tuple[str, ...] | None = None
    names_path = root / "neuron_names.txt"
    if names_path.exists():
        candidate = tuple(line.strip() for line in names_path.read_text(encoding="utf-8").splitlines() if line.strip())
        if len(candidate) == A.shape[0]:
            names = candidate
    return PremotorModel(A=A, d=d, W=W, neuron_names=names, source=str(path))


@dataclass
class PremotorParameters:
    a: float = -2.0
    fp1: float = -0.8
    fp2: float = 0.1
    fp3: float = 1.0
    gamma: float = 1.4
    beta: float = 0.0
    tau: float = 0.2
    duration: float = 20.0
    n_samples: int = 400
    noise: float = 0.0
    stim_node: int | None = None
    stim_amplitude: float = 0.0
    stim_start: float = 2.0
    stim_end: float = 5.0
    ablate_nodes: tuple[int, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        if self.duration <= 0 or self.n_samples < 2:
            raise ValueError("duration and n_samples must be positive")


def simulate(
    model: PremotorModel,
    params: PremotorParameters,
    u0: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(time, activity)`` with activity shape ``(n_nodes, n_samples)``."""
    params.validate()
    A = model.A.copy()
    d = model.d.copy()
    W = None if model.W is None else model.W.copy()
    for node in params.ablate_nodes:
        if not 0 <= node < model.n_nodes:
            raise ValueError(f"ablate node out of range: {node}")
        A[node, :] = 0.0
        A[:, node] = 0.0
        d[node] = 0.0
        if W is not None:
            W[node, :] = 0.0
            W[:, node] = 0.0
    if params.stim_node is not None and not 0 <= params.stim_node < model.n_nodes:
        raise ValueError(f"stim node out of range: {params.stim_node}")

    rng = np.random.default_rng(seed)
    if u0 is None:
        u0 = np.zeros(model.n_nodes, dtype=np.float64)
        if params.noise > 0:
            u0 = u0 + rng.normal(0.0, params.noise, size=model.n_nodes)
    u0 = np.asarray(u0, dtype=np.float64).reshape(model.n_nodes)

    def rhs(t: float, u: np.ndarray) -> np.ndarray:
        u_clip = np.clip(u, -8.0, 8.0)
        chemical = A @ np.maximum(u_clip, 0.0)
        gap = np.zeros_like(u_clip) if W is None else (W.sum(axis=1) * u_clip - W @ u_clip)
        drive = d.copy()
        if (
            params.stim_node is not None
            and params.stim_amplitude != 0.0
            and params.stim_start <= t <= params.stim_end
        ):
            drive[params.stim_node] += params.stim_amplitude
        du = (
            params.a * (u_clip - params.fp1) * (u_clip - params.fp2) * (u_clip - params.fp3)
            + drive
            + params.gamma * chemical
            + params.beta * gap
        ) / params.tau
        return np.clip(du, -1e4, 1e4)

    t_eval = np.linspace(0.0, params.duration, params.n_samples)
    solution = solve_ivp(
        rhs,
        (0.0, params.duration),
        u0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-4,
        atol=1e-6,
        max_step=max(params.duration / 200.0, 1e-3),
    )
    if not solution.success:
        raise RuntimeError(f"premotor ODE failed: {solution.message}")
    return solution.t, solution.y
