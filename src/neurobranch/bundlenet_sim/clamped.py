"""Published-configuration premotor simulation with clamped sensory/premotor signals.

The public Celegans_premotor model has 205 nodes, but the published simulation
uses an **84-neuron core**: 15 intrinsic neurons are integrated and the other 69
are clamped to the recorded calcium traces.  This module implements that exact
configuration, including the published parameters and the forward/backward
behaviour readout.

It is deliberately separate from :mod:`premotor`, which keeps the 205-node
free-run sandbox.  The clamped mode is a replay/sandbox: it shows what the
published equations do when driven by a measured recording.  It is not a
digital twin and it does not establish whole-brain mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from .premotor import PremotorModel
from .recording import INTRINSIC_NEURONS, ClampedDataError, ClampedRecording, behavior_readout


class ClampedSimulationError(RuntimeError):
    """Raised when the clamped ODE cannot be integrated."""


@dataclass(frozen=True)
class SignalPulse:
    neuron: str
    amplitude: float
    start_s: float
    duration_s: float

    def validate(self, n_frames: int, dt: float) -> None:
        if not self.neuron:
            raise ValueError("pulse neuron must be non-empty")
        if self.duration_s <= 0:
            raise ValueError("pulse duration must be positive")
        if self.start_s < 0 or self.start_s >= n_frames * dt:
            raise ValueError("pulse start is outside the recording")


@dataclass
class ClampedParameters:
    a: float = -2.0
    fp1: float = -0.8
    fp2: float = 0.1
    fp3: float = 1.0
    gamma: float = 1.4
    beta: float = 10.0
    tau: float = 0.2
    clip: float = 20.0
    signal_pulses: tuple[SignalPulse, ...] = field(default_factory=tuple)
    ablate_intrinsic: tuple[str, ...] = field(default_factory=tuple)

    def validate(self, recording: ClampedRecording) -> None:
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        if self.clip <= 0:
            raise ValueError("clip must be positive")
        known = set(recording.signal_names)
        for pulse in self.signal_pulses:
            if pulse.neuron not in known:
                raise ValueError(f"pulse neuron is not a clamped signal: {pulse.neuron}")
            pulse.validate(recording.n_frames, recording.dt)
        for name in self.ablate_intrinsic:
            if name not in recording.intrinsic_names:
                raise ValueError(f"ablation target is not intrinsic: {name}")


@dataclass
class ClampedResult:
    recording_id: str
    time: np.ndarray
    intrinsic: np.ndarray
    signals: np.ndarray
    full: np.ndarray
    truth_intrinsic: np.ndarray
    behavior_truth: np.ndarray
    behavior_sim: np.ndarray
    parameters: ClampedParameters
    metrics: dict[str, object]
    status: str
    method: str
    source: str

    @property
    def intrinsic_names(self) -> tuple[str, ...]:
        return INTRINSIC_NEURONS


def _js(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float).reshape(-1)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / np.maximum(b[mask], 1e-12))))

    return 0.5 * (kl(p, m) + kl(q, m))


def _distribution(labels: np.ndarray) -> np.ndarray:
    return np.bincount(labels + 1, minlength=3).astype(float)


def _transition(labels: np.ndarray) -> np.ndarray:
    matrix = np.full((3, 3), 1e-9, dtype=float)
    for a, b in zip(labels[:-1], labels[1:]):
        matrix[int(a) + 1, int(b) + 1] += 1.0
    return matrix / matrix.sum(axis=1, keepdims=True)


def _dwell_seconds(labels: np.ndarray, value: int, dt: float) -> float:
    runs: list[int] = []
    current = 0
    for label in labels:
        if int(label) == value:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return float(np.mean(runs) * dt) if runs else float("nan")


def _metrics(
    truth: np.ndarray,
    sim: np.ndarray,
    truth_behavior: np.ndarray,
    sim_behavior: np.ndarray,
    dt: float,
    max_abs: float,
    status: str,
) -> dict[str, object]:
    return {
        "finite": bool(np.isfinite(sim).all()),
        "max_abs": float(max_abs),
        "status": status,
        "state_agreement": float(np.mean(truth_behavior == sim_behavior)),
        "behavior_distribution_js": _js(
            _distribution(truth_behavior), _distribution(sim_behavior)
        ),
        "behavior_transition_js": _js(_transition(truth_behavior), _transition(sim_behavior)),
        "truth_distribution": {
            str(k): float(v) for k, v in zip((-1, 0, 1), _distribution(truth_behavior) / len(truth_behavior))
        },
        "sim_distribution": {
            str(k): float(v) for k, v in zip((-1, 0, 1), _distribution(sim_behavior) / len(sim_behavior))
        },
        "truth_dwell_s": {
            "forward": _dwell_seconds(truth_behavior, 1, dt),
            "reverse": _dwell_seconds(truth_behavior, -1, dt),
        },
        "sim_dwell_s": {
            "forward": _dwell_seconds(sim_behavior, 1, dt),
            "reverse": _dwell_seconds(sim_behavior, -1, dt),
        },
        "truth_mean": float(truth.mean()),
        "sim_mean": float(sim.mean()),
        "truth_std": float(truth.std()),
        "sim_std": float(sim.std()),
    }


def _rhs_factory(
    A: np.ndarray,
    d: np.ndarray,
    W: np.ndarray,
    recording: ClampedRecording,
    parameters: ClampedParameters,
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    intrinsic_index = {name: i for i, name in enumerate(recording.intrinsic_names)}
    core_index = {name: i for i, name in enumerate(recording.core_names)}
    ids_intrinsic = np.array([core_index[name] for name in recording.intrinsic_names], dtype=np.int64)
    ids_signal = np.array([core_index[name] for name in recording.signal_names], dtype=np.int64)
    signals = recording.traces[ids_signal].copy()
    for pulse in parameters.signal_pulses:
        row = recording.signal_names.index(pulse.neuron)
        mask = (recording.time >= pulse.start_s) & (
            recording.time < pulse.start_s + pulse.duration_s
        )
        signals[row, mask] += pulse.amplitude
    ablated = np.array(
        [intrinsic_index[name] for name in parameters.ablate_intrinsic], dtype=np.int64
    )
    ablated_core = ids_intrinsic[ablated] if len(ablated) else np.array([], dtype=np.int64)

    def rhs(t: float, u: np.ndarray) -> np.ndarray:
        ut = np.empty(len(recording.core_names), dtype=float)
        ut[ids_intrinsic] = u
        ut[ids_signal] = np.array(
            [np.interp(t, recording.time, row) for row in signals], dtype=float
        )
        if len(ablated_core):
            ut[ablated_core] = 0.0
        chemical = A @ np.maximum(ut, 0.0)
        # The notebook's diag(W * (repeat(u, 1, N) - repeat(u, 1, N)')) expands
        # to sum_j W[i, j] * (u_j - u_i), not the opposite sign.
        gap = W @ ut - W.sum(axis=1) * ut
        du = (
            parameters.a
            * (ut - parameters.fp1)
            * (ut - parameters.fp2)
            * (ut - parameters.fp3)
            + d
            + parameters.gamma * chemical
            + parameters.beta * gap
        ) / parameters.tau
        if len(ablated):
            du[ablated_core] = 0.0
        return np.clip(du[ids_intrinsic], -1e5, 1e5)

    return rhs, ids_intrinsic, ids_signal, signals


def simulate_clamped(
    model: PremotorModel,
    recording: ClampedRecording,
    parameters: ClampedParameters | None = None,
) -> ClampedResult:
    """Integrate the published 15-intrinsic/69-signal clamped core."""
    parameters = parameters or ClampedParameters()
    parameters.validate(recording)
    if model.W is None:
        raise ClampedDataError("clamped simulation requires the recovered gap-junction matrix W")
    names = model.neuron_names
    if names is None:
        raise ClampedDataError("clamped simulation requires neuron names")
    core_index = {name: i for i, name in enumerate(recording.core_names)}
    model_index = {name: i for i, name in enumerate(names)}
    missing = [name for name in recording.core_names if name not in model_index]
    if missing:
        raise ClampedDataError(f"core neurons absent from model: {missing}")

    core = np.array([model_index[name] for name in recording.core_names], dtype=np.int64)
    A = model.A[np.ix_(core, core)].copy()
    d = model.d[core].copy()
    W = model.W[np.ix_(core, core)].copy()
    for name in parameters.ablate_intrinsic:
        local = core_index[name]
        A[local, :] = 0.0
        A[:, local] = 0.0
        d[local] = 0.0
        W[local, :] = 0.0
        W[:, local] = 0.0

    rhs, ids_intrinsic, ids_signal, signals = _rhs_factory(A, d, W, recording, parameters)
    u0 = recording.traces[ids_intrinsic, 0].copy()
    t_eval = recording.time
    method = "RK45"
    solution = solve_ivp(
        rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        u0,
        t_eval=t_eval,
        method=method,
        rtol=1e-5,
        atol=1e-7,
        max_step=recording.dt,
    )
    if not solution.success or solution.y.shape != (len(recording.intrinsic_names), len(t_eval)):
        method = "LSODA"
        solution = solve_ivp(
            rhs,
            (float(t_eval[0]), float(t_eval[-1])),
            u0,
            t_eval=t_eval,
            method=method,
            rtol=1e-5,
            atol=1e-7,
            max_step=recording.dt,
        )
    if not solution.success or solution.y.shape != (len(recording.intrinsic_names), len(t_eval)):
        raise ClampedSimulationError(
            f"{recording.recording_id}: integration failed ({solution.message}); "
            f"last t={solution.t[-1] if len(solution.t) else 'n/a'}"
        )

    intrinsic = np.asarray(solution.y, dtype=float)
    full = recording.traces.copy()
    full[ids_intrinsic] = intrinsic
    full[ids_signal] = signals
    behavior_sim = behavior_readout(full, recording.core_names)
    max_abs = float(np.nanmax(np.abs(intrinsic))) if intrinsic.size else float("nan")
    status = "ok" if np.isfinite(intrinsic).all() and max_abs <= parameters.clip else "unstable"
    metrics = _metrics(
        recording.traces[ids_intrinsic],
        intrinsic,
        recording.behavior,
        behavior_sim,
        recording.dt,
        max_abs,
        status,
    )
    return ClampedResult(
        recording_id=recording.recording_id,
        time=t_eval,
        intrinsic=intrinsic,
        signals=signals,
        full=full,
        truth_intrinsic=recording.traces[ids_intrinsic],
        behavior_truth=recording.behavior,
        behavior_sim=behavior_sim,
        parameters=parameters,
        metrics=metrics,
        status=status,
        method=method,
        source=recording.source,
    )


def negative_controls(
    model: PremotorModel,
    recording: ClampedRecording,
    parameters: ClampedParameters | None = None,
) -> dict[str, ClampedResult]:
    """Run the three controls used in the validation report."""
    parameters = parameters or ClampedParameters()
    beta0 = ClampedParameters(**{**parameters.__dict__, "beta": 0.0})
    no_signals = ClampedParameters(**{**parameters.__dict__, "signal_pulses": ()})
    # Zero-signal control is implemented by shifting all signals to their
    # recording mean; this preserves the ODE interface without fabricating data.
    zero = ClampedParameters(**{**parameters.__dict__, "signal_pulses": ()})
    shifted = ClampedRecording(
        recording_id=recording.recording_id,
        dt=recording.dt,
        time=recording.time,
        traces=recording.traces.copy(),
        core_names=recording.core_names,
        intrinsic_names=recording.intrinsic_names,
        signal_names=recording.signal_names,
        behavior=recording.behavior,
        replacement_map=recording.replacement_map,
        source=recording.source,
    )
    signal_ids = [recording.core_names.index(name) for name in recording.signal_names]
    shifted.traces[signal_ids] = 0.0
    return {
        "beta0": simulate_clamped(model, recording, beta0),
        "zero_signals": simulate_clamped(model, shifted, zero),
        "no_pulses": simulate_clamped(model, recording, no_signals),
    }


def validation_record(result: ClampedResult) -> dict[str, object]:
    return {
        "recording_id": result.recording_id,
        "status": result.status,
        "method": result.method,
        "source": result.source,
        "parameters": {
            "beta": result.parameters.beta,
            "gamma": result.parameters.gamma,
            "tau": result.parameters.tau,
            "a": result.parameters.a,
            "fp1": result.parameters.fp1,
            "fp2": result.parameters.fp2,
            "fp3": result.parameters.fp3,
            "clip": result.parameters.clip,
        },
        "metrics": result.metrics,
    }


def default_recordings_dir() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "references"
        / "external"
        / "Celegans_premotor"
        / "data"
        / "Atanas_data"
    )
