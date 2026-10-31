"""Unified simulator adapter around the existing dynamics checkpoints."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

from neurobranch.data.kato import BEHAVIOR_LABEL_NEURONS
from neurobranch.dynamics.checkpointing import load_model
from neurobranch.preprocessing import WormData, split_starts
from neurobranch.simulator.types import (
    PerturbationSpec,
    Rollout,
    RolloutEnsemble,
    SimulatorSpec,
)

PerturbFn = Callable[[np.ndarray, int], np.ndarray]


class SupportModel:
    """Distance-to-training-support diagnostic.

    The threshold is the 99th percentile of the leave-one-out training distance
    (second nearest neighbour because the first is the point itself).  It is a
    descriptive OOD flag, not a biological plausibility test.
    """

    def __init__(self, train_X: np.ndarray, *, quantile: float = 0.99):
        X = np.asarray(train_X, dtype=np.float64)
        if X.ndim != 2 or len(X) < 3:
            raise ValueError("SupportModel needs at least three training states")
        self.nn = NearestNeighbors(n_neighbors=1).fit(X)
        dist, _ = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X)
        # dist[:, 0] is self; the second neighbour estimates local support.
        self.threshold = float(np.quantile(dist[:, 1], quantile))

    def score(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None]
        d, _ = self.nn.kneighbors(arr)
        return d[:, 0]


class NeuralSimulator:
    """Free-running simulator for one worm and one fitted dynamics model.

    Parameters
    ----------
    model:
        Any existing dynamics model with ``rollout``, ``predict_next`` and
        ``behavior_proba``.
    data:
        The worm-specific preprocessed data.  Only training states are used to
        build the support diagnostic; no test state is passed to the model.
    """

    def __init__(
        self,
        model: Any,
        data: WormData,
        *,
        checkpoint: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
        empirical_evidence: dict[str, Any] | None = None,
    ):
        self.model = model
        self.data = data
        self.checkpoint = str(checkpoint) if checkpoint is not None else None
        self.metadata = dict(metadata or {})
        self.empirical_evidence = empirical_evidence
        self.support = SupportModel(data.X[data.masks["train"]])
        if int(model.n_neurons) != int(data.X.shape[1]):
            raise ValueError(
                f"Checkpoint has {model.n_neurons} neurons, data has {data.X.shape[1]}"
            )

    @property
    def n_neurons(self) -> int:
        return int(self.data.X.shape[1])

    @property
    def model_kind(self) -> str:
        return str(getattr(self.model, "name", type(self.model).__name__))

    @property
    def spec(self) -> SimulatorSpec:
        return SimulatorSpec(
            model_kind=self.model_kind,
            history=int(getattr(self.model, "history_len", 1)),
            checkpoint=self.checkpoint,
            metadata=self.metadata,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        data: WormData,
        *,
        empirical_evidence: dict[str, Any] | None = None,
    ) -> NeuralSimulator:
        model, metadata = load_model(checkpoint)
        return cls(
            model,
            data,
            checkpoint=checkpoint,
            metadata=metadata,
            empirical_evidence=empirical_evidence,
        )

    def _target_index(self, target: str) -> int:
        if target in BEHAVIOR_LABEL_NEURONS:
            raise ValueError(
                f"{target} defines the behavioural label and is not in the predictor set; "
                "the observational model cannot represent an intervention on it."
            )
        return int(self.data.neuron_index(target))

    def _apply(self, x: np.ndarray, idx: int, spec: PerturbationSpec) -> np.ndarray:
        y = np.asarray(x, dtype=np.float64).copy()
        if spec.mode == "additive":
            y[idx] += spec.amplitude
        elif spec.mode == "clamp":
            y[idx] = spec.amplitude
        elif spec.mode == "zero":
            y[idx] = 0.0
        elif spec.mode == "scale":
            y[idx] *= spec.amplitude
        return y

    def _perturbation_callback(
        self, spec: PerturbationSpec | None
    ) -> tuple[PerturbFn | None, np.ndarray, dict[str, Any]]:
        if spec is None:
            return None, np.asarray(self.data.X[:0]), {}
        if spec.protocol == "empirical_response":
            if not (self.empirical_evidence and self.empirical_evidence.get("validated", False)):
                raise PermissionError(
                    "empirical_response requires a matched, validated stimulation-response "
                    "model. None is attached to this simulator; use dynamical_input or "
                    "initial_state and label the output model-dependent."
                )
            raise PermissionError(
                "No validated empirical response model is registered in this build. "
                "Gate 6 is not_evaluable for Kato."
            )
        idx = self._target_index(spec.target)
        if spec.protocol == "initial_state":
            return None, np.asarray([idx]), {"applied": "initial_state_only"}
        if spec.protocol != "dynamical_input":
            raise ValueError(f"Unknown perturbation protocol {spec.protocol!r}")

        def callback(x: np.ndarray, step: int) -> np.ndarray:
            if spec.onset <= step < spec.onset + spec.duration:
                return self._apply(x, idx, spec)
            return np.asarray(x, dtype=np.float64).copy()

        return callback, np.asarray([idx]), {
            "applied": "dynamical_input",
            "window": [spec.onset, spec.onset + spec.duration],
        }

    def _latent(self, states: np.ndarray) -> np.ndarray | None:
        fn = getattr(self.model, "latent", None)
        if fn is None:
            return None
        try:
            z = fn(states)
        except Exception:
            return None
        if z is None:
            return None
        arr = np.asarray(z, dtype=np.float64)
        return arr if arr.size else None

    def rollout(
        self,
        x0: np.ndarray,
        steps: int,
        *,
        seed: int = 0,
        noise_std: float = 0.0,
        perturbation: PerturbationSpec | None = None,
        include_latent: bool = False,
    ) -> Rollout:
        """Generate one free-running trajectory from ``x0``.

        Only the supplied initial state and model inputs are used.  There is no
        argument for future observations.
        """
        if steps < 1:
            raise ValueError("steps must be at least 1")
        if noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        x = np.asarray(x0, dtype=np.float64)
        if x.shape != (self.n_neurons,):
            raise ValueError(f"x0 must have shape ({self.n_neurons},), got {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError("x0 contains non-finite values")

        callback, idx, notes = self._perturbation_callback(perturbation)
        if perturbation is not None and perturbation.protocol == "initial_state":
            x = self._apply(x, int(idx[0]), perturbation)
            callback = None
        states = np.asarray(
            self.model.rollout(x, steps, seed=int(seed), noise_std=float(noise_std), perturb=callback),
            dtype=np.float64,
        )
        if states.shape != (steps + 1, self.n_neurons):
            raise RuntimeError(f"Model returned rollout shape {states.shape}")
        probs = np.asarray(self.model.behavior_proba(states[:-1]), dtype=np.float64)
        if probs.shape != (steps, 4):
            raise RuntimeError(f"Behaviour probe returned shape {probs.shape}")
        support_distance = self.support.score(states)
        latent = self._latent(states[:-1]) if include_latent else None
        metadata = {
            "support_threshold": self.support.threshold,
            "perturbation_notes": notes,
            "history": int(getattr(self.model, "history_len", 1)),
            "warning": "Model output; not an experimental intervention.",
        }
        return Rollout(
            states=states,
            behavior_proba=probs,
            support_distance=support_distance,
            seed=int(seed),
            model_kind=self.model_kind,
            noise_std=float(noise_std),
            perturbation=perturbation,
            latent=latent,
            valid=bool(np.isfinite(states).all() and np.isfinite(probs).all()),
            metadata=metadata,
        )

    def rollout_ensemble(
        self,
        x0: np.ndarray,
        n: int,
        steps: int,
        *,
        base_seed: int = 0,
        noise_std: float = 0.0,
        perturbation: PerturbationSpec | None = None,
        include_latent: bool = False,
    ) -> RolloutEnsemble:
        if n < 1:
            raise ValueError("n must be at least 1")
        seeds = tuple(int(base_seed + i) for i in range(n))
        rollouts = tuple(
            self.rollout(
                x0,
                steps,
                seed=seed,
                noise_std=noise_std,
                perturbation=perturbation,
                include_latent=include_latent,
            )
            for seed in seeds
        )
        return RolloutEnsemble(
            x0=np.asarray(x0, dtype=np.float64),
            rollouts=rollouts,
            seeds=seeds,
            model_kind=self.model_kind,
        )

    def test_starts(self, *, horizon: int = 20) -> np.ndarray:
        """Held-out test frames whose full real future stays in the test block."""
        starts = split_starts(self.data, "test", history=1, horizon=1)
        starts = starts[starts + horizon < self.data.n_frames]
        mask = self.data.masks["test"]
        keep = [int(t) for t in starts if mask[int(t) : int(t) + horizon + 1].all()]
        return np.asarray(keep, dtype=np.int64)

    def validation_starts(self, *, horizon: int = 1) -> np.ndarray:
        starts = split_starts(self.data, "val", history=1, horizon=horizon)
        mask = self.data.masks["val"]
        return np.asarray(
            [int(t) for t in starts if mask[int(t) : int(t) + horizon + 1].all()],
            dtype=np.int64,
        )


def available_predictors(data: WormData) -> tuple[str, ...]:
    return tuple(data.predictor_names)
