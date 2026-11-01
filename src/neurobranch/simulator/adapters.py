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

    The default threshold is the 99th percentile of the leave-one-out training
    distance (second nearest neighbour because the first is the point itself).
    If ``calibration_X`` is supplied, the threshold is instead the requested
    quantile of held-out (validation) distances.  The calibrated form is the
    Phase 6 default because a training-only envelope has no external check.
    """

    def __init__(
        self,
        train_X: np.ndarray,
        *,
        quantile: float = 0.99,
        calibration_X: np.ndarray | None = None,
        audit_quantiles: tuple[float, ...] = (0.90, 0.95, 0.99),
    ):
        X = np.asarray(train_X, dtype=np.float64)
        if X.ndim != 2 or len(X) < 3:
            raise ValueError("SupportModel needs at least three training states")
        self.nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(X)
        dist, _ = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(X).kneighbors(X)
        # dist[:, 0] is self; the second neighbour estimates local support.
        self.training_threshold = float(np.quantile(dist[:, 1], quantile))
        self.calibration_quantile = float(quantile) if calibration_X is not None else None
        self.calibration_scores: np.ndarray | None = None
        if calibration_X is None:
            self.threshold = self.training_threshold
            scores = dist[:, 1]
        else:
            cal = np.asarray(calibration_X, dtype=np.float64)
            if cal.ndim != 2 or cal.shape[1] != X.shape[1] or len(cal) == 0:
                raise ValueError("calibration_X must be a non-empty (n, n_neurons) array")
            scores = self.score(cal)
            self.calibration_scores = np.asarray(scores, dtype=np.float64)
            self.threshold = float(np.quantile(scores, quantile))
        self.quantile = float(quantile)
        if any(not (0 < q < 1) for q in audit_quantiles):
            raise ValueError("audit quantiles must lie strictly between 0 and 1")
        self.audit_quantiles = tuple(float(q) for q in audit_quantiles)
        self.thresholds = {
            f"q{int(round(q * 100)):02d}": float(np.quantile(scores, q))
            for q in self.audit_quantiles
        }

    def threshold_at(self, quantile: float) -> float:
        """Audit threshold at a requested calibration quantile."""
        if self.calibration_scores is None:
            raise RuntimeError("no calibration scores; threshold_at is unavailable")
        return float(np.quantile(self.calibration_scores, float(quantile)))

    def support_table(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Raw support distances and every audited Boolean flag.

        Raw distances are retained so a q95 conclusion can be recomputed without
        re-running the simulator.
        """
        d = self.score(X)
        return {
            "distance": d,
            **{name: d <= value for name, value in self.thresholds.items()},
        }

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
        support_calibration_X: np.ndarray | None = None,
        support_quantile: float = 0.99,
    ):
        self.model = model
        self.data = data
        self.checkpoint = str(checkpoint) if checkpoint is not None else None
        self.metadata = dict(metadata or {})
        self.empirical_evidence = empirical_evidence
        calibration = (
            np.asarray(support_calibration_X, dtype=np.float64)
            if support_calibration_X is not None
            else data.X[data.masks["val"]]
        )
        self.support = SupportModel(
            data.X[data.masks["train"]],
            quantile=support_quantile,
            calibration_X=calibration,
        )
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
        support_calibration_X: np.ndarray | None = None,
        support_quantile: float = 0.99,
    ) -> NeuralSimulator:
        model, metadata = load_model(checkpoint)
        return cls(
            model,
            data,
            checkpoint=checkpoint,
            metadata=metadata,
            empirical_evidence=empirical_evidence,
            support_calibration_X=support_calibration_X,
            support_quantile=support_quantile,
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
        x0: np.ndarray | None,
        steps: int,
        *,
        initial_history: np.ndarray | None = None,
        seed: int = 0,
        noise_std: float = 0.0,
        perturbation: PerturbationSpec | None = None,
        include_latent: bool = False,
        stochastic: bool | None = None,
        external_input: np.ndarray | None = None,
    ) -> Rollout:
        """Generate one free-running trajectory.

        Supply exactly one of ``x0`` or ``initial_history``.  The latter must be
        the real observed window for models with more than one input frame; a
        single frame is padded only for the backwards-compatible ``x0`` path,
        and that fact is recorded in the rollout metadata.

        Only the supplied history and model inputs are used.  There is no
        argument for future observations.  ``external_input`` is a *known
        schedule* of length ``steps`` (for example measured oxygen); it is not a
        future neural observation.
        """
        if steps < 1:
            raise ValueError("steps must be at least 1")
        if noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        input_array: np.ndarray | None = None
        if external_input is not None:
            if not bool(getattr(self.model, "supports_input", False)):
                raise ValueError("this model does not accept an external input")
            input_array = np.asarray(external_input, dtype=np.float64)
            if input_array.ndim == 1:
                input_array = input_array[:, None]
            if input_array.shape[0] != steps:
                raise ValueError(
                    f"external_input must have {steps} rows, got {input_array.shape[0]}"
                )
            if not np.isfinite(input_array).all():
                raise ValueError("external_input contains non-finite values")
            expected = int(getattr(self.model, "input_dim", 0))
            if expected and input_array.shape[1] != expected:
                raise ValueError(
                    f"external_input has {input_array.shape[1]} channels, model expects {expected}"
                )
        history_len = int(getattr(self.model, "history_len", 1))
        if (x0 is None) == (initial_history is None):
            raise ValueError("supply exactly one of x0 or initial_history")
        if initial_history is None:
            arr = np.asarray(x0, dtype=np.float64)
            if arr.shape != (self.n_neurons,):
                raise ValueError(f"x0 must have shape ({self.n_neurons},), got {arr.shape}")
            hist = np.repeat(arr[None], history_len, axis=0)
            padded = history_len > 1
        else:
            hist = np.asarray(initial_history, dtype=np.float64)
            if hist.ndim != 2 or hist.shape[1] != self.n_neurons:
                raise ValueError(
                    f"initial_history must be (history, {self.n_neurons}), got {hist.shape}"
                )
            if len(hist) != history_len:
                raise ValueError(
                    f"model requires {history_len} history frames, got {len(hist)}"
                )
            padded = False
        if not np.isfinite(hist).all():
            raise ValueError("initial history contains non-finite values")
        x = hist[-1].copy()

        callback, idx, notes = self._perturbation_callback(perturbation)
        if perturbation is not None and perturbation.protocol == "initial_state":
            hist = hist.copy()
            hist[-1] = self._apply(hist[-1], int(idx[0]), perturbation)
            x = hist[-1].copy()
            callback = None
        use_stochastic = (
            bool(getattr(self.model, "native_stochastic", False))
            if stochastic is None
            else bool(stochastic)
        )
        rollout_kwargs = {
            "seed": int(seed),
            "noise_std": float(noise_std),
            "perturb": callback,
            "initial_history": hist,
            "stochastic": use_stochastic,
        }
        if input_array is not None:
            rollout_kwargs["inputs"] = input_array
        states = np.asarray(
            self.model.rollout(x, steps, **rollout_kwargs),
            dtype=np.float64,
        )
        if states.shape != (steps + 1, self.n_neurons):
            raise RuntimeError(f"Model returned rollout shape {states.shape}")
        probs = np.asarray(self.model.behavior_proba(states[:-1]), dtype=np.float64)
        if probs.shape != (steps, 4):
            raise RuntimeError(f"Behaviour probe returned shape {probs.shape}")
        support_distance = self.support.score(states)
        latent = self._latent(states[:-1]) if include_latent else None
        threshold = self.support.threshold
        readout_supported = float(np.mean(support_distance[:-1] <= threshold))
        metadata = {
            "support_threshold": threshold,
            "training_support_threshold": self.support.training_threshold,
            "support_calibration_quantile": self.support.calibration_quantile,
            "perturbation_notes": notes,
            "history": history_len,
            "initial_history_padded": bool(padded),
            "readout_supported_fraction": readout_supported,
            "external_input_supplied": bool(input_array is not None),
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
            initial_history=hist,
            native_stochastic=bool(getattr(self.model, "native_stochastic", False)),
            metadata=metadata,
        )

    def rollout_ensemble(
        self,
        x0: np.ndarray | None,
        n: int,
        steps: int,
        *,
        initial_history: np.ndarray | None = None,
        initial_macro: int | None = None,
        base_seed: int = 0,
        noise_std: float = 0.0,
        perturbation: PerturbationSpec | None = None,
        include_latent: bool = False,
        stochastic: bool | None = None,
        external_input: np.ndarray | None = None,
    ) -> RolloutEnsemble:
        if n < 1:
            raise ValueError("n must be at least 1")
        seeds = tuple(int(base_seed + i) for i in range(n))
        rollouts = tuple(
            self.rollout(
                x0,
                steps,
                initial_history=initial_history,
                seed=seed,
                noise_std=noise_std,
                perturbation=perturbation,
                include_latent=include_latent,
                stochastic=stochastic,
                external_input=external_input,
            )
            for seed in seeds
        )
        reference = rollouts[0].states[0]
        return RolloutEnsemble(
            x0=np.asarray(reference, dtype=np.float64),
            rollouts=rollouts,
            seeds=seeds,
            model_kind=self.model_kind,
            initial_macro=initial_macro,
        )

    def test_starts(self, *, horizon: int = 20, history: int | None = None) -> np.ndarray:
        """Held-out test frames whose full past and future stay in the block."""
        h = int(history or getattr(self.model, "history_len", 1))
        starts = split_starts(self.data, "test", history=h, horizon=horizon)
        mask = self.data.masks["test"]
        return np.asarray(
            [int(t) for t in starts if mask[int(t) - h + 1 : int(t) + horizon + 1].all()],
            dtype=np.int64,
        )

    def validation_starts(self, *, horizon: int = 1, history: int | None = None) -> np.ndarray:
        h = int(history or getattr(self.model, "history_len", 1))
        starts = split_starts(self.data, "val", history=h, horizon=horizon)
        mask = self.data.masks["val"]
        return np.asarray(
            [int(t) for t in starts if mask[int(t) - h + 1 : int(t) + horizon + 1].all()],
            dtype=np.int64,
        )

    def initial_history(self, frame: int, history: int | None = None) -> np.ndarray:
        """Real observed history ending at ``frame``; never padded."""
        h = int(history or getattr(self.model, "history_len", 1))
        frame = int(frame)
        if frame - h + 1 < 0:
            raise ValueError(f"frame {frame} has no complete {h}-frame history")
        return self.data.X[frame - h + 1 : frame + 1].copy()

    def input_slice(self, frame: int, steps: int) -> np.ndarray | None:
        """Known external schedule for frames ``frame+1 .. frame+steps``.

        This is a schedule, not a neural observation.  Returns ``None`` when the
        data have no attached external input.
        """
        values = getattr(self.data, "external_input", None)
        if values is None:
            return None
        start = int(frame) + 1
        stop = start + int(steps)
        if start < 0 or stop > len(values):
            raise ValueError("external schedule does not cover the requested rollout")
        return np.asarray(values[start:stop], dtype=np.float64)


def available_predictors(data: WormData) -> tuple[str, ...]:
    return tuple(data.predictor_names)
