"""Unified facade for the three Vienna workbench tiers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dataset import BEHAVIOR_NAMES, WormTrajectory, load_trajectories
from .dynamics import FittedModels
from .manifold_walk import ManifoldWalk
from .ncmcm import NCMCMWorm, load_ncmcm
from .persistence import load_models
from .premotor import PremotorModel, PremotorParameters, load_premotor, simulate
from .switching import SwitchingFit, fit_switching


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class ViennaWorkbench:
    """Lazy-loading facade shared by the UI and notebooks."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else repo_root()
        self._trajectories: list[WormTrajectory] | None = None
        self._models: FittedModels | None = None
        self._switching: SwitchingFit | None = None
        self._walk: ManifoldWalk | None = None
        self._ncmcm: dict[int, NCMCMWorm] = {}
        self._premotor: PremotorModel | None = None

    @property
    def trajectories(self) -> list[WormTrajectory]:
        if self._trajectories is None:
            self._trajectories = load_trajectories(self.root / "data" / "processed" / "bundlenet_sim")
        return self._trajectories

    @property
    def models(self) -> FittedModels:
        if self._models is None:
            path = self.root / "artifacts" / "bundlenet_sim" / "model.npz"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} missing; run `python -m neurobranch.bundlenet_sim.train` first"
                )
            self._models = load_models(path)
        return self._models

    @property
    def switching(self) -> SwitchingFit:
        if self._switching is None:
            self._switching = fit_switching(self.trajectories)
        return self._switching

    @property
    def manifold_walk(self) -> ManifoldWalk:
        if self._walk is None:
            self._walk = ManifoldWalk().fit(self.trajectories)
        return self._walk

    def ncmcm(self, worm_id: int) -> NCMCMWorm:
        if worm_id not in self._ncmcm:
            self._ncmcm[worm_id] = load_ncmcm(worm_id, self.root / "data" / "processed" / "ncmcm")
        return self._ncmcm[worm_id]

    @property
    def premotor(self) -> PremotorModel:
        if self._premotor is None:
            self._premotor = load_premotor(self.root / "data" / "processed" / "premotor")
        return self._premotor

    def tier1_simulate(
        self,
        worm_id: int,
        start: int,
        horizon: int,
        mode: str = "manifold_walk",
        seed: int = 0,
        temperature: float = 1.0,
        target_behavior: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        traj = next(t for t in self.trajectories if t.worm_id == worm_id)
        y0 = traj.latent[start]
        b0 = int(traj.behavior[start])
        if mode == "manifold_walk":
            sim = self.manifold_walk.rollout(
                y0,
                n_steps=horizon,
                seed=seed,
                target_behavior=target_behavior,
                temperature=temperature,
            )
        elif mode == "switching":
            sim, _, _ = self.switching.simulate(y0, b0, n_steps=horizon, seed=seed)
        elif mode == "quadratic":
            sim = self.models.dynamics.rollout(y0, n_steps=horizon, seed=seed)
        else:
            raise ValueError(f"unknown mode: {mode}")
        proba = self.models.behavior.predict_proba(sim)
        return sim, proba

    def tier3_simulate(
        self,
        params: PremotorParameters,
        u0: np.ndarray | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return simulate(self.premotor, params, u0=u0, seed=seed)

    def metrics(self) -> dict[str, object]:
        path = self.root / "artifacts" / "bundlenet_sim" / "metrics.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def tier1_benchmark(self) -> dict[str, object]:
        path = self.root / "reports" / "vienna_workbench" / "tier1_benchmark.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def provenance(self) -> dict[str, object]:
        return {
            "tier1": {
                "model": "BunDLe-Net consistent embeddings",
                "source": "NC-MCM precomputed_results/bundlenet_consistent_embedding_worm_*.npz",
                "doi": "10.1101/2023.08.08.551978",
                "claim_level": "latent manifold simulator; not mechanistic",
            },
            "tier2": {
                "model": "NC-MCM precomputed models",
                "source": "NC-MCM precomputed_results/nc_mcm_model_worm_*.npz",
                "doi": "10.1371/journal.pcbi.1012674",
                "claim_level": "descriptive causal-state model; interventions not identified",
            },
            "tier3": {
                "model": "Celegans_premotor fitted synaptic matrix",
                "source": "Celegans_premotor data/model_weights/avg_model_weights.jld2",
                "doi": "10.1371/journal.pcbi.1013818",
                "claim_level": "biophysical sandbox; gap-junction matrix and neuron order absent from export",
            },
            "behaviors": list(BEHAVIOR_NAMES),
        }
