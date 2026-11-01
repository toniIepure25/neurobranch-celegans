"""Model checkpointing with explicit architecture and preprocessing metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA

from neurobranch.dynamics.context_models import (
    CompositionalLatentLDS,
    SharedLatentModel,
    StableLatentLDS,
)
from neurobranch.dynamics.models import (
    BehaviorProbe,
    LatentKNNDynamics,
    LatentMLPDynamics,
    MLPDynamics,
    PCALatent,
    PersistenceModel,
    VARModel,
    _LatentNet,
)
from neurobranch.dynamics.stable_var import StableVARModel
from neurobranch.dynamics.switching import SwitchingVARModel
from neurobranch.dynamics.torch_utils import GRUNet, MLPNet


def save_model(model: Any, path: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"metadata": metadata or {}, "name": getattr(model, "name", type(model).__name__)}
    if isinstance(model, PersistenceModel):
        payload.update(kind="persistence", probe=model.probe.state_dict(), n_neurons=model.n_neurons)
    elif isinstance(model, SharedLatentModel):
        payload.update(
            kind="shared_latent",
            A=model.A,
            b=model.b,
            process_cov=model.process_cov,
            obs_cov=model.obs_cov,
            initial_cov=model.initial_cov,
            D=model.D,
            mean=model.mean,
            R=model.R,
            latent_dim=model.latent_dim,
            spectral_cap=model.spectral_cap,
            diagnostics=model.diagnostics,
            probe=model.probe.state_dict(),
            n_neurons=model.n_neurons,
        )
    elif isinstance(model, CompositionalLatentLDS):
        payload.update(
            kind="compositional_lds",
            A=model.A,
            b=model.b,
            B=model.B,
            F=model.F,
            coefficient_weights=model.coefficient_weights,
            operator_names=model.operator_names,
            coefficient_diagnostics=model.coefficient_diagnostics,
            n_operators=model.n_operators,
            process_cov=model.process_cov,
            obs_cov=model.obs_cov,
            initial_cov=model.initial_cov,
            pca_components=model.pca.components_,
            pca_mean=model.pca.mean_,
            pca_explained_variance=model.pca.explained_variance_,
            latent_dim=model.latent_dim,
            input_dim=model.input_dim,
            input_mean=model.input_mean,
            input_std=model.input_std,
            spectral_cap=model.spectral_cap,
            stability=model.stability,
            diagnostics=model.diagnostics,
            probe=model.probe.state_dict(),
            n_neurons=model.n_neurons,
        )
    elif isinstance(model, StableLatentLDS):
        payload.update(
            kind="latent_lds",
            A=model.A,
            b=model.b,
            B=model.B,
            process_cov=model.process_cov,
            obs_cov=model.obs_cov,
            initial_cov=model.initial_cov,
            pca_components=model.pca.components_,
            pca_mean=model.pca.mean_,
            pca_explained_variance=model.pca.explained_variance_,
            latent_dim=model.latent_dim,
            input_dim=model.input_dim,
            input_mean=model.input_mean,
            input_std=model.input_std,
            spectral_cap=model.spectral_cap,
            stability=model.stability,
            diagnostics=model.diagnostics,
            probe=model.probe.state_dict(),
            n_neurons=model.n_neurons,
        )
    elif isinstance(model, StableVARModel):
        payload.update(
            kind="stable_var",
            A=model.A,
            b=model.b,
            history_len=model.history_len,
            ridge=model.ridge,
            spectral_cap=model.spectral_cap,
            noise_cov=model.noise_cov,
            residuals=model.residuals,
            noise_model=model.noise_model,
            bootstrap_block=model.bootstrap_block,
            native_stochastic=model.native_stochastic,
            stabilization=model.stabilization,
            probe=model.probe.state_dict(),
            n_neurons=model.n_neurons,
        )
    elif isinstance(model, VARModel):
        payload.update(
            kind="var",
            A=model.A,
            b=model.b,
            history_len=model.history_len,
            ridge=model.ridge,
            spectral_cap=model.spectral_cap,
            noise_cov=model.noise_cov,
            residuals=model.residuals,
            noise_model=model.noise_model,
            bootstrap_block=model.bootstrap_block,
            native_stochastic=model.native_stochastic,
            probe=model.probe.state_dict(),
            n_neurons=model.n_neurons,
        )
    elif isinstance(model, SwitchingVARModel):
        payload.update(
            kind="switching_var",
            A=model.A,
            b=model.b,
            variances=model.variances,
            P=model.P,
            pi=model.pi,
            K=model.K,
            ridge=model.ridge,
            log_likelihood=model.log_likelihood,
            diagnostics=model.diagnostics,
            diagonal=model.diagonal,
            probe=model.probe.state_dict(),
            n_neurons=model.n_neurons,
        )
    elif isinstance(model, MLPDynamics):
        net = model.net
        if isinstance(net, GRUNet):
            arch = {
                "kind": "gru",
                "n_neurons": model.n_neurons,
                "hidden_dim": net.gru.hidden_size,
                "n_classes": net.behavior.out_features,
                "dropout": net.dropout.p,
            }
        else:
            arch = {
                "kind": "mlp",
                "n_neurons": model.n_neurons,
                "hidden_dim": net.encoder[0].out_features,
                "n_classes": net.behavior.out_features,
                "dropout": net.encoder[3].p,
                "residual": net.residual,
            }
        payload.update(
            kind=arch["kind"],
            arch=arch,
            state_dict=net.state_dict(),
            history_len=model.history_len,
            probe=model.probe.state_dict(),
        )
    elif isinstance(model, LatentMLPDynamics):
        enc = model.encoder
        net = model.net
        payload.update(
            kind="latent_mlp",
            arch={
                "latent_dim": int(enc.pca.n_components_),
                "hidden_dim": net.body[0].out_features,
                "n_classes": net.behavior.out_features,
                "dropout": net.body[3].p,
            },
            state_dict=net.state_dict(),
            pca_components=enc.pca.components_,
            pca_mean=enc.pca.mean_,
            pca_explained_variance=enc.pca.explained_variance_,
            z_mean=enc.z_mean,
            z_std=enc.z_std,
            probe=model.probe.state_dict(),
            residual_std=model.residual_std,
            n_neurons=model.n_neurons,
        )
    elif isinstance(model, LatentKNNDynamics):
        enc = model.encoder
        payload.update(
            kind="latent_knn",
            pca_components=enc.pca.components_,
            pca_mean=enc.pca.mean_,
            pca_explained_variance=enc.pca.explained_variance_,
            z_mean=enc.z_mean,
            z_std=enc.z_std,
            Z_train=model.Z_train,
            Z_next=model.Z_next,
            probe=model.probe.state_dict(),
            k=model.k,
            temperature=model.temperature,
            n_neurons=model.n_neurons,
            latent_dim=model.latent_dim,
        )
    else:
        raise TypeError(f"Unsupported model type {type(model)!r}")
    torch.save(payload, target)
    return target


def _restore_pca(payload: dict[str, Any]) -> PCALatent:
    pca = PCA(n_components=int(np.asarray(payload["pca_components"]).shape[0]))
    pca.components_ = np.asarray(payload["pca_components"])
    pca.mean_ = np.asarray(payload["pca_mean"])
    pca.explained_variance_ = np.asarray(payload["pca_explained_variance"])
    pca.n_features_in_ = pca.components_.shape[1]
    pca.n_components_ = pca.components_.shape[0]
    return PCALatent(
        pca=pca,
        z_mean=np.asarray(payload["z_mean"]),
        z_std=np.asarray(payload["z_std"]),
    )


def _restore_raw_pca(payload: dict[str, Any]) -> PCA:
    pca = PCA(n_components=int(np.asarray(payload["pca_components"]).shape[0]))
    pca.components_ = np.asarray(payload["pca_components"])
    pca.mean_ = np.asarray(payload["pca_mean"])
    pca.explained_variance_ = np.asarray(payload["pca_explained_variance"])
    pca.n_features_in_ = pca.components_.shape[1]
    pca.n_components_ = pca.components_.shape[0]
    pca.explained_variance_ratio_ = pca.explained_variance_ / max(
        1e-12, float(np.sum(pca.explained_variance_))
    )
    return pca


def load_model(path: str | Path) -> tuple[Any, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    kind = payload["kind"]
    metadata = payload.get("metadata", {})
    if kind == "persistence":
        return (
            PersistenceModel(
                n_neurons=int(payload["n_neurons"]),
                probe=BehaviorProbe.from_state_dict(payload["probe"]),
            ),
            metadata,
        )
    if kind == "stable_var":
        return (
            StableVARModel(
                n_neurons=int(payload["n_neurons"]),
                A=np.asarray(payload["A"]),
                b=np.asarray(payload["b"]),
                history_len=int(payload["history_len"]),
                probe=BehaviorProbe.from_state_dict(payload["probe"]),
                ridge=float(payload.get("ridge", 1e-2)),
                spectral_cap=payload.get("spectral_cap"),
                noise_cov=np.asarray(payload["noise_cov"]) if payload.get("noise_cov") is not None else None,
                residuals=np.asarray(payload["residuals"]) if payload.get("residuals") is not None else None,
                noise_model=str(payload.get("noise_model", "none")),
                bootstrap_block=int(payload.get("bootstrap_block", 5)),
                native_stochastic=bool(payload.get("native_stochastic", False)),
                stabilization=payload.get("stabilization", {}),
            ),
            metadata,
        )
    if kind == "latent_lds":
        return (
            StableLatentLDS(
                pca=_restore_raw_pca(payload),
                A=np.asarray(payload["A"]),
                b=np.asarray(payload["b"]),
                B=np.asarray(payload["B"]),
                process_cov=np.asarray(payload["process_cov"]),
                obs_cov=np.asarray(payload["obs_cov"]),
                initial_cov=np.asarray(payload["initial_cov"]),
                probe=BehaviorProbe.from_state_dict(payload["probe"]),
                n_neurons=int(payload["n_neurons"]),
                latent_dim=int(payload["latent_dim"]),
                input_dim=int(payload.get("input_dim", 0)),
                input_mean=np.asarray(payload.get("input_mean", np.zeros(0))),
                input_std=np.asarray(payload.get("input_std", np.ones(0))),
                spectral_cap=float(payload.get("spectral_cap", 0.995)),
                stability=payload.get("stability", {}),
                diagnostics=payload.get("diagnostics", {}),
            ),
            metadata,
        )
    if kind == "compositional_lds":
        return (
            CompositionalLatentLDS(
                pca=_restore_raw_pca(payload),
                A=np.asarray(payload["A"]),
                b=np.asarray(payload["b"]),
                B=np.asarray(payload["B"]),
                process_cov=np.asarray(payload["process_cov"]),
                obs_cov=np.asarray(payload["obs_cov"]),
                initial_cov=np.asarray(payload["initial_cov"]),
                probe=BehaviorProbe.from_state_dict(payload["probe"]),
                n_neurons=int(payload["n_neurons"]),
                latent_dim=int(payload["latent_dim"]),
                input_dim=int(payload.get("input_dim", 0)),
                input_mean=np.asarray(payload.get("input_mean", np.zeros(0))),
                input_std=np.asarray(payload.get("input_std", np.ones(0))),
                spectral_cap=float(payload.get("spectral_cap", 0.995)),
                stability=payload.get("stability", {}),
                diagnostics=payload.get("diagnostics", {}),
                F=np.asarray(payload["F"]),
                coefficient_weights=np.asarray(payload["coefficient_weights"]),
                operator_names=tuple(payload.get("operator_names", ())),
                coefficient_diagnostics=payload.get("coefficient_diagnostics", {}),
                n_operators=int(payload.get("n_operators", 0)),
            ),
            metadata,
        )
    if kind == "shared_latent":
        return (
            SharedLatentModel.from_shared(
                A=np.asarray(payload["A"]),
                b=np.asarray(payload["b"]),
                process_cov=np.asarray(payload["process_cov"]),
                D=np.asarray(payload["D"]),
                mean=np.asarray(payload["mean"]),
                R=np.asarray(payload["R"]),
                probe=BehaviorProbe.from_state_dict(payload["probe"]),
                n_neurons=int(payload["n_neurons"]),
                latent_dim=int(payload["latent_dim"]),
                obs_cov=np.asarray(payload["obs_cov"]),
                initial_cov=np.asarray(payload["initial_cov"]),
                spectral_cap=float(payload.get("spectral_cap", 0.995)),
                metadata=payload.get("diagnostics", {}),
            ),
            metadata,
        )
    if kind == "var":
        return (
            VARModel(
                n_neurons=int(payload["n_neurons"]),
                A=np.asarray(payload["A"]),
                b=np.asarray(payload["b"]),
                history_len=int(payload["history_len"]),
                probe=BehaviorProbe.from_state_dict(payload["probe"]),
                ridge=float(payload.get("ridge", 1e-2)),
                spectral_cap=payload.get("spectral_cap"),
                noise_cov=np.asarray(payload["noise_cov"]) if payload.get("noise_cov") is not None else None,
                residuals=np.asarray(payload["residuals"]) if payload.get("residuals") is not None else None,
                noise_model=str(payload.get("noise_model", "none")),
                bootstrap_block=int(payload.get("bootstrap_block", 5)),
                native_stochastic=bool(payload.get("native_stochastic", False)),
            ),
            metadata,
        )
    if kind == "switching_var":
        return (
            SwitchingVARModel(
                n_neurons=int(payload["n_neurons"]),
                K=int(payload["K"]),
                A=np.asarray(payload["A"]),
                b=np.asarray(payload["b"]),
                variances=np.asarray(payload["variances"]),
                P=np.asarray(payload["P"]),
                pi=np.asarray(payload["pi"]),
                probe=BehaviorProbe.from_state_dict(payload["probe"]),
                ridge=float(payload.get("ridge", 1e-2)),
                log_likelihood=float(payload.get("log_likelihood", float("-inf"))),
                diagnostics=payload.get("diagnostics", {}),
                diagonal=bool(payload.get("diagonal", True)),
            ),
            metadata,
        )
    if kind in {"mlp", "gru"}:
        arch = payload["arch"]
        if kind == "gru":
            net = GRUNet(
                int(arch["n_neurons"]),
                int(arch["hidden_dim"]),
                int(arch["n_classes"]),
                dropout=float(arch["dropout"]),
            )
        else:
            net = MLPNet(
                int(arch["n_neurons"]) * int(payload["history_len"]),
                int(arch["hidden_dim"]),
                int(arch["n_neurons"]),
                int(arch["n_classes"]),
                dropout=float(arch["dropout"]),
                residual=bool(arch["residual"]),
            )
        net.load_state_dict(payload["state_dict"])
        net.eval()
        return (
            MLPDynamics(
                net=net,
                n_neurons=int(arch["n_neurons"]),
                history_len=int(payload["history_len"]),
                probe=BehaviorProbe.from_state_dict(payload["probe"]),
                device="cpu",
                name=kind,
            ),
            metadata,
        )
    if kind == "latent_mlp":
        arch = payload["arch"]
        enc = _restore_pca(payload)
        net = _LatentNet(
            int(arch["latent_dim"]),
            int(arch["hidden_dim"]),
            int(arch["n_classes"]),
            float(arch["dropout"]),
        )
        net.load_state_dict(payload["state_dict"])
        net.eval()
        return (
            LatentMLPDynamics(
                net=net,
                encoder=enc,
                n_neurons=int(payload["n_neurons"]),
                latent_dim=int(arch["latent_dim"]),
                probe=BehaviorProbe.from_state_dict(payload["probe"]),
                residual_std=np.asarray(payload["residual_std"]),
                device="cpu",
            ),
            metadata,
        )
    if kind == "latent_knn":
        return (
            LatentKNNDynamics(
                encoder=_restore_pca(payload),
                Z_train=np.asarray(payload["Z_train"]),
                Z_next=np.asarray(payload["Z_next"]),
                probe=BehaviorProbe.from_state_dict(payload["probe"]),
                n_neurons=int(payload["n_neurons"]),
                latent_dim=int(payload["latent_dim"]),
                k=int(payload["k"]),
                temperature=float(payload["temperature"]),
            ),
            metadata,
        )
    raise ValueError(f"Unknown checkpoint kind {kind!r}")
