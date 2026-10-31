"""Fit train-only latent encoders and compare real vs simulated trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT, ExperimentConfig
from neurobranch.simulator.latent import (
    BunDLeEncoder,
    PCAEncoder,
    latent_comparison,
    linear_cka,
    pca_latent_starts,
    procrustes_error,
)
from neurobranch.simulator.train import checkpoint_root, load_datasets, load_simulator_for


def _latent_ood(encoder: PCAEncoder | BunDLeEncoder, train_X: np.ndarray, Zs: np.ndarray) -> tuple[float, float]:
    Ztr = encoder.transform(train_X)
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=1).fit(Ztr)
    d, _ = nn.kneighbors(Zs)
    train_d, _ = NearestNeighbors(n_neighbors=2).fit(Ztr).kneighbors(Ztr)
    threshold = float(np.quantile(train_d[:, 1], 0.99))
    return float(np.mean(d[:, 0] > threshold)), threshold


def evaluate_worm(
    data,
    *,
    root: Path,
    selected_kind: str,
    seed: int,
    pca: PCAEncoder,
    bundle: BunDLeEncoder | None,
    starts: int,
    horizon: int,
    noise_std: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frames = pca_latent_starts(data, horizon=horizon, n=starts)
    train_X = data.X[data.masks["train"]]
    model_kinds = [selected_kind]
    if selected_kind != "latent_knn" and _has_checkpoint(root, "latent_knn", data.worm_id, seed):
        model_kinds.append("latent_knn")
    for kind in model_kinds:
        sim = load_simulator_for(data, kind, seed, root=root)
        for t in frames:
            real = data.X[int(t) : int(t) + horizon + 1]
            rollout = sim.rollout(data.X[int(t)], horizon, seed=100 + int(t), noise_std=noise_std)
            for name, encoder in (("pca", pca), ("bundlenet", bundle)):
                if encoder is None:
                    continue
                metrics = latent_comparison(real, rollout.states, encoder)
                Zs = encoder.transform(rollout.states)
                ood, threshold = _latent_ood(encoder, train_X, Zs)
                rows.append(
                    {
                        "worm_id": int(data.worm_id),
                        "kind": kind,
                        "frame": int(t),
                        "seed": int(seed),
                        "encoder": name,
                        **metrics,
                        "latent_ood_fraction": ood,
                        "latent_ood_threshold": threshold,
                    }
                )
    # Encoder-space comparison of the latent model's own trajectory at the
    # trajectory level is already captured by encoding its decoded states.
    return rows, {"worm_id": int(data.worm_id), "n_frames": int(len(frames)), "n_models": len(model_kinds)}


def _has_checkpoint(root: Path, kind: str, worm_id: int, seed: int) -> bool:
    return (root / f"{kind}__worm{worm_id}__seed{seed}" / "model.pt").is_file()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/blocked_suite.yaml")
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports/simulator")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--starts", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--stability-seeds", nargs="+", type=int, default=[17, 23, 31])
    parser.add_argument("--skip-bundlenet", action="store_true")
    args = parser.parse_args(argv)

    cfg = ExperimentConfig.load(args.config)
    root = checkpoint_root(cfg)
    reports = args.reports.resolve()
    (reports / "latent_encoders").mkdir(parents=True, exist_ok=True)
    datasets = load_datasets(cfg)
    selection = json.loads((reports / "selection.json").read_text(encoding="utf-8"))
    selected = str(selection["selected_kind"])
    rows: list[dict[str, Any]] = []
    encoder_rows: list[dict[str, Any]] = []

    for data in datasets:
        pca = PCAEncoder.fit(data, n_components=3)
        bundle = None
        encoder_path = reports / "latent_encoders" / f"bundlenet_worm{data.worm_id}_seed{args.seed}.pt"
        encoder_sha = None
        if not args.skip_bundlenet:
            bundle = BunDLeEncoder.fit(data, epochs=args.epochs, seed=args.seed)
            bundle.save(str(encoder_path))
            encoder_sha = hashlib.sha256(encoder_path.read_bytes()).hexdigest()
        encoder_rows.append(
            {
                "worm_id": int(data.worm_id),
                "pca_explained_3d": float(pca.explained_variance.sum()),
                "pca_components": int(pca.pca.n_components_),
                "bundlenet_best_val_loss": bundle.metadata["best_val_loss"] if bundle else None,
                "bundlenet_best_epoch": bundle.metadata["best_epoch"] if bundle else None,
                "bundlenet_train_windows": bundle.metadata["train_windows"] if bundle else None,
                "bundlenet_weights": encoder_path.relative_to(PROJECT_ROOT).as_posix() if encoder_sha else None,
                "bundlenet_sha256": encoder_sha,
            }
        )
        worm_rows, _ = evaluate_worm(
            data,
            root=root,
            selected_kind=selected,
            seed=args.seed,
            pca=pca,
            bundle=bundle,
            starts=args.starts,
            horizon=args.horizon,
            noise_std=args.noise_std,
        )
        rows.extend(worm_rows)

    df = pd.DataFrame(rows)
    df.to_csv(reports / "latent_metrics.csv", index=False)
    pd.DataFrame(encoder_rows).to_csv(reports / "latent_encoders.csv", index=False)

    stability = []
    if not args.skip_bundlenet and len(args.stability_seeds) >= 2:
        data = datasets[0]
        embeddings = []
        for seed in args.stability_seeds:
            enc = BunDLeEncoder.fit(data, epochs=args.epochs, seed=seed)
            idx = np.flatnonzero(data.masks["test"])[:200]
            embeddings.append(enc.transform(data.X[idx]))
        base = embeddings[0]
        for seed, Z in zip(args.stability_seeds[1:], embeddings[1:]):
            stability.append(
                {
                    "worm_id": int(data.worm_id),
                    "seed_a": int(args.stability_seeds[0]),
                    "seed_b": int(seed),
                    "linear_cka": linear_cka(base, Z),
                    "procrustes_error": procrustes_error(base, Z),
                }
            )
    (reports / "latent_stability.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selected_model": selected,
                "stability": stability,
                "note": (
                    "BunDLe-Net is used as an encoder and visualisation. It has no published "
                    "inverse decoder, so a named-neuron perturbation is not one latent coordinate."
                ),
            },
            indent=2,
            default=float,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(df)} latent rows and {len(stability)} stability comparisons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
