"""NeuroBranch command line interface."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from neurobranch import EPISTEMIC_WARNING, __version__
from neurobranch.config import PROJECT_ROOT, ExperimentConfig
from neurobranch.data.kato import load_kato, write_audit

console = Console()
app = typer.Typer(no_args_is_help=True, help="NeuroBranch: C. elegans population dynamics and virtual perturbation.")
data_app = typer.Typer(no_args_is_help=True, help="Dataset audit and integrity checks.")
reproduce_app = typer.Typer(no_args_is_help=True, help="Reference artifact verification.")
train_app = typer.Typer(no_args_is_help=True, help="Model training.")
simulate_app = typer.Typer(no_args_is_help=True, help="Baseline and virtual-perturbation simulation.")
branching_app = typer.Typer(no_args_is_help=True, help="Branching analysis.")
benchmark_app = typer.Typer(no_args_is_help=True, help="Reliability ceilings, design resolution and claims.")
app.add_typer(data_app, name="data")
app.add_typer(reproduce_app, name="reproduce")
app.add_typer(train_app, name="train")
app.add_typer(simulate_app, name="simulate")
app.add_typer(branching_app, name="branching")
app.add_typer(benchmark_app, name="benchmark")


def _load_data(config: Path):
    from neurobranch.dynamics.training import load_worm_data

    cfg = ExperimentConfig.load(config)
    return cfg, load_worm_data(cfg)


def _model_path(worm: int, model: str, seed: int, run_dir: Path) -> Path:
    return run_dir / f"{model}__worm{worm}__seed{seed}" / "model.pt"


@benchmark_app.command("reliability")
def benchmark_reliability(
    trials: Annotated[Path, typer.Option(help="Trial table with dataset_id/target/neuron/response_mean")] = PROJECT_ROOT
    / "reports"
    / "phase3"
    / "randi_trial_table_phase3.csv.gz",
    out: Annotated[Path, typer.Option(help="Output directory")] = PROJECT_ROOT / "reports" / "phase9",
) -> None:
    """Estimate repeat and across-animal reliability ceilings."""
    import pandas as pd

    from neurobranch.benchmark.reporting import write_reliability_report

    if not trials.exists():
        raise typer.BadParameter(f"trial table not found: {trials}")
    summary = write_reliability_report(pd.read_csv(trials), out, input_path=trials)
    t = Table(title="Reliability ceilings")
    t.add_column("facet")
    t.add_column("mean", justify="right")
    t.add_column("95% CI", justify="right")
    t.add_row(
        "repeat (within animal)",
        f"{summary['mean_repeat_reliability']:.3f}",
        f"[{summary['repeat_ci_low']:.3f}, {summary['repeat_ci_high']:.3f}]",
    )
    t.add_row(
        "single animal (held out)",
        f"{summary['mean_animal_reliability']:.3f}",
        f"[{summary['animal_ci_low']:.3f}, {summary['animal_ci_high']:.3f}]",
    )
    console.print(t)
    console.print(f"wrote {out}")


@benchmark_app.command("claims")
def benchmark_claims(
    summary: Annotated[Path, typer.Option(help="reliability_summary.json")] = PROJECT_ROOT
    / "reports"
    / "phase9"
    / "reliability_summary.json",
    out: Annotated[Path, typer.Option(help="Output claims ledger")] = PROJECT_ROOT / "reports" / "phase9" / "claims.json",
) -> None:
    """Build the machine-readable claims ledger."""
    from neurobranch.benchmark.claims import build_claims, write_claims

    if not summary.exists():
        raise typer.BadParameter(f"summary not found: {summary}")
    payload = json.loads(summary.read_text(encoding="utf-8"))
    claims = build_claims(payload)
    write_claims(out, claims)
    t = Table(title="Claims ledger")
    t.add_column("claim")
    t.add_column("status")
    for claim in claims:
        t.add_row(claim.id, claim.status)
    console.print(t)
    console.print(f"wrote {out}")


@app.command()
def version() -> None:
    """Print version and the standing epistemic warning."""
    console.print(f"neurobranch {__version__}")
    console.print(EPISTEMIC_WARNING, style="bold yellow")


@data_app.command("audit")
def data_audit(
    out: Annotated[Path, typer.Option(help="Output report directory")] = PROJECT_ROOT / "reports",
) -> None:
    """Verify the Kato 2015 archive and write audit tables."""
    ds = load_kato()
    paths = write_audit(ds, out)
    t = Table(title="Kato 2015 audit")
    t.add_column("field")
    t.add_column("value")
    t.add_row("worms", str(len(ds)))
    t.add_row("frames", str(sum(r.n_frames for r in ds)))
    t.add_row("named cells (total)", str(sum(r.identified_mask.sum() for r in ds)))
    t.add_row("sha256", ds.source_sha256)
    console.print(t)
    for k, v in paths.items():
        console.print(f"{k}: {v}")


@data_app.command("list")
def data_list() -> None:
    """List recordings, frame counts, and identified cell counts."""
    for r in load_kato():
        console.print(
            f"worm {r.worm_id}: {r.n_frames} frames, {r.n_frames / r.fps:.1f}s, "
            f"{r.n_neurons} segmented, {int(r.identified_mask.sum())} named"
        )


@reproduce_app.command("ncmcm")
def reproduce_ncmcm(
    k: Annotated[int, typer.Option(help="Number of cognitive states")] = 3,
    n_surrogates: Annotated[int, typer.Option(min=100, help="Lag-2 surrogate chains")] = 1000,
    out: Annotated[Path, typer.Option(help="JSON output")] = PROJECT_ROOT / "artifacts" / "reproduction_ncmcm.json",
) -> None:
    """Verify distributed NC-MCM artifacts and re-run the Markov surrogate test."""
    import csv

    from neurobranch.reproduction.ncmcm import summarize_worm

    rows = []
    for w in range(1, 6):
        s = summarize_worm(w, selected_k=k, n_markov_sims=n_surrogates)
        if s.get("available"):
            rows.append(s)
            console.print(
                f"worm {w}: p(pre)={s['markov_p_precomputed']:.3f} "
                f"p(reimpl)={s['markov_p_reimplemented']:.3f} "
                f"decoder_acc={s['decoder_cv']['accuracy']:.3f}"
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, default=float), encoding="utf-8")
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "worm_id",
                "selected_k",
                "markov_p_precomputed",
                "markov_p_reimplemented",
                "markov_p_abs_diff",
                "decoder_accuracy",
                "decoder_balanced_accuracy",
                "n_frames",
            ],
        )
        writer.writeheader()
        for s in rows:
            cv = s.get("decoder_cv") or {}
            writer.writerow(
                {
                    "worm_id": s.get("worm_id"),
                    "selected_k": s.get("selected_k"),
                    "markov_p_precomputed": s.get("markov_p_precomputed"),
                    "markov_p_reimplemented": s.get("markov_p_reimplemented"),
                    "markov_p_abs_diff": s.get("markov_p_abs_diff"),
                    "decoder_accuracy": cv.get("accuracy"),
                    "decoder_balanced_accuracy": cv.get("balanced_accuracy"),
                    "n_frames": s.get("n_frames"),
                }
            )
    console.print(f"wrote {out} and {csv_path}")


@reproduce_app.command("bundlenet")
def reproduce_bundlenet(
    worm: Annotated[int, typer.Option()] = 0,
    epochs: Annotated[int, typer.Option()] = 120,
    out: Annotated[
        Path, typer.Option(help="JSON output")
    ] = PROJECT_ROOT / "artifacts" / "reproduction_bundlenet_lite.json",
) -> None:
    """Train the PyTorch BunDLe-Net reimplementation and report honest holdout metrics."""
    from neurobranch.representation.bundlenet import train_bundlenet_lite

    r = train_bundlenet_lite(worm, epochs=epochs)
    payload = {"worm_id": worm, "metrics": r.metrics, "history": r.history, "metadata": r.metadata}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    console.print(json.dumps(r.metrics, indent=2))
    console.print("Deviations from upstream:", style="bold")
    for d in r.metadata["deviations"]:
        console.print(f" - {d}")
    console.print(f"wrote {out}")


@train_app.command("dynamics")
def train_dynamics(
    config: Annotated[Path, typer.Option(exists=True)] = PROJECT_ROOT / "configs" / "experiments" / "blocked_suite.yaml",
    models: Annotated[str, typer.Option(help="Comma-separated model kinds")] = "persistence,var,mlp,mlp_residual,latent_mlp,latent_knn,gru",
    seeds: Annotated[str, typer.Option(help="Comma-separated seeds")] = "17,23,31",
    worms: Annotated[str, typer.Option(help="Comma-separated worm ids")] = "0,1,2,3,4",
    out: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Fit dynamics models with blocked, purged splits."""
    from neurobranch.dynamics.training import run_suite

    cfg = ExperimentConfig.load(config)
    df = run_suite(
        [m.strip() for m in models.split(",") if m.strip()],
        cfg=cfg,
        seeds=[int(s) for s in seeds.split(",")],
        worm_ids=[int(w) for w in worms.split(",")],
        output_dir=out or cfg.output_dir,
    )
    console.print(df.groupby("model")[["one_step_r2", "behavior_accuracy", "transition_total_variation"]].mean().round(3))


@app.command("evaluate")
def evaluate(
    run_dir: Annotated[Path, typer.Option(exists=True)] = PROJECT_ROOT / "artifacts" / "runs" / "blocked_suite",
) -> None:
    """Summarise a completed run directory."""
    import pandas as pd

    path = run_dir / "suite_metrics.csv"
    if not path.exists():
        raise typer.BadParameter(f"No suite_metrics.csv in {run_dir}")
    d = pd.read_csv(path)
    console.print(
        d.groupby("model")[
            ["one_step_r2", "behavior_accuracy", "behavior_persistence_accuracy", "transition_total_variation"]
        ]
        .agg(["mean", "std"])
        .round(3)
        .to_string()
    )


@branching_app.command("analyze")
def branching_analyze(
    worm: Annotated[int, typer.Option()] = 0,
    config: Annotated[Path, typer.Option(exists=True)] = PROJECT_ROOT / "configs" / "experiments" / "blocked_suite.yaml",
    out: Annotated[Path, typer.Option()] = PROJECT_ROOT / "artifacts" / "branching_worm0.json",
) -> None:
    """Compute empirical local outcome diversity in a 3-D PCA projection."""
    from sklearn.decomposition import PCA

    from neurobranch.branching import (
        candidate_frames,
        corrected_branching_test,
        entropy_across_latent_dims,
        local_outcome_diversity,
        permutation_branching_test,
        summarize_branching,
    )
    from neurobranch.preprocessing import split_starts

    cfg, datasets = _load_data(config)
    data = datasets[worm]
    tr = split_starts(data, "train", history=1, horizon=1)
    pca = PCA(3, random_state=0).fit(data.X[tr])
    Z = pca.transform(data.X)
    cand = candidate_frames(data, horizon=20)
    res = local_outcome_diversity(Z, data, cand, k=40, horizon=20)
    payload = {
        "worm_id": worm,
        "summary": summarize_branching(res, top_n=20),
        "permutation": permutation_branching_test(Z, data, cand, k=40, horizon=20, n_permutations=100),
        "corrected": corrected_branching_test(
            Z, data, cand, k=40, horizon=20, n_permutations=500, seed=0
        ),
        "latent_dim_stability": entropy_across_latent_dims(data)["spearman_vs_base"],
        "caveat": "Local outcome diversity in a projection is not a bifurcation.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    console.print(json.dumps(payload["summary"], indent=2)[:2000])
    console.print(f"wrote {out}")


@simulate_app.command("baseline")
def simulate_baseline(
    worm: Annotated[int, typer.Option()] = 0,
    model_name: Annotated[str, typer.Option("--model")] = "mlp",
    seed: Annotated[int, typer.Option()] = 17,
    horizon: Annotated[int, typer.Option()] = 15,
    out: Annotated[Path, typer.Option()] = PROJECT_ROOT / "artifacts" / "baseline_rollout.npz",
) -> None:
    """Roll out a fitted model from held-out test states."""
    import numpy as np

    from neurobranch.dynamics.checkpointing import load_model
    from neurobranch.preprocessing import split_starts

    cfg, datasets = _load_data(PROJECT_ROOT / "configs" / "experiments" / "blocked_suite.yaml")
    data = datasets[worm]
    model, _ = load_model(_model_path(worm, model_name, seed, Path(cfg.output_dir)))
    frames = split_starts(data, "test", history=1, horizon=1)[:20]
    sims = np.stack([model.rollout(data.X[t], horizon, seed=seed, noise_std=0.04) for t in frames])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, simulations=sims, initial_frames=frames)
    console.print(f"wrote {out} with {sims.shape[0]} trajectories of length {sims.shape[1]}")
    console.print(EPISTEMIC_WARNING, style="yellow")


@simulate_app.command("intervention")
def simulate_intervention(
    worm: Annotated[int, typer.Option()] = 0,
    model_name: Annotated[str, typer.Option("--model")] = "mlp",
    seed: Annotated[int, typer.Option()] = 17,
    spec_file: Annotated[Path, typer.Option(exists=True)] = PROJECT_ROOT / "configs" / "interventions" / "rivl_plus2.yaml",
    out: Annotated[Path, typer.Option()] = PROJECT_ROOT / "artifacts" / "intervention.json",
) -> None:
    """Run a paired baseline/intervention counterfactual. Observational, not causal."""
    from neurobranch.dynamics.checkpointing import load_model
    from neurobranch.interventions import InterventionSpec, run_counterfactual
    from neurobranch.interventions.spec import make_perturbation

    cfg, datasets = _load_data(PROJECT_ROOT / "configs" / "experiments" / "blocked_suite.yaml")
    data = datasets[worm]
    model, _ = load_model(_model_path(worm, model_name, seed, Path(cfg.output_dir)))
    import yaml

    payload = yaml.safe_load(spec_file.read_text(encoding="utf-8"))
    spec = InterventionSpec.from_dict(payload)
    make_perturbation(data, spec)  # fail early with a clear error for excluded neurons
    res = run_counterfactual(model, data, spec, seed=seed, noise_std=0.04)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res.summary(), indent=2, default=float), encoding="utf-8")
    console.print(json.dumps({k: v for k, v in res.summary().items() if k != "intervention"}, indent=2)[:2500])
    console.print(EPISTEMIC_WARNING, style="bold yellow")
    console.print(f"wrote {out}")


@app.command("report")
def report() -> None:
    """Write the results summary from existing artifacts."""
    from neurobranch.reporting import write_results_report

    paths = write_results_report(PROJECT_ROOT / "reports")
    for p in paths:
        console.print(str(p))


@app.command("app")
def launch_app(
    port: Annotated[int, typer.Option()] = 8501,
) -> None:
    """Launch the Streamlit explorer."""
    app_path = PROJECT_ROOT / "app" / "streamlit_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
    console.print(f"Launching {cmd}")
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    app()
