"""Compare the Python clamped core with the authors' Julia reference run.

The Julia CSV is produced by ``scripts/reference/run_julia_reference.jl``.
This script does not re-run Julia; it recomputes the Python side, aligns traces
by neuron name, and writes a machine-readable comparison plus input hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from neurobranch.bundlenet_sim.clamped import ClampedParameters, simulate_clamped
from neurobranch.bundlenet_sim.premotor import load_premotor
from neurobranch.bundlenet_sim.recording import (
    INTRINSIC_NEURONS,
    build_clamped_recording,
    load_atanas_recording,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REF = ROOT / "references" / "external" / "Celegans_premotor"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict[str, object]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        # Generated reports are expected to change after the code is committed;
        # only the code/docs/tests that produced them define dirty state.
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--", "src", "scripts", "tests"],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"git_sha": sha, "dirty": dirty, "dirty_scope": "src scripts tests"}
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": None, "dirty": None, "dirty_scope": "src scripts tests"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", default="2023-01-23-15")
    parser.add_argument(
        "--julia-csv",
        type=Path,
        default=ROOT
        / "reports"
        / "vienna_workbench"
        / "julia_reference"
        / "2023-01-23-15_julia.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "vienna_workbench" / "tier3_julia_reference.json",
    )
    args = parser.parse_args()

    model = load_premotor()
    recording = build_clamped_recording(
        model,
        load_atanas_recording(DEFAULT_REF / "data" / "Atanas_data" / f"{args.recording}.json"),
    )
    result = simulate_clamped(model, recording, ClampedParameters())
    julia = pd.read_csv(args.julia_csv)

    per_neuron: dict[str, dict[str, float]] = {}
    for i, name in enumerate(INTRINSIC_NEURONS):
        python_trace = result.intrinsic[i]
        julia_trace = julia[f"sim_{name}"].to_numpy(dtype=float)
        truth_trace = result.truth_intrinsic[i]
        delta = python_trace - julia_trace
        per_neuron[name] = {
            "pearson_r": float(np.corrcoef(python_trace, julia_trace)[0, 1]),
            "max_abs_difference": float(np.max(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(delta**2))),
            "python_truth_pearson_r": float(np.corrcoef(python_trace, truth_trace)[0, 1]),
            "julia_truth_pearson_r": float(np.corrcoef(julia_trace, truth_trace)[0, 1]),
        }

    reference_sha = {
        "recording_json": sha256(
            DEFAULT_REF / "data" / "Atanas_data" / f"{args.recording}.json"
        ),
        "weights_jld2": sha256(DEFAULT_REF / "data" / "model_weights" / "avg_model_weights.jld2"),
        "connectome_csv": sha256(DEFAULT_REF / "data" / "White_data_processed" / "df5.csv"),
        "julia_csv": sha256(args.julia_csv),
        "W_npy": sha256(ROOT / "data" / "processed" / "premotor" / "W.npy"),
        "neuron_names": sha256(ROOT / "data" / "processed" / "premotor" / "neuron_names.txt"),
    }
    report = {
        "schema": "vienna-tier3-julia-reference-v1",
        "recording": args.recording,
        "julia_reference": {
            "source": "scripts/reference/run_julia_reference.jl",
            "julia_version": "1.11.2",
            "solver": "Tsit5",
            "max_abs": float(np.max(np.abs(julia.filter(like="sim_").to_numpy()))),
        },
        "python": {
            "solver": result.method,
            "max_abs": float(result.metrics["max_abs"]),
            "status": result.status,
            "state_agreement": float(result.metrics["state_agreement"]),
            "behavior_distribution_js": float(result.metrics["behavior_distribution_js"]),
        },
        "summary": {
            "mean_pearson_r": float(np.mean([v["pearson_r"] for v in per_neuron.values()])),
            "min_pearson_r": float(np.min([v["pearson_r"] for v in per_neuron.values()])),
            "max_abs_difference": float(
                np.max([v["max_abs_difference"] for v in per_neuron.values()])
            ),
            "max_trace_abs": float(np.max(np.abs(result.intrinsic))),
        },
        "per_neuron": per_neuron,
        "provenance": {
            **git_state(),
            "command": "PYTHONPATH=src python scripts/compare_tier3_reference.py",
            "inputs_sha256": reference_sha,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "julia": "1.11.2 (pod)",
            },
            "acceptance_tolerance": {
                "mean_pearson_r_min": 0.999,
                "max_abs_difference_max": 0.05,
                "note": "Port parity; not a biological equivalence threshold.",
            },
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
