"""Compare the Python clamped core with the authors' Julia reference run.

The Julia CSV is produced by ``scripts/reference/run_julia_reference.jl``.
This script does not re-run Julia; it recomputes the Python side, aligns traces
by neuron name, and writes a machine-readable comparison plus input hashes.

Use ``--all`` to compare every ``*_julia.csv`` in the reference directory and
write both per-recording reports and one aggregate report.
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
DEFAULT_JULIA_DIR = ROOT / "reports" / "vienna_workbench" / "julia_reference"


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


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "julia": "1.11.2 (pod)",
    }


def compare_one(recording_id: str, julia_csv: Path) -> dict[str, object]:
    model = load_premotor()
    recording = build_clamped_recording(
        model,
        load_atanas_recording(DEFAULT_REF / "data" / "Atanas_data" / f"{recording_id}.json"),
        expected_core=None,
    )
    result = simulate_clamped(model, recording, ClampedParameters())
    julia = pd.read_csv(julia_csv)

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

    julia_behavior_sim = julia["behavior_sim"].to_numpy(dtype=int)
    julia_behavior_truth = julia["behavior_truth"].to_numpy(dtype=int)
    behavior_port_agreement = float(np.mean(result.behavior_sim == julia_behavior_sim))
    julia_behavior_agreement = float(np.mean(julia_behavior_truth == julia_behavior_sim))

    reference_sha = {
        "recording_json": sha256(
            DEFAULT_REF / "data" / "Atanas_data" / f"{recording_id}.json"
        ),
        "weights_jld2": sha256(DEFAULT_REF / "data" / "model_weights" / "avg_model_weights.jld2"),
        "connectome_csv": sha256(DEFAULT_REF / "data" / "White_data_processed" / "df5.csv"),
        "julia_csv": sha256(julia_csv),
        "W_npy": sha256(ROOT / "data" / "processed" / "premotor" / "W.npy"),
        "neuron_names": sha256(ROOT / "data" / "processed" / "premotor" / "neuron_names.txt"),
    }
    return {
        "schema": "vienna-tier3-julia-reference-v1",
        "recording": recording_id,
        "julia_reference": {
            "source": "scripts/reference/run_julia_reference.jl",
            "julia_version": "1.11.2",
            "solver": "Tsit5",
            "max_abs": float(np.max(np.abs(julia.filter(like="sim_").to_numpy()))),
            "core_neurons": int(result.full.shape[0]),
            "n_frames": int(len(julia)),
            "behavior_agreement": julia_behavior_agreement,
        },
        "python": {
            "solver": result.method,
            "max_abs": float(result.metrics["max_abs"]),
            "status": result.status,
            "state_agreement": float(result.metrics["state_agreement"]),
            "behavior_distribution_js": float(result.metrics["behavior_distribution_js"]),
            "core_neurons": int(result.full.shape[0]),
        },
        "summary": {
            "mean_pearson_r": float(np.mean([v["pearson_r"] for v in per_neuron.values()])),
            "min_pearson_r": float(np.min([v["pearson_r"] for v in per_neuron.values()])),
            "max_abs_difference": float(
                np.max([v["max_abs_difference"] for v in per_neuron.values()])
            ),
            "max_trace_abs": float(np.max(np.abs(result.intrinsic))),
            "behavior_agreement": float(result.metrics["state_agreement"]),
            "behavior_port_agreement": behavior_port_agreement,
            "behavior_agreement_delta_python_minus_julia": float(
                result.metrics["state_agreement"] - julia_behavior_agreement
            ),
        },
        "per_neuron": per_neuron,
        "provenance": {
            **git_state(),
            "command": f"PYTHONPATH=src python scripts/compare_tier3_reference.py --recording {recording_id}",
            "inputs_sha256": reference_sha,
            "environment": _environment(),
            "acceptance_tolerance": {
                "mean_pearson_r_min": 0.999,
                "max_abs_difference_max": 0.05,
                "note": "Port parity; not a biological equivalence threshold.",
            },
        },
    }


def _aggregate(reports: dict[str, dict[str, object]]) -> dict[str, object]:
    summaries = [report["summary"] for report in reports.values()]
    return {
        "n_recordings": len(reports),
        "mean_pearson_r": float(np.mean([s["mean_pearson_r"] for s in summaries])),
        "min_pearson_r": float(np.min([s["min_pearson_r"] for s in summaries])),
        "max_abs_difference": float(np.max([s["max_abs_difference"] for s in summaries])),
        "mean_behavior_agreement": float(np.mean([s["behavior_agreement"] for s in summaries])),
        "min_behavior_agreement": float(np.min([s["behavior_agreement"] for s in summaries])),
        "mean_behavior_port_agreement": float(
            np.mean([s["behavior_port_agreement"] for s in summaries])
        ),
        "min_behavior_port_agreement": float(
            np.min([s["behavior_port_agreement"] for s in summaries])
        ),
        "all_trace_parity_at_least_0_999": bool(
            all(s["mean_pearson_r"] >= 0.999 for s in summaries)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", default="2023-01-23-15")
    parser.add_argument(
        "--julia-csv",
        type=Path,
        default=DEFAULT_JULIA_DIR / "2023-01-23-15_julia.csv",
    )
    parser.add_argument("--julia-dir", type=Path, default=DEFAULT_JULIA_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "vienna_workbench" / "tier3_julia_reference.json",
    )
    parser.add_argument("--all", action="store_true", help="compare every available Julia CSV")
    args = parser.parse_args()

    if args.all:
        csvs = sorted(args.julia_dir.glob("*_julia.csv"))
        if not csvs:
            raise SystemExit(f"no *_julia.csv files in {args.julia_dir}")
        reports: dict[str, dict[str, object]] = {}
        for csv in csvs:
            recording_id = csv.name.removesuffix("_julia.csv")
            report = compare_one(recording_id, csv)
            reports[recording_id] = report
            per_out = args.out.parent / f"tier3_julia_reference_{recording_id}.json"
            if recording_id == "2023-01-23-15":
                per_out = args.out
            per_out.parent.mkdir(parents=True, exist_ok=True)
            per_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"wrote {per_out}")
        aggregate = {
            "schema": "vienna-tier3-julia-parity-v1",
            "aggregate": _aggregate(reports),
            "recordings": {
                recording_id: {
                    "core_neurons": report["python"]["core_neurons"],
                    "julia": report["julia_reference"],
                    "python": report["python"],
                    "summary": report["summary"],
                    "julia_csv_sha256": report["provenance"]["inputs_sha256"]["julia_csv"],
                }
                for recording_id, report in reports.items()
            },
            "provenance": {
                **git_state(),
                "command": "PYTHONPATH=src python scripts/compare_tier3_reference.py --all",
                "environment": _environment(),
                "acceptance_tolerance": {
                    "mean_pearson_r_min": 0.999,
                    "max_abs_difference_max": 0.05,
                    "note": "Port parity; not a biological equivalence threshold.",
                },
            },
        }
        aggregate_path = args.out.parent / "tier3_julia_parity_all.json"
        aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(json.dumps(aggregate["aggregate"], indent=2))
        print(f"wrote {aggregate_path}")
        return

    report = compare_one(args.recording, args.julia_csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
