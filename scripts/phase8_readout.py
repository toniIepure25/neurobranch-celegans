"""Turn a completed Phase 8 run into a cautious readout and result table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from neurobranch.config import PROJECT_ROOT, git_state, resolve_path


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _g(value: Any) -> Any:
    if isinstance(value, float) and not pd.notna(value):
        return None
    return value


def build_readout(output: Path) -> dict[str, Any]:
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    metrics_path = output / "metrics.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    h1 = summary.get("h1_twin_vs_cro", {})
    h2 = summary.get("h2_k8_vs_k0", {})
    h3 = summary.get("h3_eig_vs_random", {})
    h4 = summary.get("h4_sequence_rollout", {})
    deployment: dict[str, Any] = {}
    if not metrics.empty and "strategy" in metrics.columns:
        primary = metrics[(metrics["model"] == "twin") & (metrics["k"] == 8)]
        if not primary.empty:
            table = (
                primary.groupby("strategy")["pair_spearman"]
                .agg(["mean", "std", "count"])
                .reset_index()
                .sort_values("mean", ascending=False)
            )
            deployment["strategy_table_k8"] = table.to_dict(orient="records")
    if not metrics.empty and "k" in metrics.columns:
        budget = metrics[
            (metrics["model"] == "twin") & (metrics["strategy"] == "random")
        ].groupby("k")["pair_spearman"].agg(["mean", "std", "count"]).reset_index()
        deployment["random_budget_curve"] = budget.to_dict(orient="records")
    audit_clean = bool(
        not audit.get("fold_leaks")
        and not audit.get("animal_overlaps")
        and not audit.get("support_test_target_overlaps")
        and not audit.get("support_test_event_overlaps")
        and not audit.get("test_event_in_train_or_val")
    )
    decisions = {
        "H1_primary": {
            "supported": bool(h1.get("supported", False)),
            "pair_delta": _g(h1.get("delta")),
            "pair_ci": [_g(h1.get("ci_low")), _g(h1.get("ci_high"))],
            "trajectory_delta": _g(h1.get("trajectory_delta")),
            "gate": "pair Δ ≥ +0.010 & CI low > 0 & trajectory Δ ≥ +0.050",
        },
        "H2_personalisation": {
            "supported": bool(h2.get("supported", False)),
            "delta": _g(h2.get("delta")),
            "ci": [_g(h2.get("ci_low")), _g(h2.get("ci_high"))],
            "gate": "Δ ≥ +0.010 & CI low > 0",
        },
        "H3_active_eig": {
            "supported": bool(h3.get("supported", False)),
            "confirmatory": bool(h3.get("confirmatory", False)),
            "delta": _g(h3.get("delta")),
            "ci": [_g(h3.get("ci_low")), _g(h3.get("ci_high"))],
            "gate": "Δ ≥ +0.010 & CI low > 0; confirmatory only if H2 supported",
        },
        "H4_sequence": {
            "supported": bool(h4.get("supported", False)),
            "trajectory_delta": _g(h4.get("trajectory_delta")),
            "coverage": _g(h4.get("coverage")),
            "gate": "trajectory Δ ≥ +0.050 and coverage 0.80–0.95",
        },
    }
    readout = {
        "phase": 8,
        "name": "Adaptive Perturbation Twin",
        "warning": "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.",
        "audit_clean": audit_clean,
        "audit": audit,
        "decisions": decisions,
        "deployment_readout": deployment,
        "calibration_scale_by_fold": [
            _g(item.get("calibration_scale")) for item in summary.get("fold_extras", [])
        ],
        "n_predictions": summary.get("n_predictions"),
        "n_metrics": summary.get("n_metrics"),
        "git": git_state(PROJECT_ROOT),
        "config": summary.get("config", {}),
        "artifacts": {
            "summary": {"sha256": _hash(output / "summary.json")},
            "audit": {"sha256": _hash(output / "audit.json")},
        },
    }
    if metrics_path.exists():
        readout["artifacts"]["metrics"] = {"sha256": _hash(metrics_path)}
    readout_path = output / "readout.json"
    readout_path.write_text(json.dumps(readout, indent=2, allow_nan=False), encoding="utf-8")
    _write_markdown(output, readout)
    return readout


def _write_markdown(output: Path, readout: dict[str, Any]) -> None:
    d = readout["decisions"]
    lines = [
        "# Phase 8 — Adaptive Perturbation Twin: readout",
        "",
        "> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**",
        "",
        "This is a pseudo-prospective, model-based benchmark. It does not establish",
        "causal effects, a closed-loop experimental result, or a validated digital twin.",
        "",
        "## Hypothesis decisions",
        "",
        "| Hypothesis | Supported | Effect | Gate |",
        "|---|---|---|---|",
        f"| H1 APT(k=0) vs CRO | {d['H1_primary']['supported']} | pair Δ {d['H1_primary']['pair_delta']}, trajectory Δ {d['H1_primary']['trajectory_delta']} | {d['H1_primary']['gate']} |",
        f"| H2 APT(k=8) vs k=0 | {d['H2_personalisation']['supported']} | Δ {d['H2_personalisation']['delta']} | {d['H2_personalisation']['gate']} |",
        f"| H3 EIG vs random | {d['H3_active_eig']['supported']} | Δ {d['H3_active_eig']['delta']} | {d['H3_active_eig']['gate']} |",
        f"| H4 sequence rollout | {d['H4_sequence']['supported']} | trajectory Δ {d['H4_sequence']['trajectory_delta']}, coverage {d['H4_sequence']['coverage']} | {d['H4_sequence']['gate']} |",
        "",
        f"Leakage audit clean: **{readout['audit_clean']}**.",
        "",
        "## Strategy readout at k=8",
        "",
    ]
    table = readout.get("deployment_readout", {}).get("strategy_table_k8", [])
    if table:
        lines.extend(["| strategy | mean pair Spearman | std | n |", "|---|---|---|---|"])
        for row in table:
            lines.append(
                f"| {row['strategy']} | {row['mean']:.6f} | {row['std']:.6f} | {int(row['count'])} |"
            )
    else:
        lines.append("No strategy metrics were produced.")
    lines.extend(
        [
            "",
            "## What this does and does not show",
            "",
            "- It tests animal-conditional, support-set personalisation under",
            "  target-disjoint support/test splits.",
            "- It does **not** test a wet-lab closed loop, causal graph, cross-dataset",
            "  transfer, or a foundation model.",
            "- `eig` is a deep-ensemble disagreement proxy, not a formal posterior",
            "  expected-information-gain guarantee.",
            "- A null result is reported with intervals, not converted into a claim of",
            "  absence.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "python scripts/phase8_run.py --config configs/phase8/apt.yaml",
            "python scripts/phase8_readout.py",
            "```",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/phase8")
    args = parser.parse_args()
    out = resolve_path(args.output, PROJECT_ROOT)
    readout = build_readout(out)
    print(json.dumps(readout["decisions"], indent=2))
    print("\nMODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
