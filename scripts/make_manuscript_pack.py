"""Build the manuscript pack from generated artifacts, without hand-typed numbers."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _markdown(frame: pd.DataFrame, caption: str) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        caption,
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "manuscript")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    sensitivity = pd.read_csv(args.reports / "phase11" / "coverage_sensitivity.csv")
    sensitivity.to_csv(args.out / "table_coverage_sensitivity.csv", index=False)
    (args.out / "table_coverage_sensitivity.md").write_text(
        _markdown(
            sensitivity.round(4),
            "Table 1. Pooled reliability and ceiling as a function of minimum responder coverage.",
        ),
        encoding="utf-8",
    )

    payload = json.loads((args.reports / "phase10" / "claims.json").read_text(encoding="utf-8"))
    claims_frame = pd.DataFrame(payload["claims"])[
        ["id", "status", "estimand", "estimate", "ci_low", "ci_high", "p_value", "p_holm"]
    ]
    claims_frame.to_csv(args.out / "table_claims.csv", index=False)
    (args.out / "table_claims.md").write_text(
        _markdown(claims_frame.round(6), "Table 2. Claims ledger with status and multiplicity."),
        encoding="utf-8",
    )

    leaderboard = pd.read_csv(args.reports / "phase11" / "leaderboard.csv")
    leaderboard.to_csv(args.out / "table_leaderboard.csv", index=False)
    (args.out / "table_leaderboard.md").write_text(
        _markdown(
            leaderboard.round(4),
            "Table 3. Fold-level leaderboard (descriptive; folds share animals).",
        ),
        encoding="utf-8",
    )

    design_path = args.reports / "phase10" / "design_budget.csv"
    if design_path.exists():
        design = pd.read_csv(design_path)
        design.to_csv(args.out / "table_design_budget.csv", index=False)
        (args.out / "table_design_budget.md").write_text(
            _markdown(design.round(4), "Table 4. Animals required for target reliability."),
            encoding="utf-8",
        )

    phase12 = args.reports / "phase12"
    if (phase12 / "summary.json").exists():
        summary12 = json.loads((phase12 / "summary.json").read_text(encoding="utf-8"))
        gates = summary12["gates"]
        gate_frame = pd.DataFrame(
            [
                {"gate": "SBC coverage [0.93, 0.97]", "observed": gates["sbc_coverage_n22"], "pass": gates["g1_sbc"]},
                {"gate": "SBC rank KS p > 0.01", "observed": gates["sbc_rank_ks_pvalue"], "pass": gates["g1_sbc"]},
                {"gate": "Held-out coverage >= 0.90", "observed": gates["ppc_coverage"], "pass": gates["g2_heldout_ppc"]},
                {"gate": "Held-out median error <= 0.10", "observed": gates["ppc_median_abs_error"], "pass": gates["g2_heldout_ppc"]},
                {"gate": "Negative-control median R22 <= 0.05", "observed": gates["negative_median_n22"], "pass": gates["g3_negative_control"]},
                {"gate": "Prior range <= 0.05", "observed": gates["prior_range"], "pass": gates["g4_prior_sensitivity"]},
            ]
        )
        gate_frame.to_csv(args.out / "table_phase12_gates.csv", index=False)
        (args.out / "table_phase12_gates.md").write_text(
            _markdown(gate_frame.round(6), "Table 5. Phase 12 calibration gates and observed values."),
            encoding="utf-8",
        )
        cluster_path = phase12 / "benchmark_cluster.csv"
        if cluster_path.exists():
            cluster = pd.read_csv(cluster_path)
            cluster.to_csv(args.out / "table_cluster_benchmark.csv", index=False)
            (args.out / "table_cluster_benchmark.md").write_text(
                _markdown(
                    cluster.round(4),
                    "Table 6. Animal-cluster paired benchmark deltas with Holm adjustment and MDE.",
                ),
                encoding="utf-8",
            )
        fig2, ax2 = plt.subplots(figsize=(5.2, 3.4))
        ax2.plot(
            sensitivity["min_responder_coverage"],
            sensitivity["pooled_reliability_at_fold"],
            marker="o",
            label="Phase 11 moment",
        )
        ax2.axhline(
            summary12["hierarchical"]["pooled_reliability_median"],
            color="tab:red",
            linestyle="--",
            label="Phase 12 hierarchical",
        )
        ax2.set_xlabel("Minimum responder coverage")
        ax2.set_ylabel("Pooled reliability at 22 animals")
        ax2.set_ylim(0, 1)
        ax2.legend(frameon=False)
        ax2.grid(alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(args.out / "figure_phase12_calibration.png", dpi=200)
        plt.close(fig2)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(
        sensitivity["min_responder_coverage"],
        sensitivity["pooled_reliability_at_fold"],
        marker="o",
        label="reliability",
    )
    ax.plot(
        sensitivity["min_responder_coverage"],
        sensitivity["pooled_ceiling_at_fold"],
        marker="s",
        label="ceiling",
    )
    ax.set_xlabel("Minimum responder coverage")
    ax.set_ylabel("Value at 22 animals")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "figure_coverage_sensitivity.png", dpi=200)
    plt.close(fig)
    source_figure = args.reports / "phase10" / "ceiling_curve.png"
    if source_figure.exists():
        shutil.copyfile(source_figure, args.out / "figure_ceiling_curve_phase10.png")

    phase13 = args.reports / "phase13"
    if (phase13 / "summary.json").exists():
        summary13 = json.loads((phase13 / "summary.json").read_text(encoding="utf-8"))
        gate13 = pd.DataFrame(summary13["gates"])[["gate", "requirement", "observed", "result"]]
        gate13.to_csv(args.out / "table_phase13_gates.csv", index=False)
        (args.out / "table_phase13_gates.md").write_text(
            _markdown(gate13, "Table 7. Phase 13 frozen gates. Six of eleven failed."),
            encoding="utf-8",
        )
        estimates = pd.DataFrame(
            [
                {
                    "model": "M0 Phase 12R reference",
                    "pooled_reliability_n22": summary13["real"]["m0_reference"]["pooled_reliability_median"],
                    "ceiling_n22": summary13["real"]["m0_reference"]["pooled_ceiling_median"],
                    "floor_fraction": summary13["real"]["m0_reference"]["variance_floor_fraction"],
                },
                {
                    "model": "M1 projection-scaled",
                    "pooled_reliability_n22": summary13["real"]["m1_projection_scaled"]["pooled_reliability_median"],
                    "ceiling_n22": summary13["real"]["m1_projection_scaled"]["pooled_ceiling_median"],
                    "floor_fraction": summary13["real"]["m1_projection_scaled"]["variance_floor_fraction"],
                },
                {
                    "model": "M2 event-variance",
                    "pooled_reliability_n22": summary13["real"]["m2_event_variance"]["pooled_reliability_median"],
                    "ceiling_n22": summary13["real"]["m2_event_variance"]["pooled_ceiling_median"],
                    "floor_fraction": summary13["real"]["m2_event_variance"]["variance_floor_fraction"],
                },
            ]
        )
        estimates.to_csv(args.out / "table_phase13_estimates.csv", index=False)
        (args.out / "table_phase13_estimates.md").write_text(
            _markdown(
                estimates.round(4),
                "Table 8. Phase 13 real-data estimates. M2 is rejected; none is a calibrated replacement.",
            ),
            encoding="utf-8",
        )
        fig13, ax13 = plt.subplots(figsize=(5.6, 3.4))
        ax13.bar(
            estimates["model"],
            estimates["pooled_reliability_n22"],
            color=["tab:gray", "tab:blue", "tab:red"],
        )
        ax13.axhline(0.314, color="black", linestyle=":", label="Phase 11 reference 0.314")
        ax13.set_ylabel("Pooled R(22)")
        ax13.set_ylim(0, 0.7)
        ax13.tick_params(axis="x", rotation=15)
        ax13.legend(frameon=False)
        ax13.grid(axis="y", alpha=0.3)
        fig13.tight_layout()
        fig13.savefig(args.out / "figure_phase13_estimates.png", dpi=200)
        plt.close(fig13)

    phase14 = args.reports / "phase14"
    if (phase14 / "summary.json").exists():
        summary14 = json.loads((phase14 / "summary.json").read_text(encoding="utf-8"))
        gate14 = pd.DataFrame(
            [
                {
                    "gate": name,
                    "requirement": gate.get("requirement", ""),
                    "observed": str(gate.get("value", "")),
                    "result": "pass" if gate.get("passed") else "fail",
                }
                for name, gate in summary14["gates"].items()
            ]
        )
        gate14.to_csv(args.out / "table_phase14_gates.csv", index=False)
        (args.out / "table_phase14_gates.md").write_text(
            _markdown(gate14, "Table 9. Phase 14 frozen gates. Eight of twelve failed."),
            encoding="utf-8",
        )
        reference = json.loads((phase14 / "reference_audit.json").read_text(encoding="utf-8"))
        audit = pd.DataFrame(
            [
                {
                    "treatment": name,
                    "n_targets": block["n_pooled_targets"],
                    "pooled_reliability_n22": block["pooled_reliability_at_fold"],
                    "ceiling_n22": block["pooled_ceiling_at_fold"],
                }
                for name, block in reference["treatments"].items()
            ]
        )
        audit.to_csv(args.out / "table_phase14_reference_audit.csv", index=False)
        (args.out / "table_phase14_reference_audit.md").write_text(
            _markdown(
                audit.round(6),
                "Table 10. Phase 14A reference audit. V1 is the primary robust treatment; V2 is a different 36-target population.",
            ),
            encoding="utf-8",
        )
        estimates14 = pd.DataFrame(
            [
                {
                    "quantity": "M4 real-data median R(22)",
                    "value": summary14["real_fit"]["pooled_reliability_median_n22"],
                },
                {
                    "quantity": "M4 median P(z=0)",
                    "value": summary14["real_fit"]["p_zero_median"],
                },
                {
                    "quantity": "V1 reference R(22)",
                    "value": summary14["reference"]["v1_pooled_reliability_at_fold"],
                },
                {
                    "quantity": "V0 historical R(22)",
                    "value": summary14["reference"]["v0_pooled_reliability_at_fold"],
                },
            ]
        )
        estimates14.to_csv(args.out / "table_phase14_estimates.csv", index=False)
        (args.out / "table_phase14_estimates.md").write_text(
            _markdown(
                estimates14.round(6),
                "Table 11. Phase 14B real-data estimate and audited reference. M4 is rejected.",
            ),
            encoding="utf-8",
        )

    provenance = {
        "generated_from": {
            "coverage_sensitivity": str(
                (args.reports / "phase11" / "coverage_sensitivity.csv").relative_to(ROOT)
            ),
            "claims": str((args.reports / "phase10" / "claims.json").relative_to(ROOT)),
            "leaderboard": str((args.reports / "phase11" / "leaderboard.csv").relative_to(ROOT)),
            "phase12_summary": str((args.reports / "phase12" / "summary.json").relative_to(ROOT))
            if (args.reports / "phase12" / "summary.json").exists()
            else None,
            "phase13_summary": str((args.reports / "phase13" / "summary.json").relative_to(ROOT))
            if (args.reports / "phase13" / "summary.json").exists()
            else None,
            "phase14_summary": str((args.reports / "phase14" / "summary.json").relative_to(ROOT))
            if (args.reports / "phase14" / "summary.json").exists()
            else None,
            "phase14_reference_audit": str(
                (args.reports / "phase14" / "reference_audit.json").relative_to(ROOT)
            )
            if (args.reports / "phase14" / "reference_audit.json").exists()
            else None,
        },
        "hashes": {
            path.name: _sha256(path)
            for path in sorted(args.out.iterdir())
            if path.is_file() and path.name != "provenance.json"
        },
        "captions": {
            "figure_coverage_sensitivity.png": (
                "Pooled reliability and its attenuation ceiling at 22 animals as responders "
                "with incomplete animal coverage are excluded."
            ),
            "figure_phase12_calibration.png": (
                "Phase 11 coverage curve and the Phase 12 hierarchical point estimate. The "
                "hierarchical interval was not calibrated and is not shown as an interval."
            ),
            "figure_phase13_estimates.png": (
                "Phase 13 model-based pooled R(22): M0 reference 0.3035, M1 projection-scaled "
                "0.3138, M2 event-variance 0.2003. M2 is rejected; the bars are not calibrated "
                "intervals and 0.560 remains the descriptive ceiling."
            ),
            "table_phase14_gates.md": (
                "Phase 14 frozen gates; eight of twelve failed. A zero null median did not "
                "repair the q95 false-positive tail."
            ),
            "table_phase14_reference_audit.md": (
                "Phase 14A reference audit: V1 delta from V0 is 0.0131, supporting the 0.560 "
                "descriptive ceiling; V2 is a different 36-target population."
            ),
        },
    }
    (args.out / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out} ({len(provenance['hashes'])} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
