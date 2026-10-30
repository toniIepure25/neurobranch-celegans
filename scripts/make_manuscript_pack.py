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
        },
    }
    (args.out / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out} ({len(provenance['hashes'])} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
