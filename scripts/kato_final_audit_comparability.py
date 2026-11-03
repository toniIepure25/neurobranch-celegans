"""Audit the ordered 51/49 decomposition and model comparability."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports" / "simulator" / "kato_common_state"
OUT = ROOT / "reports" / "simulator" / "kato_final_audit"


def main() -> None:
    summary = pd.read_csv(BASE / "model_summary.csv")
    energy = dict(zip(summary["model"], summary["energy_h20"], strict=False))
    b0, b1m, b3, b4m, b5 = (
        float(energy[k])
        for k in ("B0_persistence", "B1_matched_own", "B3_population", "B4_matched_own_500", "B5_common_k10")
    )
    total = b4m - b3
    common = b4m - b5
    residual = b5 - b3
    assert abs(total - (common + residual)) < 1e-12

    orders = []
    for baseline_name, baseline in (("B4m (reported)", b4m), ("B1m", b1m), ("B0 persistence", b0)):
        total_b = baseline - b3
        common_b = baseline - b5
        residual_b = b5 - b3
        orders.append(
            {
                "baseline": baseline_name,
                "baseline_energy": baseline,
                "total_advantage": total_b,
                "common_increment": common_b,
                "residual_increment": residual_b,
                "common_share": common_b / total_b,
                "residual_share": residual_b / total_b,
            }
        )

    config = pd.read_csv(BASE / "model_config.csv")
    rows = []
    for _, row in config[config["model"].isin(
        ["B1_matched_own", "B2_own_trend", "B4_matched_own_500", "B5_common_k10", "B3_population", "B6_coefficient_ablation"]
    )].iterrows():
        rows.append(
            {
                "model": row["model"],
                "window": row["window"],
                "parameters_per_neuron_or_total": row["n_parameters_per_neuron_or_total"],
                "penalty": row["penalty"],
                "intercept": row["intercept"],
                "innovation": row["innovation"],
                "role": row["role"],
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "model_comparability.csv", index=False)
    pd.DataFrame(orders).to_csv(OUT / "decomposition_order_sensitivity.csv", index=False)

    parameter_rows = []
    for n in (103, 127, 123, 118, 121):
        parameter_rows.append(
            {
                "n_neurons": n,
                "B3_parameters": n * (n + 1),
                "B5_k10_parameters": 12 * n + 110,
                "B3_to_B5_ratio": n * (n + 1) / (12 * n + 110),
                "B3_extra_parameters": n * (n + 1) - (12 * n + 110),
            }
        )
    pd.DataFrame(parameter_rows).to_csv(OUT / "parameter_counts.csv", index=False)

    payload = {
        "schema": "neurobranch.kato_final_audit.comparability.v1",
        "energies": {"B0": b0, "B1m": b1m, "B3": b3, "B4m": b4m, "B5_k10": b5},
        "reported_decomposition": {
            "total_B4m_minus_B3": total,
            "common_B4m_minus_B5": common,
            "residual_B5_minus_B3": residual,
            "common_share": common / total,
            "residual_share": residual / total,
        },
        "order_sensitivity": orders,
        "percentages_are": "successive increments in one ordered model sequence, not unique variance components",
        "matched": [
            "same 175 held-out landmarks and targets",
            "same J5 common innovations",
            "500-frame causal window for B3/B4m/B5",
            "ridge 1.0 on standardized features",
            "unpenalized intercept",
            "forecast initialized at the same current frame",
            "same ten-dimensional PCA evaluation space",
        ],
        "not_matched": [
            "B3 has n*(n+1) parameters; B5-k10 has 12*n+110",
            "B3 has full cross-neuron current state; B5 has target own value plus ten shared factor scores",
            "B5 is a restricted diagonal-plus-factor model, not a refit of B3",
            "B4m is a deliberately weak own-history baseline; B1m is a stronger own-history baseline",
            "the ten-factor representation is fixed at k=10 and is not selected afresh",
        ],
        "parameter_counts": parameter_rows,
        "interpretation": (
            "Arithmetic is correct. The 51/49 split is an ordering artifact: changing the "
            "baseline to B1m changes the reported shares to roughly 24/76, and B0 to 72/28. "
            "The split does not identify physiological common input or neuron-specific coupling."
        ),
    }
    (OUT / "comparability_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
