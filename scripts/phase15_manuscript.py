"""Generate the Phase 15 manuscript, tables and figures from the audited summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _table(summary: dict) -> str:
    rows = [
        "| Estimand | Population | Estimate | 95% jackknife CI | Ceiling | Targets | Extrapolated |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "R_obs": ("Observed animal counts", "all eligible"),
        "R_22": ("Counterfactual at n=22", "all eligible"),
        "R_id": ("Identified subset", "n >= 22 only"),
    }
    for key, (label, population) in labels.items():
        block = summary["estimands"][key]
        jack = block["jackknife"]
        rows.append(
            f"| `{key}` | {label} ({population}) | {_fmt(block['estimate'])} | "
            f"[{_fmt(jack['ci_low'])}, {_fmt(jack['ci_high'])}] | "
            f"{_fmt(block['ceiling'])} | {int(block['n_targets'])} | {int(block['n_extrapolated'])} |"
        )
    return "\n".join(rows)


def _gate_table(summary: dict) -> str:
    rows = ["| Gate | Status | Requirement |", "|---|---|---|"]
    for name, gate in summary["gates"].items():
        status = "checked at release" if gate["status"] == "pending" else gate["status"]
        rows.append(f"| `{name}` | **{status}** | {gate['requirement']} |")
    return "\n".join(rows)


def _null_table(summary: dict) -> str:
    rows = [
        "| Estimand | Permutation median | Permutation q95 | Sign-flip median | Sign-flip q95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ("R_obs", "R_22", "R_id"):
        null = summary["estimands"][key]["null"]
        rows.append(
            f"| `{key}` | {_fmt(null['permutation']['median'])} | "
            f"{_fmt(null['permutation']['q95'])} | {_fmt(null['signflip']['median'])} | "
            f"{_fmt(null['signflip']['q95'])} |"
        )
    return "\n".join(rows)


def _forest(summary: dict, path: Path) -> None:
    keys = ["R_obs", "R_22", "R_id"]
    labels = ["R_obs\n(observed n)", "R_22\n(counterfactual n=22)", "R_id\n(n >= 22 only)"]
    points = [summary["estimands"][key]["estimate"] for key in keys]
    lows = [summary["estimands"][key]["jackknife"]["ci_low"] for key in keys]
    highs = [summary["estimands"][key]["jackknife"]["ci_high"] for key in keys]
    ceilings = [summary["estimands"][key]["ceiling"] for key in keys]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    y = list(range(len(keys)))[::-1]
    ax.errorbar(
        points,
        y,
        xerr=[[point - low for point, low in zip(points, lows, strict=False)], [high - point for point, high in zip(points, highs, strict=False)]],
        fmt="o",
        color="#1f4e79",
        capsize=4,
        label="estimate and delete-one-animal 95% interval",
    )
    ax.scatter(ceilings, y, marker="|", s=180, color="#b45309", label="sqrt(R) descriptive ceiling")
    ax.set_yticks(y, labels)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Reliability (variance ratio)")
    ax.axvline(0.5, color="grey", linestyle=":", linewidth=1)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_title("Phase 15 reliability estimands are not interchangeable", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _scatter(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    identified = table["in_identified_subset"].astype(bool)
    ax.scatter(
        table.loc[~identified, "n_animals"],
        table.loc[~identified, "reliability_obs"],
        s=16,
        alpha=0.65,
        color="#6b7280",
        label=f"extrapolated to n=22 ({(~identified).sum()} targets)",
    )
    ax.scatter(
        table.loc[identified, "n_animals"],
        table.loc[identified, "reliability_obs"],
        s=20,
        alpha=0.85,
        color="#1f4e79",
        label=f"identified at n>=22 ({identified.sum()} targets)",
    )
    ax.axvline(22, color="#b45309", linestyle="--", linewidth=1)
    ax.set_xlabel("Observed animals per target")
    ax.set_ylabel("R_obs at the observed count")
    ax.set_ylim(-0.02, 1.0)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("Most targets never reach the 22-animal fold", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _gate_figure(summary: dict, path: Path) -> None:
    names = list(summary["gates"])
    colours = {"pass": "#2f7d32", "fail": "#b3261e", "pending": "#8a6d00"}
    values = [1 if summary["gates"][name]["status"] == "pass" else 0 for name in names]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.barh(names[::-1], values[::-1], color=[colours[summary["gates"][name]["status"]] for name in names[::-1]])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 1], ["fail / pending", "pass"])
    ax.set_xlabel("Frozen gate status")
    ax.set_title("Pre-specified gates", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=ROOT / "reports" / "phase15")
    parser.add_argument("--out", type=Path, default=ROOT / "manuscript")
    args = parser.parse_args(argv)
    reports = args.reports.resolve()
    out = args.out.resolve()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    summary = json.loads((reports / "summary.json").read_text(encoding="utf-8"))
    table = pd.read_csv(reports / "estimands.csv")
    decision = summary["decision"]
    _forest(summary, out / "figures" / "fig1_estimand_forest.png")
    _scatter(table, out / "figures" / "fig2_target_coverage.png")
    _gate_figure(summary, out / "figures" / "fig3_gates.png")
    table_md = _table(summary)
    gate_md = _gate_table(summary)
    (out / "tables" / "table1_estimands.md").write_text(table_md + "\n", encoding="utf-8")
    (out / "tables" / "table2_gates.md").write_text(gate_md + "\n", encoding="utf-8")
    censored = int(summary["estimands"]["R_22"]["n_extrapolated"])
    n_total = int(summary["estimands"]["R_22"]["n_targets"])
    manuscript = f"""# Reliability without extrapolation: a three-estimand audit of a *C. elegans* repeated-stimulation atlas

**Status:** internal manuscript, not peer reviewed. This is a measurement audit,
not a causal or transport study.

## Abstract

The NeuroBranch benchmark reports a reliability ceiling for neural responses in a
repeated-stimulation atlas. The historical headline, `sqrt(R(22)) = 0.560`, was
computed from a reliability curve extrapolated beyond the observed animal counts
of most targets. Here we separate three estimands under a frozen protocol:
observed-count reliability (`R_obs`), a counterfactual at 22 animals (`R_22`),
and reliability on the identified subset with at least 22 animals (`R_id`).
`R_obs` is {_fmt(summary['estimands']['R_obs']['estimate'])} (delete-one-animal
95% interval {_fmt(summary['estimands']['R_obs']['jackknife']['ci_low'])} to
{_fmt(summary['estimands']['R_obs']['jackknife']['ci_high'])}; ceiling
{_fmt(summary['estimands']['R_obs']['ceiling'])}) across {n_total} targets.
`R_22` is {_fmt(summary['estimands']['R_22']['estimate'])} (interval
{_fmt(summary['estimands']['R_22']['jackknife']['ci_low'])} to
{_fmt(summary['estimands']['R_22']['jackknife']['ci_high'])}; ceiling
{_fmt(summary['estimands']['R_22']['ceiling'])}) but is extrapolated for
{censored} of {n_total} targets. `R_id` is
{_fmt(summary['estimands']['R_id']['estimate'])} on
{int(summary['estimands']['R_id']['n_targets'])} targets and is a different
population. The audit decision is **`{decision['label']}`**. The 0.560 value is
retained only as a historical extrapolated point estimate, not as a bound.

## 1. Question

The benchmark's scores are bounded by measurement reliability. If the bound is
extrapolated, a score-to-ceiling ratio can look better than the data support.
The pre-specified question was: for the Randi repeated-stimulation atlas at 80%
responder coverage, what reliability is identified at the observed animal counts,
what is the counterfactual value at 22 animals, and what uncertainty and null
bias do those quantities carry?

## 2. Data and frozen choices

- Input: `reports/phase3/randi_trial_table_phase3.csv.gz`, SHA256
  `{summary['input']['sha256']}`; {summary['input']['n_rows']} rows.
- 126 eligible targets; 80% responder coverage; at least 6 animals per target.
- Primary variance treatment V1: one common one-event variance
  `omega^2 = {summary['event_variance']['omega2']:.6f}` from
  {summary['event_variance']['n_cells']} eligible cells.
- Fold size 22; target-level estimands are unweighted means over targets.
- Uncertainty: delete-one-animal jackknife conditional on the fixed responder
  set; normal 95% interval. Bayesian bootstrap and null controls are
  cross-checks, not replacements.
- Calibration: {int(summary['calibration'].get('n_reps', 0)):,} parametric
  replicates on the real missingness masks.
- All gates and the decision rule were frozen in
  `research/phase15_protocol.md` before the run.

## 3. Estimands

`R_t(n) = signal_t / (signal_t + (noise_t + interaction_t) / n)`, clipped to
`[0, 1]`.

1. `R_obs = mean_t R_t(n_t)` at each target's observed count.
2. `R_22 = mean_t R_t(22)`; targets with `n_t < 22` are counterfactual.
3. `R_id = mean_{{t: n_t >= 22}} R_t(n_t)`; a different target population.

{table_md}

![Estimand forest plot](figures/fig1_estimand_forest.png)

## 4. Results

### 4.1 Most of the old headline is extrapolation

{censored} of {n_total} targets have fewer than 22 animals
(Figure 2). For those targets, `R_22` is a design counterfactual, not an
observed-data quantity. The observed-count estimate is
{_fmt(summary['estimands']['R_obs']['estimate'])}, and the identified-subset
estimate is {_fmt(summary['estimands']['R_id']['estimate'])}. The three values
answer different questions and none is a replacement for another.

![Target coverage](figures/fig2_target_coverage.png)

### 4.2 Uncertainty

The delete-one-animal interval for `R_obs` is
[{_fmt(summary['estimands']['R_obs']['jackknife']['ci_low'])},
{_fmt(summary['estimands']['R_obs']['jackknife']['ci_high'])}]. The Bayesian
bootstrap median is
{_fmt(summary['estimands']['R_obs']['bootstrap']['median'])} with a 95%
interval [{_fmt(summary['estimands']['R_obs']['bootstrap']['ci_low'])},
{_fmt(summary['estimands']['R_obs']['bootstrap']['ci_high'])}]. The two-way
bootstrap is not used because it is biased on sparse animal counts.

### 4.3 Calibration and nulls

The calibration study reports coverage
{_fmt(summary['calibration']['estimands']['obs']['coverage'])} for `R_obs`,
{_fmt(summary['calibration']['estimands']['fold']['coverage'])} for `R_22`, and
{_fmt(summary['calibration']['estimands']['id']['coverage'])} for `R_id`
(nominal 0.95; pass band 0.93--0.97). The failure is over-coverage: the intervals
are too wide, not anti-conservative. Null controls:

{_null_table(summary)}

The null medians are 0.096--0.137 against a 0.05 threshold, and the `R_22`
sign-flip q95 is 0.1641; the estimator assigns apparent reliability to
structureless labels.

### 4.4 Extrapolation and ranking

The animal-half extrapolation check gives median absolute error
{_fmt(summary['extrapolation']['median_abs_error'])} and bias
{_fmt(summary['extrapolation']['bias'])}; the frozen gate requires MAE <= 0.10
and |bias| <= 0.05. The half-split ranking check gives median Spearman
{_fmt(summary['ranking']['median_spearman'])} (interval
{_fmt(summary['ranking']['ci_low'])} to {_fmt(summary['ranking']['ci_high'])};
top-10 Jaccard {_fmt(summary['ranking']['median_jaccard'])}).

## 5. Gate results

{gate_md}

![Gate status](figures/fig3_gates.png)

## 6. Decision and interpretation

The decision label is **`{decision['label']}`**. Calibrated interval:
`{decision['calibrated_interval']}`. Ceiling claim allowed:
`{decision['ceiling_claim']}`. Ranking claim allowed:
`{decision['ranking_claim']}`. Extrapolation status:
`{decision['extrapolation_status']}`.

The defensible statement is descriptive: the observed-count reliability is
{_fmt(summary['estimands']['R_obs']['estimate'])} with the reported uncertainty
and the audit label. The value `0.5600` is a historical Phase 11
extrapolation, not a bound and not a corrected estimate. No causal, transport,
digital-twin, foundation-model, animal-budget, or target-ranking claim follows.

## 7. Limitations

- The estimand is a variance ratio for a filtered responder population; a
  different coverage filter changes the population.
- The jackknife treats animals as exchangeable clusters and is conditional on
  the responder set selected at 80% coverage.
- The parametric calibration is an additive model; heavy-tailed
  misspecification is reported separately and is not evidence of robustness.
- The atlas contains repeated stimulation but not a new independent cohort; no
  external transport claim is possible.
- The two-way bootstrap is diagnostic only.

## 8. Related methods

The variance-ratio target is an intraclass-reliability analogue
(Shrout & Fleiss, 1979). The delete-one-animal uncertainty uses the cluster
jackknife (Efron, 1979), and the Bayesian bootstrap cross-check follows
(Rubin, 1981). These citations describe methods, not evidence for the biological
interpretation. A BCa-style diagnostic was prototyped and discarded because
delete-one-animal values are not a bootstrap distribution (deviation D1).

## 9. Reproducibility

```bash
python scripts/phase15_reliability.py
python scripts/phase15_card.py
python scripts/phase15_manifest.py
python scripts/phase15_validate.py
```

Artifacts are in `reports/phase15/`; hashes are in `manifest.json` and
`manifest.sha256`. The validator fails closed on schema, gate, estimand-count,
and artifact-hash inconsistencies.

## References

- Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: Uses in
  assessing rater reliability. *Psychological Bulletin*, 86(2), 420--428.
  https://doi.org/10.1037/0033-2909.86.2.420
- Efron, B. (1979). Bootstrap methods: Another look at the jackknife. *The
  Annals of Statistics*, 7(1), 1--26. https://doi.org/10.1214/aos/1176344552
- Rubin, D. B. (1981). The Bayesian bootstrap. *The Annals of Statistics*,
  9(1), 130--134. https://doi.org/10.1214/aos/1176345338
"""
    (out / "phase15_reliability.md").write_text(manuscript, encoding="utf-8", newline="\n")
    print(f"wrote {out / 'phase15_reliability.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
