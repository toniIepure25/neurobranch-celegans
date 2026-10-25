# Phase 7 artifacts

**Warning:** MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.

This directory contains the canonical pseudo-prospective active-acquisition run:

| Path | Content |
|---|---|
| `metrics.csv` | one row per fold × acquisition seed × strategy × budget |
| `budget_curves.csv` | mean pair Spearman by strategy and budget |
| `deltas.csv` | paired strategy-minus-random deltas with block-bootstrap CIs |
| `targets_to_threshold.csv` | targets needed to reach 90% of the full-atlas score |
| `predictions.parquet` | primary-budget outer-test predictions for every strategy |
| `selection_log.jsonl` | every acquisition decision, candidate-pool hash and selected target |
| `audit.json` | leakage checks, split hashes, config and git state |
| `connectivity/edge_metrics.csv` | direct-edge and anatomy-unexplained AUROC |
| `discrepancy_pairs.csv` | top anatomy-unexplained functional-coupling candidates || `design/` | target and pair hypotheses for a future experiment |
| `state/` | pre-stimulus cache and extraction summary |
| `state_ablation/` | reduced H3 base-vs-state ablation, if run |
| `manifest.json` | hashes of all Phase 7 artifacts |

Raw Randi recordings are not redistributed. The predictions are derived from the Phase 3 trial table and are not a substitute for a prospective experiment.

The primary decision is in `summary.json`; the human-readable report is `docs/PHASE7_RESULTS.md`.
