# Kato final audit artifact index

Status: `FULL_END_TO_END_REPRODUCTION_PASSED`; decision
`READY_FOR_SUPERVISOR_REVIEW`; scientific tier `SUPPORTED_EXPLORATORY`.

## Reproduction and provenance

| File | Purpose |
| --- | --- |
| `preflight.json` | Git/env/dataset/splits/seeds/config snapshot |
| `full_rerun_status.json` | PID, command, elapsed time, stdout/stderr and output counts |
| `clean_rerun/` | Independently regenerated end-to-end outputs (21 files) |
| `reproduction_comparison.json` | Frozen-vs-rerun comparison, tolerances and gate statuses |
| `environment/pip_freeze.txt` | Installed package versions |
| `logs/` | Clean-run stdout/stderr/start/pid |
| `manifest.json` | SHA256 manifest for the audit package |
| `claims_ledger.json` | 22 claims with evidence tiers and caveats |
| `decision_record.json` | Final decision and conditions |

## Audit tables

| File | Purpose |
| --- | --- |
| `uncertainty_bootstrap.csv` | Block 3/5/10/20 intervals |
| `uncertainty_summary.json` | Primary interval, dependence and subsampling summary |
| `session_estimates.csv` | Per-session effects and within-session intervals |
| `leave_one_session_out.csv` | Pooled means excluding each session |
| `paired_serial_dependence.csv` | Lag dependence of paired differences |
| `landmark_subsample_sensitivity.csv` | Every 2nd/3rd/5th landmark |
| `horizon_uncertainty.csv` | H1/H5/H10/H20 intervals |
| `energy_strata_exploratory.csv` | Descriptive energy-quartile influence |
| `ten_factor_operators.csv` | Per-landmark operator restriction diagnostic |
| `ten_factor_sessions.csv` | Session-level ten-factor diagnostics |
| `ten_factor_summary.json` | Same-PCA and residual-row interpretation |
| `model_comparability.csv` | Matched and unmatched model dimensions |
| `parameter_counts.csv` | B3/B5 parameter counts and ratios |
| `decomposition_order_sensitivity.csv` | B4m/B1m/B0 ordered shares |
| `comparability_summary.json` | Arithmetic and interpretation |
| `sign_convention_decision.json` | Sign provenance and G4 downgrade |

## Presentation

| File | Purpose |
| --- | --- |
| `figures/kato_final_audit_summary.png` | Primary result and claims-tier count |
| `figures/kato_final_audit_uncertainty.png` | Blocks, sessions and horizons |
| `figures/kato_final_audit_decomposition.png` | Representation and ordered decomposition |
| `figures/layout_report.json` | Programmatic text-extent/overlap check for all three figures |
| `KATO_FINAL_AUDIT_MEMO.pdf` | Two-page supervisor memo |

## Standing labels

```text
PROBABILISTIC CALIBRATION NOT ESTABLISHED
MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.
Biological intervention validation: NOT_EVALUABLE.
```

The raw dataset is not committed. Large model checkpoints are not committed. The
five sessions are repeated recordings, not confirmed independent animals.
