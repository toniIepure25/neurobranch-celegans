# Phase 12R artifacts

Status: **not adopted**. The aligned rerun still fails SBC rank uniformity,
predictive median absolute error, and the negative control. The Phase 11
coverage-aware ceiling **0.560** remains the only descriptive result.

Input: `reports/phase3/randi_trial_table_phase3.csv.gz`, SHA-256
`8186c96bcb33cdce15dc9ae88691e09fc596e42c5513310b6d6a636e4d5dd515`.

Protocol: `research/phase12r_protocol.md`; deviations:
`research/phase12r_deviations.md`; correction: `docs/CORRECTIONS.md` C10/C11.

| Artifact | Content |
| --- | --- |
| `summary.json` | Primary fit, gates, decision, negative-control exclusions |
| `sbc.json` | 3,000 aligned SBC fits, coverage, rank, edge and fit integrity |
| `heldout_ppc.json` | 20 animal-level splits, 293 predictions |
| `negative_control.json` | Permuted-profile null, 48 eligible of 100 requested |
| `prior_sensitivity.json` | HalfNormal scale 0.5, 1.0, 2.0 |
| `hierarchical_by_target.csv` | Per-target fit, exclusion reason, variance floor |
| `coverage_sensitivity.csv` | Coverage 0.5, 0.8, 1.0 |
| `min_events_sensitivity.csv` | Primary `min_events=1` vs sensitivity 2 |
| `design_posterior.csv` | Animal-budget planning approximation (not a guarantee) |
| `inference_alignment_audit.json` | Baseline/model truth mismatch audit |
| `inference_validation.json` | Synthetic type-I, coverage and power check |

Reproduce:

```bash
python scripts/phase12r_hierarchy.py
python scripts/phase12r_inference_audit.py
python scripts/phase12r_manifest.py
python scripts/validate_reports.py --reports reports/phase12r
python scripts/audit_claims.py
```

`reports/phase12r_first_pass/` is the first complete full run, retained for
provenance. It differs from the final run only in a reporting-only
`exclusion_reasons` field added to the negative-control artifact; all estimates
and gate values are identical.
