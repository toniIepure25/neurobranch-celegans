# Phase 13 reports

**Decision:** `not_adopted_gate_failures` (6 of 11 frozen gates failed).
**Primary model:** M2 event-level variance; pooled R(22) = 0.2003, ceiling
0.4475. **Not calibrated; not a replacement for the 0.560 ceiling.**

## Files

| File | Contents |
|---|---|
| `summary.json` | Real-data fits, gates, decision, claim boundary |
| `sbc.json` | M2 SBC ranks, coverage, one-event stratum, event-variance coverage |
| `sbc_m1.json` | M1 SBC ranks and coverage |
| `negative_control.json` | Within-animal responder permutation null |
| `signflip_control.json` | Sign-flip null |
| `prior_sensitivity.json` | Pooled R(22) at prior scales 0.5, 1.0, 2.0 |
| `coverage_sensitivity.csv` | M2 pooled R(22) at coverage 0.5, 0.8, 1.0 |
| `min_events_sensitivity.csv` | M2 pooled R(22) at `min_events` 1 and 2 |
| `hierarchical_by_target.csv` | M2 per-target fit table (conditional, not a ranking) |
| `hierarchical_by_target_m0_reference.csv` | M0 historical reference |
| `hierarchical_by_target_m1_projection_scaled.csv` | M1 diagnostic |
| `hierarchical_by_target_m2_event_variance.csv` | M2 per-target fit table |
| `manifest.json` / `manifest.sha256` | Artifact hashes and provenance |
| `validation.json` | Schema, gate, and hash validation record |

## Important numbers

- M0 Phase 12R reference: 0.3035 (ceiling 0.5509), floor fraction 0.7892.
- M1 projection-scaled: 0.3138 (ceiling 0.5602), floor fraction 0.7892.
- M2 event-variance: 0.2003 (ceiling 0.4475), floor fraction 0.0000,
  event-variance median 0.6906.
- M2 SBC: coverage 0.9244, rank KS p = 0.00306, one-event stratum coverage
  0.9231, event-variance coverage 0.9356.
- M2 negative control: median 0.1485, q95 0.4680.
- M2 sign-flip: median 0.1341, q95 0.5796.
- Prior range: 0.00661.

## Reproduction

```bash
python scripts/phase13_measurement.py --out reports/phase13
python scripts/phase13_manifest.py --reports reports/phase13
python scripts/phase13_validate.py --reports reports/phase13
```

`--quick` is a debug pilot and is not decision evidence. The input table is
referenced by SHA-256 and is not redistributed.
