# Phase 12R results: aligned calibration rerun and withdrawn benchmark claim

Phase 12R was a preregistered correction of the Phase 12 calibration harness.
It asked whether the hierarchical measurement model passes its calibration
gates after the simulation, fit and truth use the same animal centering and
standardization. The answer is **no**. The Phase 11 coverage-aware **0.560**
ceiling remains the only defensible descriptive result, and no calibrated
hierarchical interval, target ranking, or animal-budget guarantee is published.

This is a rapid internal correction, not a systematic review and not a new
confirmation study. The protocol and every deviation were frozen and recorded
before the corrected run.

## 1. What was corrected

The Phase 12R plan initially suspected that the simulation path discarded known
cell sampling variances. Direct verification showed that this was a false alarm:
the SBC passed per-cell summaries directly to the fit, which preserved
`var / n`. The actual defects were:

- templates were already animal-centered and standardized, then centered again
  by the fit, while the SBC truth formula was uncentered;
- the prose said `min_events=2`, `min_animals=2`, and two responders, while the
  executed primary used `min_events=1`, `min_animals=6`, and three responders;
- 79% of observed cell variances are exactly zero because most cells contain one
  event, so an explicit `1e-8` floor dominates the likelihood.

Phase 12R corrected the centering/standardization alignment and reran the frozen
gates. The real-data point estimate did not change.

## 2. Primary estimate and calibration gates

| Quantity | Phase 11 reference | Phase 12R |
|---|---:|---:|
| Pooled R(22), 80% coverage | 0.3136 | **0.3035** |
| Pooled ceiling | **0.5600** | 0.5509 |
| Single-animal R | 0.034 | 0.0194 |
| Eligible targets | 126 | 126 |
| Difference from Phase 11 | -- | -0.0101 |
| Calibrated interval? | no | **no** |

| Gate | Requirement | Observed | Result |
|---|---|---:|---|
| G1a SBC coverage | 0.93--0.97 | **0.933** | pass |
| G1b SBC rank | KS p > 0.01 | **1.43e-06** | fail |
| G1c grid edge mass | < 0.05 | 0.0392 | pass |
| G1d fit integrity | failures <= 5% | 0.000 | pass |
| G2 PPC coverage | >= 0.90 | **0.904** | pass |
| G2 PPC median absolute error | <= 0.10 | **0.207** | fail |
| G3 negative-control median | <= 0.05 | **0.120** | fail |
| G3 negative-control q95 | <= 0.15 | **0.594** | fail |
| G4 prior-sensitivity range | <= 0.05 | 0.0031 | pass |

The aligned SBC used 1,000 replicates x 3 templates (3,000 fits), 2,000 draws on
an 80 x 80 grid. Coverage improved from Phase 12's 0.923 to 0.933, but rank
uniformity still fails decisively (mean rank 0.533, KS p 1.43e-06). The held-out
check used 20 animal-level splits and 293 predictions; coverage now passes, but
median absolute error is still over twice the gate. The negative control used
100 requested targets; 48 were eligible and 52 were excluded, all recorded as
`too_few_animals` in the negative-control artifact. Its positive null bias is
larger at q95 than Phase 12's, so the model can still manufacture apparent
reliability from coverage geometry and marginal structure.

## 3. Why the point estimate is fragile

The 0.304 estimate is conditional on the primary event filter and on the
variance floor:

- 32,427 of 41,086 observed cells (79%) have a zero within-cell variance and use
  the `1e-8` floor;
- the preregistered `min_events=2` sensitivity keeps 36 targets and gives pooled
  R(22) = **0.199** (ceiling 0.446);
- the estimate is within 0.05 of Phase 11, but that agreement does not rescue a
  failed interval or a filter-sensitive point estimate.

The `1e-8` floor is not a measured sampling variance. It is an explicit
approximation that makes the likelihood finite when a cell contains one event.
Any future model must estimate the measurement variance from repeated events or
from a hierarchical event-level model rather than treating the floor as known.

## 4. Benchmark claim withdrawn

A separate audit found that the Phase 12 animal-cluster benchmark merge paired
baseline and model rows by animal/target/neuron aggregates rather than by the
same events. In the saved predictions, baseline and model `y_true` differ for
**1,334 of 1,334** paired rows. The Phase 12 cluster comparison is therefore
invalid, not merely underpowered. The claim is withdrawn; leaderboard values
remain descriptive only. `paired_cluster_inference` now fails closed on
unaligned truth by default.

A small synthetic check of the remaining paired procedure (120 replicates, 200
animal-cluster bootstrap draws) found nominal 95% interval coverage of
**0.900--0.917** and null rejection rates of **0.075--0.083**. Those are
procedure diagnostics, not biological validation, but they show the procedure
is slightly anticonservative/undercovering. Together with the alignment failure,
they rule out any positive benchmark claim.

## 5. Decision

The frozen decision rule is not satisfied. Phase 12R therefore:

1. retains **0.560** as the only descriptive coverage-aware ceiling;
2. publishes no calibrated reliability interval;
3. publishes no target ranking or posterior animal-budget recommendation;
4. reports the failed gates, the variance-floor approximation, and the
   `min_events=2` sensitivity as limitations;
5. does not tune the prior, grid, filter, seed, or sample size to recover a pass.

## 6. Evidence and reproduction

Artifacts: `reports/phase12r/summary.json`, `sbc.json`, `heldout_ppc.json`,
`negative_control.json`, `prior_sensitivity.json`, `hierarchical_by_target.csv`,
`min_events_sensitivity.csv`, `inference_alignment_audit.json`,
`inference_validation.json`, and the manifest.

Protocol: `research/phase12r_protocol.md`. Deviations:
`research/phase12r_deviations.md`. Correction record: `CORRECTIONS.md` C10 and
C11.

Commands:

```bash
python scripts/phase12r_hierarchy.py
python scripts/phase12r_manifest.py
python scripts/validate_reports.py --reports reports/phase12r
python scripts/audit_claims.py
```

The input table is referenced by SHA-256 and is not redistributed. No new data
were acquired.
