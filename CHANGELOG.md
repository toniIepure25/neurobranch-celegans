# Changelog

## 1.3.0 - 2026-10-30

Aligned calibration correction, explicit variance-floor sensitivity and a
withdrawn benchmark claim. The hierarchical interval remains uncalibrated and
0.560 remains the only descriptive ceiling.

### Added

- `research/phase12r_protocol.md` and `research/phase12r_deviations.md`, frozen
  before the corrected run. The suspected variance-loss bug was tested and not
  reproduced; the actual defect was double centering/standardization plus an
  uncentered SBC truth.
- `scripts/phase12r_hierarchy.py`: aligned SBC/PPC/negative-control/prior run,
  coverage and `min_events=2` sensitivities, frozen gates and a fail-closed
  decision. `phase12r_manifest.py` writes artifact, input and protocol hashes.
- `scripts/phase12r_inference_audit.py` and
  `phase12r_inference_validation.py`: real paired-truth audit plus synthetic
  type-I/coverage/power check.
- `reports/phase12r/` and `reports/phase12r_first_pass/`; Phase 12R contract
  tests; `schemas/phase12r_summary.schema.json`.
- `docs/PHASE12R_RESULTS.md`, `docs/CURRENT_STATUS.md`, corrections C10/C11 and
  a one-page status link in the docs index.

### Changed

- Phase 12R supersedes the original Phase 12 calibration evidence. SBC coverage
  is 0.933, but rank uniformity fails (KS p=1.43e-06); PPC MAE is 0.207; the
  negative control returns median R(22)=0.120 and q95=0.594. Status
  `not_adopted_gate_failures`.
- 79% of observed cells use the `1e-8` variance floor; `min_events=2` gives 36
  targets and reliability 0.199. Reported as a limitation and predeclared
  sensitivity, not hidden.
- The Phase 12 benchmark cluster claim is withdrawn, not merely labelled
  underpowered. `paired_cluster_inference` fails closed on unaligned truth.
- Claims ledger updates the hierarchical interval notes and marks the benchmark
  claim `not_supported`.

### Fixed

- SBC templates, fit transform and truth now use the same centering.
- Negative-control artifacts record exclusion reasons.
- Phase 12R reports are validated against a dedicated schema and recomputed
  gates in `audit_claims.py`.

## 1.2.0 - 2026-10-30

Hierarchical measurement audit, animal-cluster inference and release hardening.
The calibration gates failed; 0.560 remains the descriptive ceiling.

### Added

- `neurobranch.benchmark.hierarchical`: additive hierarchical reliability model
  with responder-coverage filtering, animal centering, known cell sampling
  variance, a deterministic 2-D grid posterior and per-target posterior draws.
- `neurobranch.benchmark.hierarchical_calibration`: simulation-based
  calibration, animal-holdout predictive check, within-animal responder-label
  permutation negative control and prior-sensitivity audit.
- `neurobranch.benchmark.inference`: animal-cluster paired bootstrap, Holm
  adjustment, minimum detectable effect and fail-closed benchmark status.
- `neurobranch.design.posterior_design`: model-conditional reliability curve and
  animal-budget planning table.
- `scripts/phase12_hierarchy.py`, `phase12_benchmark.py`, `phase12_manifest.py`
  and `reports/phase12/` artifacts, including SBC/PPC/negative-control JSON,
  per-target posterior CSV, cluster benchmark CSV and manifest.
- `schemas/phase12_summary.schema.json` plus contract tests for Phase 12
  artifacts, cluster inference and the hierarchical model.
- Phase 12 app panel with hierarchical estimates, gate-failure count and
  manifest provenance.
- `docs/PHASE12_ANALYSIS_PLAN.md`, `docs/METHODS_HIERARCHICAL.md` and
  `docs/PHASE12_RESULTS.md`.

### Changed

- Version 1.2.0; citation metadata and benchmark card updated.
- Claims ledger adds `phase12-hierarchical-interval` (not supported) and
  `phase12-benchmark-cluster` (underpowered).
- Claims audit checks Phase 12 gate/status consistency, ceiling arithmetic and
  benchmark support conditions.

### Fixed

- Boolean serialization in Phase 12 summary artifacts.
- Phase 11 moment-reference column mapping in the Phase 12 artifact script.
- SBC centering now matches the real-data fit.

## 1.1.0 - 2026-10-29

Coverage-aware measurement ceilings and research-release hardening.

### Added

- `neurobranch.uncertainty`: two-way (animal x responder) percentile bootstrap
  with per-target draws, diagnostics (`bootstrap_mean`, `bias_vs_point`,
  `interval_status`) and an explicit failed-interval state.
- `neurobranch.benchmark.reliability.shrink_reliability`: empirical-Bayes logit
  shrinkage with a method-of-moments variance component.
- `neurobranch.benchmark.leaderboard`: fold-level descriptive leaderboard with
  paired deltas against a named baseline and no animal-level inference.
- Responder-coverage filtering in the variance components
  (`min_responder_coverage`) for profile components and animal/pooled
  reliability.
- `extrapolation_check` at 80% coverage, `animal_half_replication` with
  optional shrinkage, and a synthetic two-way coverage study.
- `scripts/phase11_sensitivity.py`, `phase11_benchmark.py`,
  `phase11_manifest.py`, `audit_claims.py` and `make_manuscript_pack.py`.
- Packaged JSON schemas under `neurobranch.schemas` plus `py.typed`, so a wheel
  install can validate reports without the repository root.
- `docs/PHASE11_ANALYSIS_PLAN.md`, `docs/PHASE11_RESULTS.md`,
  `docs/PHASE11_DEVIATIONS.md`, `docs/LICENSE_AUDIT.md`, `docs/SEARCH_LOG.md`,
  `research/evidence.csv`, `research/phase11_protocol.md` and a generated
  manuscript pack.
- MkDocs site with strict link checking and API reference.
- App provenance panel with card version, Git revision, dirty state and
  manifest hash; explicit fallback label for Phase 10/9 artifacts.
- CI matrix (Ubuntu/Windows, Python 3.10/3.13), scoped coverage, AppTest smoke,
  wheel installation smoke, schema-sync and claims-audit gates.
- Tag release workflow and a research container (`Dockerfile`).

### Changed

- The benchmark card default is the coverage-aware pooled ceiling **0.560** at
  80% responder coverage; the Phase 10 0.833 value is historical.
- `phase10-fold-ceiling` is `not_supported` as a general estimate and keeps only
  its historical descriptive estimate.
- CRO share is reported as about 17.6% of the coverage-aware ceiling (about
  11.8% of the Phase 10 sparse-coverage ceiling); the 64% and 21% claims remain
  withdrawn.
- Version bumped to 1.1.0.

### Corrected

- **C8:** the Phase 10 pooled ceiling is coverage-sensitive. Requiring a
  responder in at least 80% of a target's animals lowers pooled reliability
  from 0.693 to 0.314 and the ceiling from 0.833 to 0.560. The filter also
  changes the responder population, so 0.560 is not a bias-corrected 0.833.
- The Phase 11 extrapolation gate failed (MAE 0.210, bias +0.016), so the curve
  beyond observed animals is not validated.
- The two-way bootstrap interval [0.497, 0.583] excludes its point estimate
  0.314; it is not a calibrated confidence interval.
- Empirical-Bayes shrinkage did not recover target rankings; no ranking is
  published.

### Known limitations

- The two-way bootstrap is biased upward on sparse real data; only the failed
  diagnostic is reported, not a confidence interval.
- The synthetic coverage study (coverage 1.000, width 0.185) is conservative
  and does not validate the real-data interval.
- The derived Randi table is not redistributed and its upstream licence is
  unresolved; the Kato OSF project had no licence metadata when checked.
- The animal-budget calculator remains a planning approximation.
- No causal, digital-twin or foundation-model claim is made.

## 1.0.0 - 2026-10-29

Metric-aligned measurement ceilings and research-release hardening.

### Added

- `neurobranch.design`: ceiling-targeted animal budget calculator
  (`animals_for_reliability`, `design_table`) and a simulation coverage study.
- `neurobranch.benchmark.robustness`: animal jackknife, summary sensitivity and
  animal-half replication.
- Pooled reliability curve over evaluation-animal counts, with an explicit
  `extrapolated` flag.
- Claims schema v2: `estimand`, `n_eval_animals`, `family`, `p_value`,
  `p_holm`, and the `phase10-fold-ceiling` claim.
- JSON schemas v2 plus versioned v1 copies for historical reports.
- CSV report contracts in `scripts/validate_reports.py`.
- `scripts/check_reproducibility.py` and artifact hashes in the Phase 10
  manifest.
- `Makefile` targets for lint, type check, reports, validation and
  reproducibility.
- `docs/PHASE10_ANALYSIS_PLAN.md`, `docs/PHASE10_RESULTS.md`,
  `docs/DESIGN_ENGINE.md`.
- BoundedBench ceiling-curve chart in the Streamlit app.

### Changed

- **Breaking:** `reports/phase10/reliability_summary.json` is schema v2 and
  reports both single-animal and pooled ceilings.
- Reliability is now computed per facet, and the pooled ceiling is reported at
  the Phase 6-8 fold size (22 animals).
- `Claim` is mutable so the ledger can attach Holm-adjusted p-values after
  building the family.
- Version bumped to 1.0.0.

### Fixed

- Sparse-responder sampling variance divided by `n_animals**2` instead of the
  per-responder count squared.
- The signal component counted `interaction / A` twice.
- Jackknife return dictionaries were inconsistent after the interval change.
- Streamlit `use_container_width` deprecation in the app.

### Corrected

- Phase 9 applied the single-animal ceiling (0.473) to pooled Phase 6-8
  metrics. The metric-aligned pooled ceiling at 22 animals is 0.833 (C6).
- Phase 9 single-animal reliability 0.223/0.473 is superseded by
  0.162/0.403 after the sparse-responder and interaction fixes (C7).
- CRO's share of its metric-aligned ceiling is about 11.8%, not 64% and not
  21%.

### Known limitations

- Target-level reliability rankings do not replicate across animal halves
  (median Spearman 0.05).
- The animal jackknife interval does not reach nominal coverage in simulation;
  it is a diagnostic, not a calibrated confidence interval.
- The design calculator is a planning approximation, not a validated
  prospective tool.
- No causal, digital-twin or foundation-model claim is made.

## 0.2.0

- Variance-corrected reliability, bounded claim statuses, Phase 9 reports.
