# Changelog

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
