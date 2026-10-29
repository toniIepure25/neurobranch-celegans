# Changelog

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
