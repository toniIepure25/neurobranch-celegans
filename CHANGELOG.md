# Changelog

## 0.2.0 - Phase 9 bounded evaluation

### Added

- `neurobranch.benchmark.variance`: variance-corrected repeat and
  single-animal reliability, attenuation ceilings, variance components.
- `neurobranch.benchmark.power`: MDE, power, sample size, TOST equivalence and
  the fixed claim-status decision rules.
- `neurobranch.benchmark.claims`: generated claims ledger with evidence paths.
- `neurobranch.benchmark.reporting`: report builders and manifests.
- `neurobranch benchmark reliability|claims` CLI commands.
- JSON schemas for the reliability summary and claims ledger, plus
  `scripts/validate_reports.py`.
- GitHub Actions CI, pre-commit configuration, Phase 9 test suite.
- BoundedBench tab in the Streamlit app (14 tabs).
- `docs/PHASE9_ANALYSIS_PLAN.md`, `docs/PHASE9_RESULTS.md`,
  `docs/MEASUREMENT_LIMITS.md`, `docs/BENCHMARK_CARD.md`, `docs/CLAIMS.md`,
  `docs/CORRECTIONS.md`.

### Changed

- Version bumped to 0.2.0.
- README rewritten around the measurement ceiling, design resolution and claim
  statuses.
- Reliability and claims are now generated artifacts with input hashes, not
  hand-written numbers.

### Corrected

- Phase 7 normalised scores above 1 are withdrawn.
- "CRO reaches 64% of the ceiling" is withdrawn; the corrected share is about
  21%.
- Phase 8 pair-level advantage is reclassified as `underpowered` under the
  declared SESOI of 0.02.

### Not included

- The EIG design portfolio, simulation-based calibration and replay engine.
- Cross-cohort transfer (H9.5), deferred until a harmonised second cohort is
  available.
