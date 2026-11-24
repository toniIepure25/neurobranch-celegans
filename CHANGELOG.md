# Changelog

## 2026-11-24 - Interventional consistency, Stage One

- Added `neurobranch.abstraction`, a self-contained instrument for measuring whether a
  macrostate abstraction commutes with the interventions of its micro system. It does
  not import the untracked P2-P8.5 `mechanistic` package, so it reproduces from a clean
  checkout.
- Added a 12-channel information contract with a JSON Schema; a comparison across
  different contracts is refused instead of quietly reported.
- Added the sup-based behavioural and dynamic commutation errors with an occupancy
  floor, a support guard (`insufficient_support`, never imputed) and a separate
  occupancy-weighted mean for continuous systems.
- Added three ground-truth families: an exactly lumpable chain with behaviour-only and
  intervention-only negatives, a ring state space with a history ladder and an
  identifiability witness, and a sampled rate circuit.
- Added `abstraction stage1` and `abstraction contracts` to the CLI. Nine frozen gates
  pass, including independence of the enumerating reference, agreement between two
  estimators, and determinism.
- Documented that the P2-P8.5 report directories and `src/neurobranch/mechanistic/`
  are untracked at the base commit; that gap is reported, not repaired here.

## 2026-11-16 - Decision Room v5

- Added an option atlas that shows the commitment, prerequisites and claim ceiling for every
  choice without recording an answer or changing the scope digest.
- Made the scope digest a copy-friendly code block.
- Added a deterministic decision packet ZIP: request brief, public-source audit, data request,
  gap matrix, current contract, JSON/Markdown transcription, schemas and a hash manifest. The
  manifest keeps `unconfirmed_transcription` and says the packet is not a signature.
- Changed the visible build label to `Prototype v5` and separated the six decisions with
  dividers for faster review.

## 2026-11-15 - Decision Room v4

- Added cross-decision consistency advisories: a causal object with no specified
  intervention target and a BCI/NeuroAI object with a worm endpoint now produce
  explicit blockers and a corrected first action.
- Export now stores the decision snapshot. Editing any decision after export shows a
  field-by-field before/after diff, blocks the digest stage, and tells the user to
  download a fresh transcription before requesting a signature.
- Contract Markdown escapes table delimiters and line breaks, and the contract card
  is now genuinely sticky and internally scrollable on tall screens; it becomes static
  below 900 px.
- Importing a digest-bound JSON restores the snapshot, so change detection survives
  a resume; legacy payloads without a digest do not pretend to be bound.

## 2026-11-14 - Decision Room v3

- Scope & decisions is now a two-column decision room: a readiness rail, per-option
  commitment/claim-ceiling guidance, a live contract (claim ceiling, blockers, first
  action, digest), and the existing six widgets with unchanged keys and types.
- Added a downloadable decision contract. Contract export records the exported digest;
  a later edit shows the stale-export state instead of hiding it.
- Signed-brief upload stores the audit status and digest in session state, so the
  readiness rail updates after parsing; signer identity remains `manual_required`.
- Overview now labels the build `Prototype v3 · decision room` and links directly to
  the room.

## Unreleased - core dynamics simulator research branch

A free-running simulator, latent comparison, branching evaluation and named-neuron
virtual-perturbation interface over the frozen Kato/Phase 2 checkpoints. The
scientific result is negative: no dynamical or branching advantage, no specific
perturbation, and no biological validation.

### Added

- `src/neurobranch/simulator/`: `NeuralSimulator`, typed `PerturbationSpec`,
  rollout ensembles, one-step/free-running/distribution metrics, train-only PCA
  and BunDLe encoders, reversal-outcome branching, and six-neuron perturbation
  sweeps.
- `scripts/simulator_{fit,evaluate,latent,branching,perturb,figures,gates,manifest}.py`.
- `schemas/simulator_gates.schema.json` plus packaged copy;
  `scripts/validate_reports.py --reports reports/simulator` validates it.
- `reports/simulator/`, `figures/simulator/`, model cards, gate report and
  manifest.
- `app/simulator_ui.py` and a fifteenth app tab, "Core Dynamics Simulator".
- `docs/NEURAL_DYNAMICS_MODEL.md`, `LATENT_SIMULATION_VALIDATION.md`,
  `BRANCHING_SIMULATION.md`, `NEURONAL_PERTURBATION_METHODS.md`,
  `SIMULATOR_RESULTS.md`, `SIMULATOR_LIMITATIONS.md`,
  `SIMULATOR_SUPERVISOR_BRIEF.md`, `QUESTIONS_FOR_DENIS.md`.
- Tests under `tests/simulator/` plus an app tab-count update.

### Results

- G1 pass; G2 fail 0/5; G3 fail 0/5; G4 pass (interface only); G5 fail; G6
  `not_evaluable`.
- Selected `mlp_residual`: validation one-step R² 0.7134, free-running test R²
  −0.5098, ΔMSE +0.8398 versus constant state, OOD 0.7903.
- VAR is the best descriptive free-running model (R² 0.5325, ΔMSE −0.0491, OOD
  0.0076) but was not selected.
- Simulated reversal entropy 0.000–0.107 versus observed 0.482–0.837; the
  marginal null is closer than the simulator in all five worms.
- No perturbation candidate exceeds the random-control 95th percentile (0.500);
  top AVEL inhibition mean |ΔTV| 0.146, 2/5 worms, OOD 0.989.
- PCA 3-D explained variance 0.386–0.531; BunDLe cross-seed CKA 0.102/0.284.

### Notes

- Every perturbation output carries
  `MODEL-DEPENDENT VIRTUAL PERTURBATION - NOT CAUSALLY VALIDATED.`
- Protocol deviations (constant-state baseline, branching state sampling, G5
  simplification, AIBL/AIBR pilot) are recorded in
  `docs/SIMULATOR_RESEARCH_PROTOCOL.md`.
- No release tag and no remote push; the branch preserves `e18139b`.

## 1.4.0 - 2026-11-01

Three-estimand reliability audit. The Phase 11 0.560 headline is **withdrawn as
a bound**: it is a historical extrapolated point estimate for 87 of 126 targets.
The audit is a fail-closed negative result: calibration and null gates fail, no
target ranking replicates, and no calibrated interval is published.

### Added

- `src/neurobranch/benchmark/variance.py`: `_reliability_at_n`,
  `estimand_table`, `_estimand_block`, `summarise_reliability_estimands`.
- `src/neurobranch/benchmark/robustness.py`: target-matrix delete-one-animal
  jackknife, `estimand` (`obs`/`fold`/`id`), coverage-aware animal jackknife and
  extrapolation checks.
- `src/neurobranch/benchmark/reliability_audit.py`: 2,000-draw animal Dirichlet
  bootstrap, 1,000-draw permutation/sign-flip nulls, parametric calibration,
  ranking stability and heavy-tailed stress test.
- `scripts/phase15_reliability.py`, `phase15_validate.py`, `phase15_manifest.py`,
  `phase15_card.py`, `phase15_calibrate.py`, `phase15_nulls.py`,
  `phase15_manuscript.py`.
- `schemas/phase15_summary.schema.json` plus packaged copy.
- `reports/phase15/`, `docs/PHASE15_RESULTS.md`, `docs/METHODS_PHASE15.md`,
  `research/phase15_protocol.md`, `research/phase15_deviations.md`, correction
  C15, manuscript and figures.

### Changed

- The status dashboard and app show `R_obs`, `R_22`, `R_id`, the decision and
  the audit gates instead of a single 0.560 ceiling.
- The claims ledger adds four Phase 15 claims as `not_supported`; 0.560 is no
  longer described as a bound anywhere current.
- Version bump to 1.4.0; CITATION and package metadata updated.
- Release workflow now runs tests and Phase 15 validation before building.

### Results

- `R_obs = 0.2606` [0.1782, 0.3430], ceiling 0.5104, 126 targets.
- `R_22 = 0.3006` [0.2087, 0.3924], ceiling 0.5482, 87/126 extrapolated.
- `R_id = 0.3501` [0.2020, 0.4982], ceiling 0.5917, 39 targets.
- V1-V0 deltas 0.0103 / 0.0131 / 0.0078 (pass).
- Calibration coverage 0.990 / 0.976 / 0.999 (fail; over-coverage).
- Null medians 0.096--0.137, `R_22` sign-flip q95 0.1641 (fail).
- Extrapolation MAE 0.204, bias +0.038 (fail); ranking Spearman 0.126,
  Jaccard 0.176 (fail).
- Decision `not_calibrated`; `R_22` is `counterfactual_only`.

### Fixed

- The Phase 11/12/13/14 texts that treated 0.560 as a bound now carry an
  explicit superseded-headline notice.
- The delete-one-animal interval is no longer described as calibrated; its
  over-coverage and the null bias are reported.
- A prototype BCa-on-LOO diagnostic was removed because LOO values are not a
  bootstrap distribution; the failed diagnostic is recorded in D1.

## 1.3.2 - 2026-10-31

Reference audit and a rejected exact zero/spike-and-slab model. The Phase 14A
audit supports the 0.560 descriptive ceiling under a robust one-event variance
treatment; Phase 14B fails 8/12 frozen gates and is not adopted.

### Added

- `src/neurobranch/benchmark/measurement.py`: `m4_spike_slab`, exact point mass
  at `sigma = 0`, `SpikeSlabPosterior`, and the spike/slab evidence calculation.
- `src/neurobranch/benchmark/measurement_calibration.py`: discrete-truth SBC,
  spike calibration, held-out PPC, null `p_zero`, and multi-`pi0` prior
  sensitivity.
- `src/neurobranch/reporting/status.py`: dashboard status loader with explicit
  `not_run`/`not_evaluable` handling and historical fallbacks.
- `scripts/phase14_reference_audit.py`, `phase14_measurement.py`,
  `phase14_manifest.py`, `phase14_validate.py`;
  `schemas/phase14_summary.schema.json` and packaged copy.
- `reports/phase14/`, `docs/PHASE14_RESULTS.md`, `docs/METHODS_PHASE14.md`,
  `docs/REFERENCE_AUDIT.md`, corrections C13/C14.

### Changed

- `cell_summaries` and `profile_components_by_target` accept optional
  `one_event_variance` / `event_variance`; defaults preserve V0 exactly.
- The Streamlit status tab now shows Phase 14/13 and labels Phase 12R/12 as
  historical fallbacks.
- The claims ledger adds `phase14-zero-inflated-model` as `not_supported`
  (12 claims total).
- Version bump to 1.3.2; Docker labels updated.

### Results

- V0 0.3136077 / 0.5600069 reproduced exactly; V1 0.3005503 / 0.5482247;
  V1h 0.3138911 / 0.5602599; V2 0.1753713 / 0.4187736 (36 targets).
- M4 real-data median R(22)=0.0543, median P(z=0)=0.3235.
- 8/12 gates failed: SBC rank p=3.35e-192, edge 0.0810, PPC coverage 0.7697 /
  MAE 0.3998, null q95 0.3535 and 0.4509, prior range 0.1220, reference gap
  0.2463, spike calibration 0.3561. Decision `not_adopted_gate_failures`.

### Fixed

- Phase 13 G6 is reclassified as a mismatched one-event variance comparison;
  the reference itself is robust under the Phase 14A audit (C13).
- The app no longer presents Phase 12R as the current measurement result.

## 1.3.1 - 2026-10-30

Projection-consistent reliability and event-level variance. The event-variance
model rejects the Phase 12R floor but still fails calibration; the decision is
`not_adopted_gate_failures` and 0.560 remains the only descriptive ceiling.

### Added

- `src/neurobranch/benchmark/measurement.py`: M0 reference, M1 projection-scaled
  known-variance and M2 event-variance models, including the scaled chi-square
  event-variance posterior.
- `src/neurobranch/benchmark/measurement_calibration.py`: SBC, within-animal
  permutation, sign-flip, prior sensitivity and event-variance recovery.
- `scripts/phase13_measurement.py`, `phase13_manifest.py`,
  `phase13_validate.py`; `schemas/phase13_summary.schema.json`.
- `reports/phase13/`, `docs/PHASE13_RESULTS.md`, `docs/METHODS_PHASE13.md`,
  correction C12, and Phase 13 manuscript tables/figure.
- Event-keyed paired inference with BCa intervals and fail-closed duplicate /
  unaligned-truth checks.

### Changed

- M2 lowers pooled R(22) from 0.3035 to 0.2003 (ceiling 0.4475), showing the old
  estimate was dominated by the one-event `1e-8` floor. M2 fails six frozen
  gates: SBC coverage 0.9244, SBC rank KS p=0.00306, PPC not run, negative
  control median 0.1485 (q95 0.4680), sign-flip median 0.1341 (q95 0.5796), and
  Phase 11 agreement (0.1137 away).
- M1 is nearly neutral for the point estimate (0.3138) but also fails SBC rank.
  The centering algebra was not the main driver.
- The claims ledger adds `phase13-event-variance-model` as `not_supported`.
- Version bump to 1.3.1; Docker labels updated.

### Fixed

- `TargetData` carries event counts and cell variances through normalization and
  target transformation.
- M0 reproduces the historical Phase 12R path exactly (0.3035306473018537).

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
