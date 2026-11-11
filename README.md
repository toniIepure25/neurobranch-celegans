# NeuroBranch

An end-to-end, leakage-safe research platform for *C. elegans* whole-brain
population dynamics, latent branching, and **model-dependent virtual
perturbation** on real calcium imaging and optogenetic atlases.

> **Epistemic status.** Every intervention output is a model-dependent virtual
> perturbation, not a causally validated effect. The models are fitted to
> observational recordings; no animal was perturbed for this project. A
> visually attractive branch is not a bifurcation, and good observational
> prediction is not causal identification. The evidence gate in
> `interventions/safety.py` refuses to label any output as empirically
> validated without a matched, provenance-checked intervention dataset.

## NeuroBranch Scientific Studio (current surface)

The active Kato reanalysis is presented by a read-only, seven-section Studio:

```powershell
python -X utf8 -m streamlit run app/neurobranch_studio.py
```

It asks whether a full population VAR (B3) retains predictive structure beyond
a ten-factor common-state control (B5-10) at the H20 PCA-10 energy endpoint.

| Item | Value |
| --- | --- |
| Dataset | 5 Kato `WT_NoStim` recordings, 15,662 frames, 592 predictors |
| Paired landmarks | 175, overlapping |
| Primary contrast | B5 − B3 = **+0.2644** H20 energy (9.79%) |
| Conditional interval | `[+0.0750, +0.4031]` |
| Recording interval | `[−0.0408, +0.4793]` |
| H1 | reverses (−0.0637) |
| MSE | null (`[−0.1574, +0.2409]`) |
| Calibration | not established |
| Verdict | `READY_FOR_METHODOLOGICAL_SUPERVISOR_REVIEW; NOT_READY_FOR_CONFIRMATORY_OR_BIOLOGICAL_CLAIMS` |

The 3D view is a training-only PCA state space, **not anatomy**. Virtual
perturbations are model counterfactuals, **not stimulation**. Start with
[`docs/STUDIO_USER_GUIDE.md`](docs/STUDIO_USER_GUIDE.md),
[`docs/STUDIO_SCIENTIFIC_EVIDENCE.md`](docs/STUDIO_SCIENTIFIC_EVIDENCE.md), and
[`docs/STUDIO_SUPERVISOR_DEMO.md`](docs/STUDIO_SUPERVISOR_DEMO.md).

The sections below document earlier research branches and are retained as
historical context; they are not pooled with the Kato Studio result.

## Vienna Neuroinformatics Workbench (new prototype)

A separate, three-tier prototype for the University of Vienna Neuroinformatics
group lives in `src/neurobranch/bundlenet_sim/`:

```powershell
python scripts/fetch_vienna_artifacts.py
python -m streamlit run src/neurobranch/bundlenet_sim/app.py
```

- **Tier 1:** published BunDLe-Net 3-D latent manifold; held-out-worm metrics and
  a stable observed-transition walk (not a mechanistic simulator).
- **Tier 2:** precomputed NC-MCM cognitive states, transition matrices and Markov
  p-values; `do()`-style panels are explicitly observational.
- **Tier 3:** published 205-node premotor `A`/`d` fitted weights integrated with
  the paper's rate ODE; gap-junction `W` and neuron names are missing from the
  public export, so this is a sandbox, not a digital twin.

Start with [`docs/VIENNA_WORKBENCH.md`](docs/VIENNA_WORKBENCH.md), the
source-to-claim ledger in
[`docs/VIENNA_WORKBENCH_EVIDENCE.md`](docs/VIENNA_WORKBENCH_EVIDENCE.md), and the
one-page [`docs/VIENNA_WORKBENCH_HANDOFF.md`](docs/VIENNA_WORKBENCH_HANDOFF.md).

The gap-closure contracts and incoming-data gates are in
[`docs/GAP_CLOSURE_MATRIX.md`](docs/GAP_CLOSURE_MATRIX.md) and
[`docs/DATA_REQUEST_PACKAGE.md`](docs/DATA_REQUEST_PACKAGE.md). The current
intervention design report is simulated, not data:

```powershell
python scripts/power_intervention.py --reps 300
python scripts/validate_annotation.py --annotations data/incoming/annotation
python scripts/validate_intervention_trials.py --trials data/incoming/intervention/trials.csv
python scripts/analyze_intervention_trials.py --trials data/incoming/intervention/trials_with_outcomes.csv
```

## Core dynamics simulator (research branch)

The `research/core-dynamics-simulator` branch implements a free-running
population simulator over the frozen Kato checkpoints:

```text
x(t+1) = F_theta(x(t), history, u(t)) + epsilon(t)
```

It can roll out from an observed initial state, compare against constant-state
and one-step-persistence baselines, quantify reversal-outcome branching against a
marginal null, project through train-only PCA/BunDLe encoders, and apply an
explicit named-neuron virtual perturbation (`initial_state` or
`dynamical_input`). It is implemented, tested, executed and documented.

The result is negative and is not hidden:

- **G1 pass:** rollouts are finite, start at `x(0)`, and never read future
  observations.
- **G2 fail, 0/5 worms:** the selected `mlp_residual` model has the best
  validation one-step R² (0.7134) and the worst free-running test R² (−0.5098,
  MSE +0.840 versus constant state, OOD 0.79). VAR is the best descriptive
  free-running model (R² 0.5325, ΔMSE −0.0491) but was not selected.
- **G3 fail, 0/5:** simulated reversal outcomes are farther from observed
  conditional distributions than the marginal null. Observed data show an
  entropy **deficit** versus a block null, not an excess.
- **G4 pass (interface only):** named neurons can be perturbed with matched
  controls.
- **G5 fail:** no candidate exceeds the random-control 95th percentile; OOD is
  ~0.99 under perturbation.
- **G6 not_evaluable:** Kato is observational; no stimulation-response model.

Start with [`docs/SIMULATOR_RESULTS.md`](docs/SIMULATOR_RESULTS.md),
[`docs/SIMULATOR_LIMITATIONS.md`](docs/SIMULATOR_LIMITATIONS.md), and
[`docs/SIMULATOR_SUPERVISOR_BRIEF.md`](docs/SIMULATOR_SUPERVISOR_BRIEF.md).
Every perturbation output carries:
`MODEL-DEPENDENT VIRTUAL PERTURBATION - NOT CAUSALLY VALIDATED.`

## What version 1.4.0 adds: the 0.560 headline was extrapolation

Version 1.4.0 replaces the single "coverage-aware ceiling" with a frozen,
three-estimand audit. The Phase 11 headline `sqrt(R(22)) = 0.5600` is retained
only as a historical extrapolated point estimate; it is **not a bound**.

- Only **39 of 126** targets have at least 22 animals. For the other 87,
  `R(22)` is a design counterfactual, not an identified observed-data quantity.
- V1 observed-count reliability is **`R_obs = 0.2606`** (delete-one-animal 95%
  interval 0.1782--0.3430; `sqrt(R_obs) = 0.5104`). This is the descriptive
  observed-data quantity.
- The counterfactual is **`R_22 = 0.3006`** (interval 0.2087--0.3924;
  `sqrt(R_22) = 0.5482`), with 87 extrapolated targets.
- The identified subset (`n >= 22`) gives **`R_id = 0.3501`** (interval
  0.2020--0.4982) on 39 targets. It is a different population, not a replacement
  for `R_obs`.
- The animal-half extrapolation check fails (median absolute error **0.204**,
  bias **+0.038** against thresholds 0.10 / 0.05). Target rankings do not
  replicate (median Spearman **0.126**, 95% interval -0.021--0.247; top-10
  Jaccard **0.176**). No target ranking is published.
- The audit decision is **`not_calibrated`** unless the frozen calibration and
  null gates pass. Read `reports/phase15/summary.json`; do not select the
  convenient estimand.

Phase 14 remains a negative result: the exact zero/spike-and-slab model is
**rejected** with 8 of 12 frozen gates failed, and its real-data median 0.0543 is
prior-dominated, not a reliability estimate. The `min_events=2` filter gives
0.1754 on 36 targets; it is a different population.

See [`docs/PHASE15_RESULTS.md`](docs/PHASE15_RESULTS.md),
[`docs/METHODS_PHASE15.md`](docs/METHODS_PHASE15.md),
[`research/phase15_protocol.md`](research/phase15_protocol.md),
[`docs/PHASE14_RESULTS.md`](docs/PHASE14_RESULTS.md),
[`docs/CORRECTIONS.md`](docs/CORRECTIONS.md) C13--C15 and
[`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md).

The Phase 12 animal-cluster benchmark comparison remains **withdrawn**: baseline
and model `y_true` differ for 1,334 of 1,334 paired rows because the old merge
paired aggregates rather than the same events. It is not an underpowered
positive result. A synthetic check also found the remaining procedure undercovers
(0.900--0.917 at nominal 0.95). See
[`docs/PHASE12R_RESULTS.md`](docs/PHASE12R_RESULTS.md) and
[`docs/CORRECTIONS.md`](docs/CORRECTIONS.md) C10/C11.

## What version 1.1.0 adds: coverage sensitivity (historical)

The Phase 10 pooled ceiling is **not robust to sparse responder coverage**. A
responder observed in one animal contributes to the pooled profile but has no
between-animal variance to subtract, inflating apparent signal. The historical
Phase 11 table lowered the apparent ceiling from 0.833 to 0.560 at 22 animals:

| Minimum responder coverage | Pooled reliability | Ceiling |
| --- | ---: | ---: |
| 0.0 (Phase 10) | 0.693 | 0.833 |
| 0.5 | 0.414 | 0.643 |
| 0.8 | 0.314 | 0.560 (historical) |
| 1.0 | 0.207 | 0.455 |

Phase 15 changes the interpretation, not the arithmetic: 0.560 is an
extrapolated point estimate for the 80% population, not a bound. The Phase 11
extrapolation gate failed (median absolute error 0.210), and the two-way
bootstrap interval excluded its own point estimate (0.314 vs [0.497, 0.583]).
Neither is used as a calibrated interval. See
[`docs/PHASE11_RESULTS.md`](docs/PHASE11_RESULTS.md), correction C8, and the
Phase 15 correction C15.

## What version 1.0.0 added: metric-aligned evaluation (historical)

Version 1.0.0 attaches every number to the estimand it actually describes. The
Phase 9 audit correctly found that the atlas is measurement-limited for
individual animals; it then applied that single-animal ceiling to pooled
metrics, which was wrong. Phase 10 separates the two and fixes two estimator
defects (sparse-responder denominators and a double-counted interaction term).

| Quantity | Estimate | 95% interval | Ceiling |
| --- | ---: | --- | ---: |
| Repeat reliability (within animal) | 0.243 | [0.206, 0.279] | 0.493 |
| Single-animal reliability | 0.162 | [0.134, 0.191] | 0.403 |
| Pooled over 22 animals (Phase 6-8 fold, sparse coverage) | 0.693 | [0.653, 0.734] | 0.833 |
| Pooled over all 110 animals | 0.873 | [0.837, 0.909] | 0.934 |

The consequence is narrow and important. Claims about one animal are severely
bounded. Pooled claims have more headroom. Against the Phase 10 sparse-coverage
ceiling the CRO of 0.0984 is about **11.8%**; against the Phase 15 observed-data
ceiling (`sqrt(R_obs) = 0.5104`) it is about **19.3%**. The old Phase 7 "64% of
ceiling" claim remains withdrawn, and so is the Phase 9 replacement (21%).
Phase 15's 0.560 value is not used as a denominator because it is extrapolated.

The Phase 8 pair advantage (+0.0442, CI [+0.0080, +0.0901]) still has a design
MDE of 0.0587 against a declared smallest effect of interest of 0.020, so it
remains **underpowered**. Two Phase 10 robustness results are deliberately
negative: target-level reliability rankings do not replicate across animal
halves (median Spearman about 0.00), and the animal jackknife interval does not reach
nominal coverage in simulation. Neither is used as an inferential interval. The
delete-one-animal aggregate spans [0.165, 1.000], so the fold-level mean is
fragile to a single animal; that is reported rather than hidden.

![Historical Phase 10 reliability ceiling curve](reports/phase10/ceiling_curve.png)

Read the audit in [`docs/PHASE10_RESULTS.md`](docs/PHASE10_RESULTS.md), the
estimator corrections in [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md), the
limits in [`docs/MEASUREMENT_LIMITS.md`](docs/MEASUREMENT_LIMITS.md), and the
design calculator in [`docs/DESIGN_ENGINE.md`](docs/DESIGN_ENGINE.md).

## Install

Python 3.10-3.13.

```bash
python -m pip install -e ".[dev]"
```

Optional GPU extra: `pip install -e ".[dev,gpu]"`.

## Quick start

```bash
# Reproduce the metric-aligned ceilings, robustness checks and claims
python scripts/phase10_reliability.py
python scripts/phase10_figure.py
python scripts/phase10_robustness.py
python scripts/phase10_design.py
python scripts/phase10_claims.py --check
python scripts/validate_reports.py --strict

# Or through the CLI
python -m neurobranch.cli.main benchmark reliability
python -m neurobranch.cli.main benchmark claims
python -m neurobranch.cli.main version

# Interactive app (14 tabs)
streamlit run app/streamlit_app.py
```

## Reproduce the phases

| Phase | Command | What it produces |
| --- | --- | --- |
| Data audit | `neurobranch data audit` | Kato archive verification |
| Phase 2-4 | `neurobranch train`, `simulate`, `branching` | Baselines, branching, intervention audits |
| Phase 6-8 | `python scripts/phase8_run.py` | CRO and adaptive-twin evaluation |
| Phase 9 | `python scripts/phase9_reliability.py` | Historical single-animal audit |
| Phase 10 | `python scripts/phase10_reliability.py` | Metric-aligned ceilings, curve, components, manifest |
| Phase 10 robustness | `python scripts/phase10_robustness.py` | Summary sensitivity, half-split, jackknife |
| Phase 10 claims | `python scripts/phase10_claims.py` | `reports/phase10/claims.json` |

Every artifact directory carries a manifest with the input hash, git revision,
parameters and library versions.

## Evaluation contract

1. **Animal-disjoint splits.** Events from one animal never appear in both
   training and test.
2. **Ceiling before score.** Report reliability and the attenuation ceiling next
   to every held-out correlation.
3. **MDE before difference.** Report the minimum detectable effect next to every
   claimed difference.
4. **Fixed claim statuses.** `supported`, `not_supported`, `bounded_null`,
   `underpowered` - assigned by code, not by hand.
5. **No causal language.** Virtual perturbation is model-dependent; the word
   "validated" is reserved for matched interventional evidence.

These rules are implemented in `src/neurobranch/benchmark/` and documented in
[`docs/BENCHMARK_CARD.md`](docs/BENCHMARK_CARD.md) and
[`docs/CLAIMS.md`](docs/CLAIMS.md).

## Repository map

```
src/neurobranch/
  benchmark/        reliability, robustness, power/MDE, claims ledger, reporting
  design/           ceiling-targeted animal budgets and coverage study
  dynamics/         latent models, training, splits
  branching/        empirical and model-based branching analysis
  interventions/    virtual perturbation engine and safety gate
  twin/             adaptive twin and active probing
  cli/              Typer CLI
app/                Streamlit + Plotly interface (14 tabs)
scripts/            phase10_* (reliability, figure, robustness, design, claims), validate_reports
reports/            generated artifacts, including phase9/ and phase10/
docs/               phase reports, benchmark card, claims policy, corrections
schemas/            JSON schemas for generated reports
tests/              unit, integration, scientific and benchmark tests
```

## Datasets and primary sources

- Kato et al., *Cell* 2015, whole-brain calcium imaging:
  [`10.1016/j.cell.2015.09.034`](https://doi.org/10.1016/j.cell.2015.09.034)
- Randi et al., *Nature* 2023, optogenetic atlas:
  [`10.1038/s41586-023-06683-4`](https://doi.org/10.1038/s41586-023-06683-4)
- Baskoylu & Flavell, Dryad 2026:
  [`10.5061/dryad.w9ghx3g4v`](https://doi.org/10.5061/dryad.w9ghx3g4v)
- Kramer et al., *Nature Neuroscience* 2026:
  [`10.1038/s41593-026-02257-5`](https://doi.org/10.1038/s41593-026-02257-5)

Provenance and hashes are recorded in [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md).
The atlas is used under its source terms; check each archive before
redistribution.

## Testing and CI

```bash
pytest -q            # 147 tests
ruff check src tests scripts
```

GitHub Actions runs the suite on Python 3.10 and 3.13, lints, and validates the
Phase 9 reports against their JSON schemas.

## Limitations

- The atlas is measurement-limited: single-animal reliability is about 0.22.
- Reliability is estimated on the same atlas used for evaluation.
- No causal claim is made anywhere in this repository.
- The design engine (EIG portfolios, simulation-based calibration) is not
  implemented; the measurement audit had to come first.
- Phase 6-8 intervals use animal-level folds but the MDE reconstruction assumes
  independent units, so design resolution is approximate.

## License

MIT for the code. Datasets retain their original licenses.
