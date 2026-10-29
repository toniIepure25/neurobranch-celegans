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

## What version 0.2.0 adds: bounded evaluation

Version 0.2.0 replaces the unstable split-half reliability ceiling with a
variance-corrected estimator, separates **within-animal repeat reliability**
from **across-animal single-profile reliability**, and attaches an explicit
status to every quantitative claim.

The result changes how the project's own models should be read:

| Quantity | Estimate | 95% interval |
| --- | ---: | --- |
| Repeat reliability (within animal) | 0.243 | [0.206, 0.279] |
| Single-animal reliability (held out) | 0.223 | [0.196, 0.251] |
| Repeat attenuation ceiling | 0.493 | - |
| Single-animal attenuation ceiling | 0.473 | - |

Both ceilings are near 0.5. The Phase 8 pair-level advantage (+0.0442,
CI [+0.0080, +0.0901]) has a reconstructed design MDE of 0.0587 against a
declared smallest effect of interest of 0.020, so it is classified
**underpowered**, not supported. The Phase 7 claim that CRO reaches 64% of the
ceiling is withdrawn; against the corrected ceiling it reaches about 21%.

Read the full audit in [`docs/PHASE9_RESULTS.md`](docs/PHASE9_RESULTS.md), the
limits in [`docs/MEASUREMENT_LIMITS.md`](docs/MEASUREMENT_LIMITS.md), and the
corrections in [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md).

## Install

Python 3.10-3.13.

```bash
python -m pip install -e ".[dev]"
```

Optional GPU extra: `pip install -e ".[dev,gpu]"`.

## Quick start

```bash
# Reproduce the measurement ceilings and the claims ledger
python scripts/phase9_reliability.py
python scripts/phase9_claims.py --check
python scripts/validate_reports.py --strict

# Or through the CLI
neurobranch benchmark reliability
neurobranch benchmark claims
neurobranch version

# Interactive app (14 tabs)
streamlit run app/streamlit_app.py
```

## Reproduce the phases

| Phase | Command | What it produces |
| --- | --- | --- |
| Data audit | `neurobranch data audit` | Kato archive verification |
| Phase 2-4 | `neurobranch train`, `simulate`, `branching` | Baselines, branching, intervention audits |
| Phase 6-8 | `python scripts/phase8_run.py` | CRO and adaptive-twin evaluation |
| Phase 9 | `python scripts/phase9_reliability.py` | Reliability ceilings, variance components, manifest |
| Phase 9 claims | `python scripts/phase9_claims.py` | `reports/phase9/claims.json` |

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
  benchmark/        reliability, power/MDE, claims ledger, report builders
  dynamics/         latent models, training, splits
  branching/        empirical and model-based branching analysis
  interventions/    virtual perturbation engine and safety gate
  twin/             adaptive twin and active probing
  cli/              Typer CLI
app/                Streamlit + Plotly interface (14 tabs)
scripts/            phase9_reliability, phase9_claims, phase9_figure, validate_reports, phase8_run
reports/            generated artifacts, including phase9/
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
