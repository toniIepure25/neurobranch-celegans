# Contributing

## Setup

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

Python 3.10-3.13. The project is validated on 3.13.5.

## Before opening a pull request

```powershell
python -m ruff check .
python -m mypy src/neurobranch/benchmark src/neurobranch/design
python -m pytest -q
python scripts/validate_reports.py --strict
python scripts/check_reproducibility.py
```

Or `make check` where `make` is available.

## Rules that are not negotiable

1. **No leakage.** Splits are animal-disjoint. A responder list, scaler or
   hyperparameter fitted on test animals invalidates the result.
2. **Match the estimand.** A pooled metric is bounded by a pooled ceiling; a
   per-animal claim by a single-animal ceiling. State which one you used.
3. **Report uncertainty.** Point estimates without intervals or MDEs are not
   accepted for a difference claim.
4. **No causal language.** Virtual perturbation is model-dependent. "Validated"
   is reserved for matched interventional evidence.
5. **No invented citations.** Verify DOI or PMID; record preprints as preprints.
6. **Negative results stay.** If a robustness check fails, report it and add it
   to `docs/LIMITATIONS.md`; do not delete the check.
7. **Regenerate, do not hand-edit, reports.** Artifacts under `reports/` come
   from scripts; manifests carry hashes.

## Adding a report

- Put the writer in `src/neurobranch/benchmark/reporting.py` or a script.
- Add a JSON schema under `schemas/` if the artifact is consumed downstream.
- Add a CSV contract in `scripts/validate_reports.py`.
- Add a test under `tests/`.
- Update `docs/EXPERIMENT_REGISTRY.md` and `docs/README.md`.
