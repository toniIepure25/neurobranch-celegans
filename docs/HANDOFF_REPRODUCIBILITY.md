# Reproducibility and handoff checks

This page is the operational handoff. It records what can be rerun, what cannot,
and which checks define "green" for this branch.

## 1. Branch and history

| Item | Value |
| --- | --- |
| Repository | `D:\ComputaCenter\NeuroBranch\neurobranch-celegans` |
| Handoff branch | `research/supervisor-handoff` |
| Branch point | `10694a80568c06e6180f6af08bd1a0f2c28e4237` |
| Previous branch | `research/joint-dynamics-calibration` |
| History policy | additive commits only; no amend, rebase, force-push, reset or squash |

The final commit hash is in `git log -1` and in the handoff response. It is not
written into this file because a file cannot contain the hash of the commit that
creates it.

## 2. Environment

- Python 3.13.5
- NumPy, SciPy, pandas, scikit-learn
- matplotlib, Plotly, Streamlit
- pytest, Ruff, MkDocs
- CPU only; no GPU is required for the handoff artifacts

The joint-dynamics runs used single-thread BLAS for determinism:

```powershell
$env:OMP_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:NUMEXPR_NUM_THREADS="1"
```

## 3. One-command rebuild of the handoff artifacts

```powershell
python scripts/build_supervisor_handoff.py
```

This reads the frozen joint-dynamics tables, writes:

- `reports/simulator/supervisor_handoff/summary.json`
- `reports/simulator/supervisor_handoff/phase_timeline.csv`
- `reports/simulator/supervisor_handoff/claim_ledger.csv`
- `reports/simulator/supervisor_handoff/provenance_audit.json`
- `reports/simulator/supervisor_handoff/figures/*.png`
- `reports/simulator/supervisor_handoff/manifest.json` (written by default; refresh
  with `python scripts/build_supervisor_handoff.py --manifest-only`)

Then build the PDF memo and presentation HTML/PDF:

```powershell
python scripts/build_handoff_pdf.py
```

Outputs:

- `reports/simulator/supervisor_handoff/NeuroBranch_Handoff_Memo.pdf`
- `reports/simulator/supervisor_handoff/NeuroBranch_Handoff_Presentation.html`
- `reports/simulator/supervisor_handoff/NeuroBranch_Handoff_Presentation.pdf`

The editable sources remain:

- `docs/HANDOFF_RESEARCH_MEMO.md`
- `docs/HANDOFF_PRESENTATION.md`

Do not edit the PDF or HTML by hand. Edit the Markdown and rebuild.

## 4. Frozen joint-dynamics pipeline

These commands regenerate the primary evidence. They are expensive and are not
required merely to view the demo.

```powershell
python scripts/jd_reproduce_historical.py
python scripts/jd_evaluate.py --members 100 --stride 10
python scripts/jd_calibration.py
python scripts/jd_parameter.py
python scripts/jd_summarize.py
python scripts/jd_figures.py
python scripts/jd_examples.py
python scripts/validate_jointdyn_reports.py
```

Expected reproduction check:

- maximum relative difference `5.5e-9` for the historical block model;
- exact reproduction of P0/P6 lead-1 scores;
- 175 landmarks, 100 members, H20 primary.

Do not rerun the model-selection search to "improve" the result. That would reuse
the test split again and invalidate the claim boundary.

## 5. Validation gates

### 5.1 Handoff validator

```powershell
python scripts/validate_handoff_reports.py
```

Checks:

- nine required `HANDOFF_*.md` documents exist and contain the standing warning;
- `summary.json` central values match `score_summary.csv` and `paired_scores.csv`
  within a fixed tolerance;
- `phase_timeline.csv` has the expected workstreams and statuses;
- every figure is a valid PNG;
- `manifest.json` hashes match the files;
- the PDF and presentation HTML exist;
- the PDF page counts and figure signatures are correct;
- the demo app and its data dependencies exist;
- forbidden positive claims do not appear as assertions.

### 5.2 Joint-dynamics validator

```powershell
python scripts/validate_jointdyn_reports.py
```

Expected: 64 artifacts verified, no schema or hash mismatch.

### 5.3 Existing report validators

```powershell
python scripts/validate_reports.py --strict
```

### 5.4 Tests

```powershell
python -m pytest -q
```

The baseline before this handoff was 380 passing tests. Handoff tests add checks
for the nine documents, summary provenance, manifest integrity, figure signatures
and the focused demo app.

### 5.5 Lint

```powershell
ruff check .
```

`ruff format --check .` reports many pre-existing unformatted files. It is not a
gate for this branch; do not reformat the repository wholesale.

### 5.6 Documentation

```powershell
mkdocs build --strict
```

### 5.7 Dashboard smoke test

The main dashboard must still import and render:

```powershell
python -m pytest tests/app/test_app.py -q
```

The focused demo has its own AppTest:

```powershell
python -m pytest tests/app/test_handoff_demo.py -q
```

## 6. Artifact manifest

`reports/simulator/supervisor_handoff/manifest.json` contains SHA256 hashes for:

- the compact summary, timeline, claim ledger and provenance audit;
- all handoff figures;
- the PDF memo, presentation HTML and presentation PDF;
- the nine handoff documents;
- the referenced joint-dynamics source tables and gates.

It excludes raw data, checkpoints, the full 1.2 MB per-landmark score table, and
the 8.2 MB visualization NPZ. The NPZ is excluded entirely; it is documented here
and loaded optionally by the demo, so its absence cannot invalidate the compact
manifest.

To verify manually:

```powershell
python scripts/validate_handoff_reports.py --check-hashes
```

## 7. Data availability

| Data | Status | Consequence |
| --- | --- | --- |
| Kato `WT_NoStim.mat` | present locally, not committed | full joint pipeline can run |
| Kato `WT_Stim.mat` | absent locally; historical pilot artifacts present | external pilot cannot be rerun without re-download |
| Randi raw archive | not committed; parsed tables present | intervention analyses can be inspected, not rebuilt from raw |
| Baskoylu raw data | not committed; derived reports present | behavioural analyses can be inspected, not rebuilt |
| Kramer raw data | not committed; derived reports present | turn-direction reconstruction can be inspected, not rebuilt |
| Model checkpoints | excluded from Git | demo uses tables/NPZ, not checkpoints |

The absence of `WT_Stim.mat` is a handoff limitation, not a claim that the
historical pilot never happened. See
[HANDOFF_METHODS_AND_LIMITATIONS.md](HANDOFF_METHODS_AND_LIMITATIONS.md) §1.1.

## 8. Determinism and known non-determinism

- Joint-dynamics ensembles are seeded and reproducible to machine precision on
  the same environment.
- BLAS threading can change floating-point last digits; use the single-thread
  environment for exact reproduction.
- Bootstrap intervals are seeded; changing the seed changes the interval at the
  fourth decimal, not the decision.
- The historical Phase 6 selection used CPU PyTorch and is not bitwise identical
  across PyTorch versions.
- The compact NPZ stores only 10 of 100 members as float32 for visualization; it
  is not the source of the scores.

## 9. Handoff checklist

- [ ] Branch is `research/supervisor-handoff`.
- [ ] `git status --short` is clean after commits.
- [ ] `python -m pytest -q` passes.
- [ ] `ruff check .` passes.
- [ ] `mkdocs build --strict` passes.
- [ ] `python scripts/validate_reports.py --strict` passes.
- [ ] `python scripts/validate_jointdyn_reports.py` passes.
- [ ] `python scripts/validate_handoff_reports.py --check-hashes` passes.
- [ ] Focused demo launches on port 8502 and all sections render.
- [ ] Main dashboard still renders 56 tabs.
- [ ] PDF memo is two pages.
- [ ] Presentation HTML opens and the figures render.
- [ ] No raw data or checkpoints are committed.
- [ ] All perturbation text carries the standing warning.
- [ ] Biological validation is labelled `NOT_EVALUABLE`.

## 10. If something fails

1. Do not modify frozen reports to make a check pass.
2. Identify whether the failure is in the handoff artifact or the source report.
3. If the source report changed, rebuild the handoff summary and manifest.
4. If only a link or heading changed, rerun MkDocs strict.
5. If a numerical check fails, compare `score_summary.csv` and
   `paired_scores.csv` before touching the validator tolerance.

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**
>
> Reproducing the pipeline reproduces a model-dependent counterfactual, not a
> biological stimulation. Biological validation is `NOT_EVALUABLE`.
