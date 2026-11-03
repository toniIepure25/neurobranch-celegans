# Final supervisor checklist and 8-minute demo

## 1. Preflight

Run from the repository root:

```powershell
python -X utf8 -m pytest -q
python -X utf8 -m ruff check src/neurobranch/studio app/neurobranch_studio.py scripts/master_review/independent_model_check.py
python -X utf8 -m mkdocs build --strict
python -X utf8 -m streamlit run app/neurobranch_studio.py
```

Expected: full suite passes, lint clean, MkDocs strict succeeds, and the Studio
opens at the printed localhost URL. If the trajectory artifact is missing, run
`python -X utf8 scripts/studio_build_trajectories.py` first.

Before the meeting, verify:

- [ ] `git status --short` is clean.
- [ ] `reports/scientific_studio/kato_trajectories.npz` exists.
- [ ] `reports/scientific_studio/manifest.json` hashes match.
- [ ] `reports/scientific_studio/final_audit/independent_model_check.json`
      status is `PASS`.
- [ ] `reports/scientific_studio/screenshots/browser_qa.json` reports zero
      console errors and zero horizontal overflow.
- [ ] The exact perturbation warning is visible on the perturbation route.
- [ ] The 3D route renders; if not, use the 2D fallback and say why.

## 2. Eight-minute route

**0:00–1:00 — Overview.** Start with the unit, not the 3D plot.

> "This is five Kato recordings and 175 overlapping landmarks, not 175 animals.
> The question is whether a full population VAR retains predictive structure
> beyond a ten-factor common-state control. The answer is exploratory and
> horizon-dependent."

**1:00–2:00 — Provenance.** Point to the archive hash/size, the inferred animal
mapping, the reused test tail, and the claim status. Say explicitly that the
Studio is not a digital twin.

**2:00–3:30 — 3D dynamics.** Rotate once; press Play briefly; stop.

> "These axes are PC1–PC3 of a training-only PCA. They show about 52% of total
> variance, about 73% of the retained PCA-10 variance; the other 27% of PCA-10
> is not shown. A point is a population-state summary, not anatomy. Cyan is
> measured; amber and violet are forecasts from the same origin."

**3:30–4:30 — Models.** Show B0/B1m/B3/B5, the equations, parameter counts, and
the matched forecast. State that B3 has 7.96–9.95× the nominal coefficients of
B5-10, so the contrast is not capacity-matched.

**4:30–5:30 — Evidence.** Show the primary mean `+0.2644`, the conditional
interval, the recording interval that includes zero, the MSE null, and the H1
reversal. Say "exploratory", not "significant".

**5:30–6:30 — Perturbation.** Make a `+1.0` edit to a native coordinate at a
saved landmark. Read the exact banner. Say the first-step response is the
fitted `A` column; it is not stimulation. Point to OOD and B3/B5 disagreement.

**6:30–7:30 — Notebook / next experiment.** Show the 17 errata and the next
experiment: independent animals, documented IDs, pre-registered endpoint, and a
capacity-matched residual model.

**7:30–8:00 — Close.**

> "The defensible claim is a within-recording, landmark-level, H20 exploratory
> contrast under a training-only PCA. The recording-level interval includes
> zero. We are ready for methodological review, not confirmatory or biological
> claims."

## 3. Questions to invite

1. Is the five-recording resampling unit credible, and what would an independent
   animal cohort change?
2. Is the H20 endpoint pre-registered for the next study, or is it selected
   after seeing H1/H5/H10?
3. How should capacity be matched without destroying the B3/B5 contrast?
4. What sham-controlled perturbation would make the virtual lab falsifiable?
5. Which historical artifact still carries the strongest residual overclaim?

## 4. Final decisions

These decisions are separate. A visual-capture pass is not a visual-quality
pass.

| Gate | Decision | Basis |
| --- | --- | --- |
| `NUMERICAL_VALIDATION` | `PASS` | H100 reference matches frozen rows; H20 root cause reproduced; 11-case boundary check |
| `SCIENTIFIC_CLAIMS` | `PASS` | Active wording is exploratory and bounded; no causal/biological/digital-twin claim |
| `3D_VISUALIZATION_INTEGRITY` | `PASS` | PCA/axis semantics, trajectory alignment, deterministic means, no future input |
| `DOCUMENTATION` | `PASS` | Six final-audit documents, corrected PCA/reproducibility text, strict MkDocs |
| `BROWSER_VISUAL_QA` | `INCOMPLETE` | 28 captures, 0 console errors, 0 overflow; agent cannot view images, human inspection pending |
| `SUPERVISOR_DEMO_READINESS` | `READY_WITH_CAVEATS` | Numerical/scientific package ready; visual readability and five-animal inference remain caveats |

## 5. Stop conditions

Stop the demo and say so if:

- the 3D route is blank and the fallback also fails;
- a displayed number differs from the Evidence section;
- the perturbation banner is missing;
- the archive hash or manifest check fails;
- a question requires an animal-level or causal claim that the evidence does
  not support.

Do not fill a missing artifact with a newly fitted model during the demo.
