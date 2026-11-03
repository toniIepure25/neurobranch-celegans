# Testing and quality assurance

## 1. Scope

Testing covers four layers:

1. **Artifact contract** — hashes, shapes, alignment, horizon, scores.
2. **Scientific arithmetic** — the primary contrast and intervals are
   recomputed, not typed.
3. **UI contract** — course, errata, badges, themes, figure builders.
4. **Application smoke** — the Streamlit app renders with no exception through
   the AppTest harness.

Browser QA adds DOM geometry, interaction checks, console-error capture, and
viewport-specific screenshots. It does not replace numerical tests.

## 2. Automated tests

| File | Coverage |
| --- | --- |
| `tests/test_studio_artifacts.py` | dataset SHA256/size, primary contrast, 175 landmarks, session identities, H100/H20 contract, observed/forecast alignment, finite scores, storage validation, missing-artifact failure |
| `tests/test_studio_ui_contract.py` | 16-chapter course, all 17 errata, no CONFIRMED tier, HTML escaping, dark/light CSS, 3D frames, 2D fallback, figure HTML |
| `tests/test_studio_app_smoke.py` | all seven routes, light mode, and Presentation through `AppTest` |
| `tests/test_studio_docs.py` | nine required documents, current-status links, verified archive size, mkdocs nav, relative links, all 17 errata |

Commands:

```powershell
python -X utf8 -m pytest -q tests/test_studio_docs.py tests/test_studio_artifacts.py tests/test_studio_ui_contract.py tests/test_studio_app_smoke.py
python -X utf8 -m pytest -q
python -X utf8 -m ruff check src/neurobranch/studio app/neurobranch_studio.py scripts/studio_build_trajectories.py scripts/studio_manifest.py
python -X utf8 -m mkdocs build --strict
```

## 3. Recorded results

**2026-11-04, repository checkout on `research/neurobranch-scientific-studio`.**

| Check | Result |
| --- | --- |
| Studio docs + artifact + UI + app tests | **37 passed** |
| Full pytest | **494 passed** in 64.4 s |
| ruff (Studio surface) | all checks passed |
| mkdocs strict | built in 8.1 s |
| 3D render, 1920×1080 | no exception, no console errors, 3 Plotly figures |
| all seven routes | no exception, no horizontal overflow |
| presentation slides 1 and 3 | no exception |
| 1366×768 | no exception, `scrollWidth == clientWidth` |
| 1280×800 | no exception, `scrollWidth == clientWidth` |
| linked 3D → activity | route changed and origin preserved |
| recording selector 0 → 3 | origin reset to a valid landmark, no exception |
| light mode | CSS and Plotly theme switched; caption contrast 4.67:1 |

Update this table only from an actual command output. Do not claim a pass from a
partial run.

## 4. Artifact-specific checks

### Horizon

The artifact metadata records `rollout_horizon = 100` and
`evaluation_horizon = 20`. Tests assert the stored truth and means are
`(20, 10)`. The 100-member ensemble is stored only at the 15 showcase
landmarks; the test asserts `(100, 20, 10)` there and eight paths elsewhere.
The UI labels both correctly.

### Alignment

For a non-showcase landmark, the test verifies:

```python
np.testing.assert_allclose(case["truth"], observed[t + 1 : t + 21], atol=1e-6)
```

and checks that forecast time is strictly after origin time. The text "aligned
at t+1" is therefore a tested claim, not a caption convention.

### Scores

All 175 rows for B0, B1m, B3, B5 are required to be finite. The 30 showcase
score checks are within `2e-5`; the maximum is `1.2678e-6`. This is a storage
check, not a new inferential threshold.

### Dataset

The test recomputes SHA256 from `WT_NoStim.mat` and requires
`7531811c…a0098a` and size 56,390,975. A mismatch is a test failure, not a
warning.

## 5. Browser QA protocol

Environment: Playwright, Chromium 1234 at
`C:\Users\ComputaCenter\AppData\Local\ms-playwright\chromium-1234\chrome-win64\chrome.exe`.
The expected bundle `chromium_headless_shell-1248` is absent; the QA command
passes the working executable path explicitly.

For each route:

1. load `http://127.0.0.1:<port>`;
2. click the exact sidebar label;
3. wait for the Plotly canvas/table;
4. assert `[data-testid='stException']` count is 0;
5. capture console errors;
6. measure `documentElement.scrollWidth` vs `clientWidth`;
7. screenshot the full page.

Routes: Overview, 3D Neural Dynamics, Neural Activity, Models & Forecasting,
Uncertainty & Evidence, Virtual Perturbations, Research Notebook.

Additional interaction checks:

- open 3D, press Play/Pause, confirm no exception;
- switch to 2D fallback;
- change recording and landmark;
- toggle Light mode;
- open Presentation mode, advance to slide 3, return;
- click linked-view buttons from 3D.

## 6. Screenshot index

Screenshots are written to `reports/scientific_studio/screenshots/`:

- `overview_1920_top.png`, `overview_1920_bottom.png`
- `dynamics_1920_top.png`, `dynamics_1920_bottom.png`
- `activity_1920_top.png`, `activity_1920_bottom.png`
- `models_1920_top.png`, `models_1920_bottom.png`
- `evidence_1920_top.png`, `evidence_1920_bottom.png`
- `perturbations_1920_top.png`, `perturbations_1920_bottom.png`
- `notebook_1920_top.png`, `notebook_1920_bottom.png`
- `dynamics_light_1920.png`, `overview_light_1920.png`
- `presentation_slide1_1920.png`
- `presentation_slide3_1920.png`
- `dynamics_1366.png`
- `overview_1280.png`

The manifest lists each file with bytes and SHA256 at handoff.

## 7. What automated QA cannot establish

- **Visual aesthetics.** Geometry and DOM checks catch clipping/overflow, not
  whether a chart is legible to a human. A final human inspection is required.
- **WebGL on every user machine.** The 2D fallback is the mitigation.
- **Scientific validity.** Tests verify that the UI reproduces the audited
  artifacts. They cannot make the design identified or the test tail fresh.
- **Streamlit version drift.** The AppTest harness is version-sensitive; the
  pinned environment is recorded in `STUDIO_REPRODUCIBILITY.md`.
- **Presentation delivery.** The speaker notes are checked for presence, not
  for delivery quality.

## 8. Regression policy

Any change to a headline number must update:

1. the frozen artifact (only with an explicit new audit), or
2. `metadata.py` if the source-of-truth mapping changes;

and must update the corresponding test. Never edit the UI copy to a new number
without a recomputation path.

Any change to the 3D projection must update the explained-variance labels, the
non-anatomy warning, and `STUDIO_3D_SCIENTIFIC_MAPPING.md`.

Any change to perturbation behavior must update the model-counterfactual
warning and the "does not rank targets" text.

## 9. Final gate

The release is ready for handoff only when:

- the full pytest suite passes;
- ruff passes on the Studio surface;
- `mkdocs build --strict` passes;
- all seven routes and presentation render without an exception;
- screenshots exist and the manifest hashes match;
- the Git worktree is clean;
- the final report states the exact HEAD.
