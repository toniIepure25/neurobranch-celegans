# Scientific Studio presentation polish

**Branch:** `research/neurobranch-presentation-polish`  
**Starting HEAD:** `0aceff522a2f7be4e650db8a51692f64c8e76975`  
**Scope:** presentation hierarchy, responsive layout, typography, navigation, and
supervisor-facing wording. No new dataset, model, fit, metric, forecast, interval,
or perturbation calculation was added.

## 1. Position

The Studio remains a read-only presentation layer over frozen Kato 2015
`WT_NoStim` artifacts. The numerical result is unchanged:

| Quantity | Value |
| --- | --- |
| B5−B3 H20 PCA-10 energy | `+0.2644379926683563` |
| B3 mean / B5 mean | `2.700149750441149` / `2.9645877431095045` |
| Relative to B3 | `9.7934565527394%` |
| Conditional interval | `[+0.0749929196, +0.4031164916]` |
| Recording-resampling interval | `[-0.0408009083, +0.4792966063]` (includes zero) |
| Positive recordings / sign-test p | `4/5` / `0.1875` |
| MSE difference and interval | `+0.0545043099`; `[-0.1574261907, +0.2409040856]` |
| Horizon contrast H1/H5/H10/H20 | `-0.0636694611` / `+0.0835331933` / `+0.1471216612` / `+0.2644380` |
| Coverage B3 / B5 | `0.9382857143` / `0.7245714286` |
| Capacity ratio | `7.9583952452–9.9485924113` |

The 3D view remains a train-only PCA projection with ten components and
`random_state=17`. It is not an anatomical reconstruction. PC1–PC3 account for
about 52% of total variance and about 73% of PCA-10 variance.

## 2. Slide-2 root cause and fix

### Root cause

The old `present.py` opened an HTML `<div class="nb-slide">` inside one
`st.markdown` call, rendered the body and metrics in later Streamlit calls, and
emitted the closing `</div>` in a final call. Streamlit closes each Markdown
document independently, so the opening element was closed before the body was
rendered. The visible `.nb-slide` shell had a `520 px` minimum height and
contained only the slide number and title. The body began around `y ≈ 651` and
the KPI cards around `y ≈ 785`, outside the dark card. The large empty card was
therefore a DOM-structure bug, not a spacing preference.

### Fix

The slide now uses a real Streamlit container:

```python
with st.container(key="nb_slide"):
    ...
```

`key="nb_slide"` produces the stable DOM class `st-key-nb_slide`. Slide header,
one-line message, body, metrics, panels, charts, status strip, and navigation all
render inside that container. There is no unclosed wrapper HTML.

At 1920×1080 the repaired slide-2 shell measures `671 px`; at 1366×768 and
1280×800 it measures `617 px`. In the final browser run every audited content
element is inside the shell, the navigation bottom is inside the viewport, and
there is no clipping or horizontal overflow.

## 3. Ten-slide audit and changes

| Slide | Purpose | Change in this pass |
| --- | --- | --- |
| 01 | Bounded framing | Two balanced “What this is / What it is not” panels plus a test-tail status strip; removed the stacked chip wall. |
| 02 | What was measured | Rebuilt around three aligned KPI cards (`Neural recordings`, `Frames`, `Included predictors`), with provenance and cohort-reading panels. This is the structural fix above. |
| 03 | 3D state space | Training-only PCA explanation moved next to the figure; chart reduced to `270 px`; variance and “not anatomy” captions retained. |
| 04 | Models | Four model roles in a 2×2 panel grid; the reported-contrast equation is displayed once, directly above the result. |
| 05 | Primary result | Metrics and a two-column result block: forest plot left, interval/MSE reading right. The redundant body paragraph was removed; the one-line message already carries direction and the zero-crossing caveat. |
| 06 | Horizon dependence | Four horizon metrics plus horizon and MSE panels; H1 reversal remains visible. |
| 07 | Calibration and capacity | Coverage/capacity metrics with an explanation panel; no implied capacity match. |
| 08 | Claim boundary | Explicit “Can defend / Cannot defend” columns and a release-scope strip. |
| 09 | Falsification | Four testable failure modes, each with a concrete falsifier. |
| 10 | Next experiment | Animal-level replication and mechanism deferred to a new study; no new analysis promised as completed. |

The slides deliberately do not share one identical grid. Text-first, metric-first,
figure-first, and two-column slides use different compositions.

## 4. Typography and responsive behavior

At 1920×1080:

- slide title: `36 px`;
- one-line message: `19.2 px`;
- body: `16.96 px` (approximately 17 px);
- captions: `14 px`.

At 1366×768 and 1280×800:

- slide title: `29.6 px`;
- message and body: `17 px`;
- captions: `14 px`.

The short-viewport rule reduces gaps and padding, not text below 17 px. Charts
use explicit presentation heights (`trajectory_3d` `270 px`; `evidence_forest`
`230 px`) so the shell and navigation fit a 768 px-high viewport. Navigation is
inside the slide shell and stays visible; it is not a fixed overlay.

## 5. Overview and global surfaces

- Hero reduced to three chips: `5 recordings · 15,662 frames`,
  `1 experimental cohort`, `H20 PCA-10 energy`.
- Question now asks whether the result survives “beyond a ten-factor
  common-state control at H20”.
- One prominent `EXPLORATORY` status replaces a wall of equal-weight caveats.
- Sidebar badges read `5 RECORDINGS`, `1 EXPERIMENTAL COHORT`, `EXPLORATORY`.
- Footer states that no data were collected, no additional test-tail model
  optimization was performed for this release, earlier phases reused the
  evaluation tails, and the findings remain exploratory.
- The exact perturbation warning is unchanged:
  `MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.`

## 6. Browser evidence

Run:

```powershell
python -X utf8 scripts/studio_presentation_polish_qa.py --url http://127.0.0.1:8501
```

Final run:

- 10 slides × 3 dark viewports = 30 slide captures;
- 10 light-mode slides at 1920×1080;
- 7 sections × 3 viewports = 21 section captures;
- presentation enter/exit at each viewport;
- `0` console errors;
- `0` horizontal overflows;
- `0` content elements outside the slide shell;
- `0` off-viewport navigation controls;
- `0` clipped title, message, body, panel, or metric elements.

Evidence:

- `reports/scientific_studio/presentation_polish/after/browser_polish_qa.json`
- `reports/scientific_studio/presentation_polish/after/slide_01_1920.png` …
  `slide_10_1920.png` (and `_1366`, `_1280`, `_light_1920` variants);
- `reports/scientific_studio/presentation_polish/after/sections/`;
- `reports/scientific_studio/presentation_polish/after/contact_sheet_all_slides_1920.png`;
- `reports/scientific_studio/presentation_polish/after/contact_sheet_all_slides_light_1920.png`;
- `reports/scientific_studio/presentation_polish/after/contact_sheet_slide2_before_after.png`;
- `reports/scientific_studio/presentation_polish/after/contact_sheet_overview_before_after.png`.

Before captures are in
`reports/scientific_studio/presentation_polish/before/`.

Interaction run:

```powershell
python -X utf8 scripts/studio_interaction_qa.py --url http://127.0.0.1:8501
```

It selects `Recording 2`, changes the forecast landmark, changes the
history-length slider from 160 to 180 frames, confirms the same recording and
landmark persist in Neural Activity, switches dark to light mode, renders the 3D
canvas, and enters/exits presentation mode. Result:
`reports/scientific_studio/presentation_polish/after/interactions/interaction_qa.json`
→ `PASS`, no console errors.

These are geometry, console, and interaction checks. They are not human visual
inspection and do not establish aesthetic quality.

## 7. What is not claimed

- No new scientific result.
- No confirmatory or prespecified status.
- No animal-level inference; the animal mapping is inferred and not
  authoritative.
- No causal perturbation.
- No capacity-matched conclusion.
- No human visual sign-off.
- Historical provenance remains `INCOMPLETE`; frozen manifests were not rewritten.

Human review is tracked in `docs/STUDIO_VISUAL_SIGNOFF.md`.

## 8. Verification status

| Gate | Result |
| --- | --- |
| Full pytest | `528 passed` |
| Ruff | PASS |
| MkDocs strict | PASS |
| `studio_release_checks.py` | `status=PASS`, exit `0` |
| Presentation browser QA | PASS: 0 console errors, 0 overflow, 0 outside-shell elements, 0 off-viewport navigation, 0 clipped elements |
| Interaction QA | PASS |
| Current Studio manifest | regenerated after the content commit; see `reports/scientific_studio/manifest.json` |
| Historical provenance | `INCOMPLETE` — two historical audit-script snapshots remain unrecoverable; no unexplained scientific artifact change |
| Human visual sign-off | `PENDING` |

The historical provenance result is deliberately not converted to PASS. The
frozen historical manifests and the two unverifiable entries were not rewritten.

