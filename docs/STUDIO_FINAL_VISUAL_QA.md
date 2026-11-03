# Final visual and browser QA

**Date:** 2026-11-04  
**Decision:** `BROWSER_VISUAL_QA: INCOMPLETE`  
**Reason:** the review model cannot ingest images. Automated geometry, console,
and screenshot capture passed, but typography, contrast, 3D legibility, legend
placement, and framing were **not visually inspected** by the agent. A human
review is required before claiming visual PASS.

## 1. What was automated

Command:

```powershell
python -X utf8 -m streamlit run app/neurobranch_studio.py --server.headless true --server.port 8505
python -X utf8 scripts/studio_browser_qa.py
```

Environment:

- Streamlit 1.66.1, Plotly 7.1.0
- Playwright Chromium 1234 at
  `C:\Users\ComputaCenter\AppData\Local\ms-playwright\chromium-1234\chrome-win64\chrome.exe`
- viewports `1920×1080`, `1366×768`, `1280×800`
- dark mode plus one light-mode capture at 1920

The script visited all seven sections at all three viewports (21 captures), two
presentation slides at all three viewports (6 captures), and one light-mode 3D
capture: **28 captures**. It recorded:

| Check | Result |
| --- | ---: |
| Captures | 28 |
| Console errors | 0 |
| Horizontal document overflow | 0 |
| Document width equals viewport width | 28 / 28 |
| Files written | `reports/scientific_studio/screenshots/*.png` |
| Machine report | `reports/scientific_studio/screenshots/browser_qa.json` |

The report is geometry evidence, not a visual-quality judgment. A non-blank
screenshot does not prove that a label is readable, that the 3D projection is
legible, or that a legend does not obscure a trace.

## 2. Screenshot set

Section captures use `{section}_{width}.png` at the viewport size. Streamlit
keeps long sections in its own scroll container, so these are top-of-route
captures, not a stitched full document. A reviewer must scroll each route for
lower-page content; the old `*_top.png` / `*_bottom.png` pairs remain as
historical scroll coverage.

- `overview_1920.png`, `overview_1366.png`, `overview_1280.png`
- `3d_neural_dynamics_{1920,1366,1280}.png`
- `neural_activity_{1920,1366,1280}.png`
- `models_forecasting_{1920,1366,1280}.png`
- `uncertainty_evidence_{1920,1366,1280}.png`
- `virtual_perturbations_{1920,1366,1280}.png`
- `research_notebook_{1920,1366,1280}.png`

Presentation: `presentation_slide1_{width}.png`,
`presentation_slide6_{width}.png`, and `dynamics_light_1920.png`.

The older top/bottom captures remain in the directory as historical scroll
coverage; the new set adds current route/viewport captures but is not a
replacement for a scrolled human inspection.

## 3. Human-review contact sheets

Because the agent cannot view images, two contact sheets are provided for a
human reviewer:

- `reports/scientific_studio/final_audit/contact_sheet_sections_1920.png`
- `reports/scientific_studio/final_audit/contact_sheet_presentation.png`

They show the top of each viewport capture, labelled by route and viewport.
They are navigation aids, not evidence of visual quality.

## 4. Visual defects found by code inspection and fixed

- PCA percentages were unqualified. The 3D panel now says "of total training
  variance", gives the PCA-10-relative percentage, and states the unshown
  PCA-10 remainder.
- 3D and 2D axis titles now read `PC1 · 26.0% total` rather than a bare
  percentage.
- Capacity range and the ordered split were hard-coded prose; both are now
  derived from `capacity_audit.json`.
- Landmark and positive-session counts were hard-coded in several sections;
  they now derive from the metadata layer.
- The perturbation route lacked the exact required warning banner; it now
  renders the constant from `adaptive.var`.

These are content and wording corrections. They do not establish visual
readability.

## 5. Human visual checklist

A reviewer should open the screenshots or run the app and check:

1. At 1920, 1366, and 1280: no clipped headings, labels, badges, or controls.
2. Light mode: body text and muted captions meet contrast expectations.
3. 3D: axis ticks and titles are readable; the origin, cyan observed
   continuation, amber B3, and violet B5 are distinguishable; the legend does
   not cover the trajectory.
4. 3D animation: Play/Pause, frame slider, and playhead move along the observed
   continuation without a layout jump.
5. Forecast panel: the H20 badges and Δ caption remain legible at 1280.
6. Evidence: forest plot labels, zero line, and interval endpoints do not
   collide.
7. Perturbation: the exact warning is visible; the baseline, perturbed, and
   difference traces are distinguishable.
8. Presentation: slides 1 and 6 fit without vertical scrolling at 1280×800.
9. Loading and missing-artifact states fail loudly rather than displaying a
   stale result.

Until that review is recorded, **BROWSER_VISUAL_QA remains INCOMPLETE**.
