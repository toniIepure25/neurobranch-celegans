# Human visual review checklist

**Status: PENDING_HUMAN_SIGNOFF.** The release agent cannot ingest images. The
automated browser run is geometry and console evidence, not aesthetic review.

## 1. Where to look

Screenshots:

`reports/scientific_studio/screenshots/`

Current viewport captures:

- `overview_{1920,1366,1280}.png`
- `3d_neural_dynamics_{1920,1366,1280}.png`
- `neural_activity_{1920,1366,1280}.png`
- `models_forecasting_{1920,1366,1280}.png`
- `uncertainty_evidence_{1920,1366,1280}.png`
- `virtual_perturbations_{1920,1366,1280}.png`
- `research_notebook_{1920,1366,1280}.png`
- `presentation_slide1_{1920,1366,1280}.png`
- `presentation_slide2_{1920,1366,1280}.png`
- `presentation_slide6_{1920,1366,1280}.png`
- `dynamics_light_1920.png`

Contact sheets:

- `reports/scientific_studio/release_candidate/contact_sheet_sections_1920.png`
- `reports/scientific_studio/release_candidate/contact_sheet_presentation.png`

The captures are viewport images. Streamlit scrolls long sections inside its own
container, so the reviewer must also scroll the live app for lower-page content.

## 2. Automated evidence to read first

- `reports/scientific_studio/screenshots/browser_qa.json`
- `reports/scientific_studio/release_candidate/browser_qa_summary.json`

Confirm:

- [ ] all expected captures exist;
- [ ] console error count is zero;
- [ ] horizontal overflow count is zero;
- [ ] presentation enters and advances through slides 1, 2, and 6;
- [ ] light mode captured.

These checks do **not** establish readability.

## 3. Per-viewport visual checks

For each of 1920, 1366, and 1280:

### Global

- [ ] No clipped title, badge, caption, or navigation label.
- [ ] Body and muted text have sufficient contrast in dark mode.
- [ ] Light mode is readable, not merely inverted.
- [ ] Long pages scroll without a fixed overlay hiding content.
- [ ] Sidebar selection is obvious and synchronized with the main panel.
- [ ] No placeholder, stale, or fallback data appears without a warning.

### Overview

- [ ] The unit of inference (five recordings, 175 landmarks, not 175 animals)
      is visible before any claim.
- [ ] “Exploratory” and the non-causal boundary are legible.
- [ ] The primary number and its interval are not visually over-prominent
      relative to the caveats.

### 3D Neural Dynamics

- [ ] Axis titles include `% total`.
- [ ] PC1/PC2/PC3 ticks are readable at 1280.
- [ ] Origin, measured history, held-out continuation, B3, B5, and sampled paths
      are distinguishable.
- [ ] The legend does not cover the trajectory or the origin.
- [ ] The 27%-of-PCA10 unshown text is visible.
- [ ] Rotation, zoom, Play/Pause, and slider work without layout jumps.
- [ ] The 2D fallback is clearly labelled and uses the same axes.
- [ ] No anatomical or neuron-location language appears.

### Neural Activity

- [ ] Raw and standardized traces are distinguishable.
- [ ] The selected origin is visible and aligned with the dynamics view.
- [ ] Long neuron labels do not overflow.
- [ ] Hover text is readable.

### Models & Forecasting

- [ ] B0/B1m/B3/B5 colors and labels are consistent.
- [ ] Equations are legible at 1280.
- [ ] Parameter counts and the 7.96–9.95× capacity range are visible.
- [ ] Forecast origin and H20 horizon are unambiguous.
- [ ] Historical rows are clearly historical, not current predictions.

### Uncertainty & Evidence

- [ ] The conditional and recording-resampling intervals are visually distinct.
- [ ] The zero line is visible and labelled.
- [ ] The recording interval crossing zero is not hidden by framing.
- [ ] Forest-plot labels and endpoints do not collide.
- [ ] MSE null and marginal-coverage caveats are readable.
- [ ] The table does not overflow horizontally.

### Virtual Perturbations

- [ ] The exact warning is present:
      `MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.`
- [ ] Baseline, perturbed, and difference traces are distinguishable.
- [ ] The selected neuron label matches the native coordinate.
- [ ] OOD and B3/B5 disagreement are visible.
- [ ] No target ranking or stimulation recommendation appears.

### Research Notebook

- [ ] The 17 errata and claim tiers are readable.
- [ ] Historical claims are visually separated from current claims.
- [ ] Long text blocks remain within the content column.
- [ ] The next-experiment section is reachable without excessive scrolling.

## 4. Presentation mode

- [ ] Slide 1 fits at 1280×800 without vertical clipping.
- [ ] Slide 2 and slide 6 fit at 1280×800 without vertical clipping.
- [ ] `Next →`, `← Previous`, and `Exit presentation` are visible and usable.
- [ ] Speaker notes are legible but visibly secondary.
- [ ] No interactive chart spills outside the slide frame.
- [ ] The slide counter matches the displayed slide.

## 5. Sign-off

Record:

- reviewer name;
- date/time;
- build HEAD;
- viewports inspected;
- defects found, with screenshot filename and slide/section;
- disposition: `PASS`, `PASS_WITH_CAVEATS`, or `FAIL`.

Do not change `HUMAN_VISUAL_SIGNOFF` in the release record without an actual
image review.

## 6. Known disposition

- Automated capture: see `browser_qa.json`.
- Human sign-off: **pending**.
- Current release decision: no visual PASS claim.
