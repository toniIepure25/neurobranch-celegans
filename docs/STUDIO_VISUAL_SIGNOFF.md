# Human visual sign-off

**Status: PENDING_HUMAN_SIGNOFF.**  
The release agent can measure DOM geometry, console output, and navigation, but
cannot ingest images or judge aesthetics. Automated browser evidence is not a
substitute for this sign-off.

## 1. What the automated run establishes

`reports/scientific_studio/presentation_polish/after/browser_polish_qa.json`
records:

- 10 slides at 1920×1080, 1366×768, and 1280×800 (30 dark captures);
- 10 light-mode slides at 1920×1080;
- all 7 sections at all 3 viewports;
- presentation enter/exit at all 3 viewports;
- `0` console errors;
- `0` horizontal overflows;
- `0` content elements outside the slide shell;
- `0` off-viewport navigation controls;
- `0` clipped title, message, body, panel, or metric elements.

`reports/scientific_studio/presentation_polish/after/interactions/interaction_qa.json`
records a passing interaction run: shared recording and landmark selection,
history-length slider change, 3D canvas render, light/dark switch, and
presentation enter/exit.

These checks do **not** establish:

- that the composition looks balanced to a human;
- that contrast is comfortable rather than merely non-overflowing;
- that chart labels and legends are aesthetically placed;
- that the scientific emphasis feels right to a supervisor;
- that projection text is readable from the back of a room.

## 2. Review assets

Open, in this order:

1. `reports/scientific_studio/presentation_polish/after/contact_sheet_slide2_before_after.png`
2. `reports/scientific_studio/presentation_polish/after/contact_sheet_overview_before_after.png`
3. `reports/scientific_studio/presentation_polish/after/contact_sheet_all_slides_1920.png`
4. `reports/scientific_studio/presentation_polish/after/contact_sheet_all_slides_light_1920.png`
5. Full-size slides in
   `reports/scientific_studio/presentation_polish/after/`
6. Section captures in
   `reports/scientific_studio/presentation_polish/after/sections/`
7. Interaction captures in
   `reports/scientific_studio/presentation_polish/after/interactions/`

The contact sheets are navigation aids. The reviewer must open full-size images
for label, legend, and contrast judgments.

## 3. Slide-by-slide sign-off

Mark each item only after looking at the 1920×1080 and 1366×768 captures. Use the
1280×800 capture for the narrowest layout.

### Slide 01 — bounded framing

- [ ] The two framing panels read as a pair, not as unequal leftovers.
- [ ] “Five recordings, one experimental cohort” is visible before any result.
- [ ] The test-tail status strip is legible and not mistaken for a result.

### Slide 02 — what was measured

- [ ] The dark card contains the body, all three KPI cards, and both lower panels.
- [ ] There is no large empty area under the title.
- [ ] `5`, `15,662`, and `592` align as a KPI row.
- [ ] The provenance panel does not compete with the KPI row.
- [ ] The cohort-reading panel makes the unit of inference clear.

### Slide 03 — 3D state space

- [ ] Axis titles include `% total`.
- [ ] The PCA-not-anatomy caption is visible without scrolling.
- [ ] Origin, measured history, held-out continuation, B3, B5, and sampled paths
      are distinguishable.
- [ ] The legend does not cover the trajectory or origin.
- [ ] Rotation, zoom, and the 2D fallback work without layout jumps.

### Slide 04 — models

- [ ] The 2×2 model grid reads in a sensible order.
- [ ] The reported-contrast equation is legible at 1280.
- [ ] B0/B1m/B3/B5 labels and colors are consistent.
- [ ] The capacity caveat is visible on the same slide as the comparison.

### Slide 05 — primary result

- [ ] The three metrics are the first visual anchor.
- [ ] The forest plot and interval panel read as one unit.
- [ ] The conditional and recording intervals are visually distinct.
- [ ] The zero line is visible and labelled.
- [ ] The recording interval crossing zero is not visually buried.
- [ ] Removing the extra paragraph did not make the slide feel under-explained.

### Slide 06 — horizon dependence

- [ ] H1 reversal is visible, not hidden by the H20 framing.
- [ ] The four horizon metrics align.
- [ ] The MSE “not established” message is readable.
- [ ] The slide does not imply a monotone or uniform advantage.

### Slide 07 — calibration and capacity

- [ ] Coverage and capacity values are legible.
- [ ] The explanation panel does not read as an excuse.
- [ ] The capacity ratio is not mistaken for a matched design.

### Slide 08 — claim boundary

- [ ] “Can defend” and “Cannot defend” are visually balanced.
- [ ] The release-scope strip is legible.
- [ ] No item in “Can defend” overstates the recording-level evidence.

### Slide 09 — falsification

- [ ] All four falsifiers are readable at 1366.
- [ ] Each falsifier names an observable outcome, not a vague future hope.
- [ ] The panels do not imply these tests were already run.

### Slide 10 — next experiment

- [ ] Animal-level replication is the first requirement.
- [ ] Mechanism is clearly deferred to a new study.
- [ ] No completed analysis is implied by the proposed design.

## 4. Global checks

- [ ] Navigation (`Previous`, progress, `Next`) is visible on every slide at all
      three viewports.
- [ ] No navigation control overlaps content.
- [ ] Presentation enter and exit are obvious.
- [ ] Light mode is readable, not merely inverted.
- [ ] Captions are at least 14 px and do not look like fine print.
- [ ] Body text at 1366 is 17 px; it should not feel cramped.
- [ ] No horizontal scrollbar appears.
- [ ] Long labels in charts do not collide.
- [ ] The exact warning
      `MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.`
      is present in Virtual Perturbations.
- [ ] Historical errata remain visually distinct from active claims.

## 5. Sign-off record

Complete only after the review:

| Field | Value |
| --- | --- |
| Reviewer | |
| Date | |
| Viewports reviewed | |
| Slides accepted | |
| Slides requiring change | |
| Overall decision | `ACCEPT` / `ACCEPT_WITH_CHANGES` / `REJECT` |
| Notes | |

Until this table is completed, the release status is
`HUMAN_VISUAL_SIGNOFF: PENDING`, regardless of automated geometry results.
