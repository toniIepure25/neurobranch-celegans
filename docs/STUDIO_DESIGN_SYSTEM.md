# Scientific Studio design system

**Intent:** a scientific instrument, not a marketing dashboard. The interface
should make the inferential object easier to see and harder to overstate.

## 1. Principles

1. **Evidence before decoration.** Every prominent number has a source and an
   uncertainty statement.
2. **One visual grammar.** Observed data are cyan; model output is amber;
   controls/uncertainty are violet; baselines are grey; red marks a material
   warning, not a brand accent.
3. **Quiet hierarchy.** Large type is reserved for the claim and the origin.
   Secondary metadata is small and muted.
4. **The boundary is visible.** Warnings are not hidden in footnotes.
5. **No fake precision.** Show three decimals where the artifact supports it,
   and label the unit.
6. **No anatomy by implication.** 3D axes are labelled PC1/PC2/PC3 with
   explained variance; the non-anatomy warning is persistent.

## 2. Palette

| Role | Dark | Light |
| --- | --- | --- |
| Background | `#071117` | `#f5f7f6` |
| Panel | `#0d1e27` | `#ffffff` |
| Ink | `#e3eef0` | `#142226` |
| Muted | `#8ca4ad` | `#5f7278` |
| Observed | `#5eead4` | `#087f72` |
| Prediction | `#f4b860` | `#a96312` |
| Control/uncertainty | `#b9a2ff` | `#6447b8` |
| Danger | `#ff7a7a` | `#a52a2a` |

Dark is the default. Light mode is a real alternate surface, not a filter. The
toggle re-renders CSS and Plotly themes; it does not change data or selection.

## 3. Typography

- Family: Segoe UI Variable / Segoe UI / Inter / Helvetica Neue / Arial.
- Hero: `clamp(2.0rem, 3.4vw, 3.4rem)`, weight 730, tight tracking.
- Section title: 1.65rem, weight 700.
- Body: 0.9–1.0rem, line-height 1.55–1.65.
- Kicker: 0.72rem, uppercase, 0.16em tracking, cyan.
- Numeric: tabular via the plotting library; no fake monospaced prose.

## 4. Components

| Component | Use | Rule |
| --- | --- | --- |
| Hero | opening claim | one per page |
| Section header | route title + lead | number and boundary in the lead |
| Chip/badge | status, unit, interval | never the only carrier of meaning |
| Stat block | one headline quantity | always has a unit label |
| Panel | grouped interpretation | no decorative gradients inside data panels |
| Note | methodological nuance | amber left rule |
| Warning | material limit | red border; must be read before the chart |
| Caption | source and alignment | every figure has one |

HTML components escape user-supplied labels with `html.escape`. Scientific
labels from the artifacts are trusted, but badges still escape their text.

## 5. Plotly conventions

- Transparent paper/plot backgrounds; grid lines use the theme line colour.
- Observed history: solid cyan, width 3–4.
- Observed continuation: dotted cyan; labelled "evaluation only".
- Forecast mean: amber (B3) or violet (B5); baseline models grey and dashed.
- Sampled model paths: same hue at 0.17 opacity, no legend entries.
- Forecast origin: diamond marker, cyan, always visible.
- Ensemble paths start at the origin plus the forecast steps; there is no
  artificial gap.
- 3D axes: `aspectmode="data"`; fixed camera; `uirevision="studio-3d"`.
- Playback: client-side Plotly frames move a marker along the observed
  continuation. It is not a generative animation and it does not alter the
  model paths.
- All figure captions state the recording, origin, alignment, and what is
  observed vs predicted.

## 6. Layout and responsiveness

- Wide layout; content max width 1500px; 2.2rem side padding.
- Primary split: 2.35 / 1 for the 3D explorer; 1.55 / 1 for evidence.
- At 1366×768 the 3D canvas remains usable; the side panel stacks below.
- At 1280×800 no horizontal page overflow is expected; the app uses
  Streamlit's responsive columns.
- Tables scroll horizontally inside their container rather than widening the
  page.

## 7. Motion and interaction

- 3D orbit, zoom, pan; a reset button changes the chart key to reset the camera.
- Playback step is user-selectable (60–400 ms).
- Hover labels use PC values and seconds, never "neuron 1 → neuron 2".
- Transitions are limited to Plotly's built-in playback; no decorative
  auto-animation on load.
- A user can always stop motion with Pause.

## 8. Accessibility and honesty

- Do not rely on hue alone: line style and labels distinguish observed,
  predicted, and controls.
- Minimum body text 0.75rem for captions; primary text 0.9rem+.
- Error messages name the missing artifact and the command to rebuild it.
- Avoid `CONFIRMED` anywhere; the tier vocabulary is fixed by the claims
  ledger.
- Avoid words that imply anatomy, causality, or animal-level generalization in
  titles, axes, hover text, and downloads.

## 9. Anti-patterns

- A wall of KPI cards with no inferential context.
- A 3D plot without explained variance or a non-anatomy label.
- A "significant" badge based on a conditional interval.
- A calibration gauge built from marginal coverage.
- A perturbation result presented as a stimulation outcome.
- An animation that implies the model is generating the observed future.
- Dark mode used to make warnings less visible.
- Replacing exact numbers with vague "improved" language.
