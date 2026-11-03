# Scientific Studio user guide

## 1. Launch

From the repository root, in a fresh terminal:

```powershell
python -X utf8 -m streamlit run app/neurobranch_studio.py
```

Open the printed local URL. The app reads committed artifacts and the raw Kato
archive; it does not need a network connection. If the raw archive is missing,
the archive-verification error is explicit and the evidence routes fail closed.

The focused legacy demo remains available in a second terminal:

```powershell
python -X utf8 -m streamlit run app/handoff_demo.py
```

## 2. Navigation

The sidebar has seven routes:

1. **Overview** — the claim, its unit, and its boundaries.
2. **3D Neural Dynamics** — the PC1–PC3 state-space explorer.
3. **Neural Activity** — raw calcium traces and the linked PC1 time series.
4. **Models & Forecasting** — equations, configuration, forecasts, capacity.
5. **Uncertainty & Evidence** — intervals, horizon, calibration, claims.
6. **Virtual Perturbations** — a model counterfactual with support warnings.
7. **Research Notebook** — timeline, course, 17 errata, next experiments.

The sidebar also has **Presentation mode**, **Light mode**, **Advanced /
Legacy**, and **Source of truth**.

## 3. Shared selectors

Recording and forecast-origin selectors are shared across the 3D, activity,
models, evidence, and perturbation routes. Changing the origin in one route
updates the state used by the others.

Landmarks are frozen H20 evaluation origins. They are not arbitrary frames.
The selector shows the frame index and seconds from recording start. "Showcase"
marks the three precomputed origins per recording.

## 4. 3D explorer

- **Recording / Forecast origin** choose the case.
- **Displayed models** overlays any of B0, B1m, B3, B5 on the same projection.
- **Observed continuation** toggles the held-out measurement; it is never an
  input.
- **Sampled model paths** draws eight innovation draws; the heavy line is the
  deterministic mean.
- **History length** changes only how much measured history is visible.
- **Playback step** controls Plotly's client-side playhead speed.
- Drag to orbit, scroll to zoom, right-drag to pan.
- **Reset camera** resets the view key.
- The **2D fallback** tab uses the same coordinates and origin when WebGL is
  unavailable.
- **Download this view (HTML)** exports the current figure.

The Play/Pause control moves a marker along the observed continuation. It does
not regenerate the model. The forecast itself begins at `t+1`.

## 5. Activity view

- Toggle **Raw ΔF/F** vs the standardized z-score used by the models.
- Toggle **Named cells only** or order by variance.
- Select up to eight traces; the line chart shows the first four.
- The heatmap is display-downsampled; the stride is printed. The model never
  sees the downsample.
- Dotted vertical lines mark train/validation boundaries.
- **Open this origin in 3D** links the state-space view.
- The behaviour-label counts are provided with a warning: labels are derived
  from command/interneuron activity, not independent ground truth.

## 6. Models & Forecasting

- **Model cards** give the fitted equation and role.
- **Matched trajectories** show 2D, PC1-over-time, and 3D views from one origin.
- **Capacity & metrics** shows the per-landmark score table and the aggregate
  energy bars.
- **Historical rows** preserves J2/J5/B1/B4r for reproduction; they are not
  additional evidence.
- The refit protocol is stated: 500-frame trailing window, no future leakage,
  but an online-refit estimand that includes earlier test frames.

## 7. Uncertainty & Evidence

Five tabs:

- **Primary contrast** — recording forest, paired histogram, bootstrap table.
- **Horizon & proper scores** — H1 reversal, H5/H10/H20 profile, MSE null.
- **Calibration** — marginal coverage, not joint calibration.
- **Capacity & identifiability** — nominal parameter ratios, ordered 51/49
  decomposition, mechanism warning.
- **Claims ledger** — all 23 claims with status filters.

Read the interval labels. "Conditional landmark block-5" and "recording
resampling block-5" are different objects.

## 8. Virtual Perturbations

1. Choose a recording, origin, and native predictor coordinate.
2. Choose an edit in training z units.
3. The app fits B3 and B5 at that origin, rolls both forward under zero
   innovation, and shows baseline, perturbed, and difference paths.
4. Read the OOD warning and the B3/B5 disagreement.

This is not a real stimulation. It does not rank targets, estimate a biological
effect, transfer a cell identity across recordings, or establish a digital
twin. A label absent from the recording has no valid mapping and is refused.

## 9. Research Notebook

- **Research timeline** shows the 14 phases, including negative and withdrawn
  results. Historical phases may use other datasets; they are not pooled with
  Kato.
- **Master-review course** reads the existing 16-chapter course.
- **Documentation errata** lists all 17 corrections and their active-surface
  status.
- **Reproducibility** shows the source-of-truth hierarchy and commands.
- **Next experiments** lists the observational-boundary proposals.

## 10. Presentation mode

Ten slides for a methodological supervisor. Use **Next/Previous**; the progress
line names the slide. Each slide has **Presenter notes** with exact phrasing
and warnings.

- Slide 3 renders the 3D explorer. **Open the interactive 3D explorer** leaves
  presentation mode and opens the full route.
- **Exit presentation** returns to the Studio.
- The sidebar remains visible outside presentation mode; it is hidden during
  the deck.

## 11. Suggested eight-minute path

1. Overview: the question and the unit. (1 min)
2. 3D: state space is not anatomy. (1.5 min)
3. Models: four models, one contrast. (1 min)
4. Evidence: the result and both intervals. (1.5 min)
5. Horizon/MSE/calibration: the boundary. (1.5 min)
6. Capacity and claims: what is not identifiable. (1 min)
7. Notebook: next experiment and falsification. (0.5 min)

## 12. Keyboard and export

- Tab/Shift-Tab move through Streamlit controls.
- Plotly supports its modebar; the Studio hides lasso/select buttons that do
  not apply to 3D.
- Use the browser print dialog for a static capture, or the per-figure HTML
  download for an interactive share.
- No export writes into the frozen evidence directories.

## 13. Troubleshooting

| Symptom | Cause | Action |
| --- | --- | --- |
| "Trajectory artifact is missing" | `reports/scientific_studio/` not built | run `python -X utf8 scripts/studio_build_trajectories.py` |
| Archive hash mismatch | wrong or incomplete `WT_NoStim.mat` | restore the OSF file; do not proceed |
| 3D canvas blank | WebGL unavailable | use the 2D fallback tab |
| Perturbation unavailable | no fit at that origin or label absent | choose a saved landmark and a native label |
| Legacy demo unavailable | separate app | run `streamlit run app/handoff_demo.py` |
