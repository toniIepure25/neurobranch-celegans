# Scientific Studio final user guide

## 1. Launch

From the repository root:

```powershell
python -X utf8 -m streamlit run app/neurobranch_studio.py
```

Open the printed localhost URL. The Studio is read-only over committed evidence;
it does not collect data or rewrite frozen results. If an artifact is missing it
shows an error and the rebuild command rather than a stale chart.

The legacy 22-tab app and focused handoff demo are still available under
**Advanced / Legacy** in the sidebar. They are not the audited Studio surface.

## 2. The seven sections

1. **Overview** — the claim, its unit, capacity boundary, and provenance.
2. **3D Neural Dynamics** — PC1–PC3 trajectory with observed history, forecast
   origin, held-out observed continuation, model means, optional sampled paths,
   and playback.
3. **Neural Activity** — raw ΔF/F and the standardized trace used by the models,
   linked to the selected origin.
4. **Models & Forecasting** — B0/B1m/B3/B5 model cards, equations, parameter
   counts, matched forecasts, capacity, and historical rows.
5. **Uncertainty & Evidence** — conditional, recording-resampling, and session
   intervals; per-landmark distribution; sensitivity checks.
6. **Virtual Perturbations** — a model counterfactual with OOD and
   model-disagreement warnings.
7. **Research Notebook** — the 16-chapter course, errata, claim ledger, and next
   experiments.

The left sidebar changes section and recording. Session and landmark selectors
are shared across routes. Changing a selector changes every linked view; it does
not refit the frozen study.

## 3. Reading the 3D view

- The axes are PC1, PC2, PC3 of a training-only ten-component PCA.
- The percentage on each axis is a fraction of **total training feature
  variance**. The panel also reports the fraction of **retained PCA-10**
  variance that the three axes show.
- PC1–PC3 show about 52% of total variance in recording 0, which is about 73%
  of the retained PCA-10 variance. About 27% of PCA-10 is not shown.
- A point is a population-state summary; distance is under the training
  covariance, not anatomical distance.
- Cyan is measured. Dotted cyan after the diamond is held-out measurement shown
  only for evaluation. Amber is B3; violet is B5. Sampled paths are draws from
  the frozen J5 innovation pool, not biological replicates.
- Playback moves a reading playhead along the observed continuation. It does not
  generate the trajectory.
- The 2D fallback uses the same scores and origin when WebGL is unavailable.

## 4. Models

| Label | Meaning |
| --- | --- |
| B0 | persistence |
| B1m | independent AR, 200-frame window |
| B3 | full population VAR, 500-frame ridge |
| B5 | own AR plus ten train-only PCA factors, zero-innovation factor path |

The primary contrast is `B5_common_k10 − B3_population`. Positive means B3 has
lower H20 PCA-10 energy. H20 means 20 forecast frames, not seconds. The 100-step
innovation rollout is scored at step 20. B3 has roughly 8–10× the nominal
coefficient count of B5-10, so the comparison is not capacity-matched.

## 5. Uncertainty and evidence

- The conditional block-5 interval resamples landmarks within the five
  recordings.
- The recording-resampling interval resamples the five recordings and is the
  relevant check for extending beyond them; it includes zero.
- The session t interval is a descriptive five-unit summary.
- Coverage is marginal per dimension, not joint 10-D calibration.
- MSE is a null result; lower energy does not imply a better conditional mean.
- The test tail was reused; this is exploratory.

## 6. Virtual perturbations

1. Choose a recording, saved landmark, and native predictor label.
2. Choose a non-zero edit in **training z units**. The input matrix is already
   train-standardized per recording, so adding one unit to the native
   coordinate is one training z unit at the origin.
3. The route fits B3 and B5 at that origin, applies the edit to the initial
   state, and rolls forward with zero innovation.
4. Read the first-step response as the fitted `A` column, not as a biological
   effect. OOD means the edited coordinate is outside the 0.5–99.5% training
   marginal.
5. The B3/B5 disagreement shows model dependence; it does not validate the
   edit.

The exact banner is **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY
VALIDATED.** Labels are not registered one-to-one across recordings. The lab
does not stimulate a neuron, rank targets, predict behaviour, or establish a
circuit.

## 7. Presentation mode

Click **Presentation mode** in the sidebar. Use **Next →** and **← Previous**.
The ten slides are a supervisor tour, not new inference. **Exit to Studio**
returns to the seven-section view. Slide 1 and slide 6 are captured at all three
viewports for the visual audit.

## 8. Troubleshooting

| Symptom | Meaning | Action |
| --- | --- | --- |
| `Required artifact is missing` | trajectory JSON/NPZ absent | run `python scripts/studio_build_trajectories.py` |
| `Perturbation unavailable` | no fit at origin, or label absent | choose a saved landmark and a native label |
| 3D blank | WebGL unavailable | use the **2D fallback · PC1–PC2** tab |
| Stale-looking screenshot | browser cache or stale file | rerun `scripts/studio_browser_qa.py` |
| Selector changed all views | shared session state is intentional | use **Reset view** in the 3D tab |

## 9. What not to say

Do not say the Studio proves a mechanism, a causal perturbation, a digital twin,
an animal-level effect, joint calibration, or a general forecasting advantage.
Say: exploratory, within-recording, landmark-level, H20, capacity-confounded,
and horizon-dependent.
