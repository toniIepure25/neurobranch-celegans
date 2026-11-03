# Scientific Studio architecture

**Status:** implemented, tested, and reviewed as a read-only presentation layer.
**Branch:** `research/neurobranch-scientific-studio`.
**Entry point:** `app/neurobranch_studio.py`.

## 1. Purpose and boundary

The Scientific Studio presents the audited Kato 2015 `WT_NoStim` reanalysis. It
answers one question: *does a full population VAR (B3) retain predictive
structure beyond a ten-factor common-state control (B5-10) at the H20 PCA-10
energy endpoint?*

The Studio does not fit new models for its inferential claims, select
architectures, tune hyperparameters, re-score the test tail, collect data, or
turn a model counterfactual into a biological intervention. The one on-demand
fit is the **virtual perturbation lab**, which fits B3 and B5 at the selected
origin to display a model-dependent counterfactual; it is explicitly excluded
from the evidence claims.

## 2. Layers

```
raw archive (data/raw/kato2015/WT_NoStim.mat)
        │  verified size + SHA256
        ▼
frozen evaluation artifacts
  reports/simulator/kato_common_state/
  reports/simulator/kato_final_audit/
  reports/master_review/
        │  read-only loaders
        ▼
compact Studio trajectory artifact
  reports/scientific_studio/kato_trajectories.npz
  reports/scientific_studio/kato_trajectories.json
        │  projection + presentation
        ▼
src/neurobranch/studio/
  paths.py          locations
  metadata.py       validated snapshot, primary aggregate
  trajectories.py   observed/predicted case access
  activity.py       lazy raw fluorescence access
  figures.py        Plotly builders
  components.py     HTML/CSS components
  theme.py          dark/light visual language
  errata.py         all 17 corrections
  course.py         existing 16-chapter course reader
  perturbations.py  on-demand model counterfactual
  controls.py       shared session/landmark/model state
  sections/         seven routes + presentation
        │
        ▼
app/neurobranch_studio.py (Streamlit shell)
```

### 2.1 Frozen evidence is the source of truth

The UI does not carry a hidden table of headline numbers. `metadata.py` reads:

- `reports/master_review/dataset_identity_audit.json`
- `reports/master_review/statistical_audit.json`
- `reports/master_review/quality_gate.json`
- `reports/master_review/claims_ledger.json`
- `reports/master_review/corrections.json`
- `reports/master_review/capacity_audit.json`
- `reports/master_review/cluster_bootstrap.csv`
- `reports/simulator/kato_common_state/model_per_landmark.csv`
- `reports/simulator/kato_common_state/model_config.csv`
- `reports/simulator/kato_common_state/session_effects.csv`
- `reports/simulator/kato_final_audit/uncertainty_summary.json`

The primary contrast is recomputed from the 175 paired rows and compared with
`statistical_audit.json`. A disagreement raises `ArtifactUnavailable`; the app
shows an error rather than a plausible-looking number.

### 2.2 Compact trajectory artifact

`scripts/studio_build_trajectories.py` refits the frozen models at each of the
175 H20 landmarks. For each landmark it:

1. samples 100 block-bootstrap innovations (`HORIZON = 100`);
2. rolls each model 100 steps;
3. stores the first 20 steps (`EVAL_HORIZON = 20`);
4. stores the observed continuation at `t+1 … t+20`;
5. recomputes the H20 energy at 30 showcase checks and compares it with
   `model_per_landmark.csv`.

The saved 100-member ensembles are retained only for the 15 showcase landmarks
(three per recording across five recordings). The artifact records 30 score
checks because B3 and B5 are both checked at each showcase landmark. Other
landmarks store eight sampled paths for display. No new score is used for
inference.

The artifact's score-validation tolerance is `2e-5`, not a new scientific
tolerance: it absorbs float32 storage and PCA sign/rounding. The observed
maximum was `1.2678e-6`.

### 2.3 Raw activity access

`activity.py` calls the same `load_nostim` loader used by the frozen pipeline.
It is lazy and cached. Display downsampling is isolated in
`rendering_downsample`; scores and model inputs never see the downsample.

### 2.4 Presentation shell

`app/neurobranch_studio.py` owns:

- page configuration and CSS injection;
- sidebar navigation and Advanced/Legacy links;
- session state for route, recording, landmark, models, light mode, and
  presentation index;
- error handling for missing artifacts.

The seven routes are thin renderers. They do not mutate frozen files and do not
write outside `reports/scientific_studio/` when the user downloads a view.

## 3. State and linking

Shared keys:

| Key | Meaning |
| --- | --- |
| `studio_section` | route id |
| `studio_session` | recording index 0–4 |
| `studio_landmark` | one of the saved H20 landmarks |
| `studio_models` | displayed models |
| `studio_light` | light-mode toggle |
| `studio_present` | presentation mode |
| `studio_present_index` | slide 0–9 |

The 3D explorer, activity view, model workbench, evidence view, and perturbation
lab all read the same recording and landmark origin. Changing the origin in one
route is visible in the others. This is real linkage, not a decorative hover.

## 4. Routes

| Route | Responsibility |
| --- | --- |
| Overview | claim, unit, three evidence categories, provenance |
| 3D Neural Dynamics | PC1–PC3 explorer, animation, 2D fallback |
| Neural Activity | raw traces, heatmap, split overlay, PC1 link |
| Models & Forecasting | equations, configuration, matched trajectories, capacity |
| Uncertainty & Evidence | forest, bootstrap, horizon, MSE, calibration, claims |
| Virtual Perturbations | model counterfactual, OOD and disagreement warnings |
| Research Notebook | 14-phase timeline, 16-chapter course, 17 errata, next steps |

Presentation mode is a separate ten-slide renderer with speaker notes. It
reuses the same snapshot and 3D figure builder.

## 5. Failure behavior

- Missing trajectory artifact: explicit error with the build command.
- Archive hash/size mismatch: the Studio refuses to render scientific claims.
- Landmark not in the saved H20 set: explicit unavailable message.
- Missing model score: displays an em dash/`None`, not zero.
- WebGL failure: the 2D fallback tab uses the same projection and origin.
- Missing raw data: activity and perturbation routes fail loudly; the evidence
  routes remain readable from frozen artifacts.

## 6. Non-goals

- No anatomical registration or connectome overlay.
- No direct connectivity claim from VAR coefficients.
- No animal-level generalization from overlapping landmarks.
- No capacity-independent B3 claim.
- No joint-calibration claim from marginal coverage.
- No real-stimulation interpretation of a virtual perturbation.
- No validated digital twin.
- No rewrite of frozen historical artifacts.

## 7. Extension points

Add a new **presentation** only if the evidence already exists in a frozen,
machine-readable artifact. Add its loader to `metadata.py`, its route to
`sections/`, and a test that recomputes the displayed number. Do not add a new
model or a new endpoint to make the interface more impressive.
