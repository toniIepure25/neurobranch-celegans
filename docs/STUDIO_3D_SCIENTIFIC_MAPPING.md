# 3D scientific mapping

**What the flagship view is:** a PC1–PC3 projection of a 10-dimensional,
training-only PCA of population calcium activity, with observed history,
held-out observed continuation, and model forecasts aligned at one origin.

**What it is not:** an anatomical reconstruction, a connectome, a physical brain
space, a causal graph, or a generative movie of the animal.

## 1. The projection

For each recording:

1. Fit `PCA(n_components=10, random_state=17)` on the training split only.
2. Transform every frame of that recording, including validation and test.
3. Store the first ten scores in `reports/scientific_studio/kato_trajectories.npz`.
4. Plot PC1, PC2, PC3.

Recording 0's training variance fractions are:

| Axis | Fraction |
| --- | ---: |
| PC1 | 0.25950035 |
| PC2 | 0.17802279 |
| PC3 | 0.08242325 |
| Cumulative PC1–PC3 | 0.51994640 |

The axis labels in the UI are computed from the artifact, not typed. A point is
a population state summary; distance in this space is distance under the
training covariance, not anatomical distance. A sign flip of a component would
not change the fit; the stored projection is fixed for reproducibility.

The four session-specific projections mean the axes are not shared across
recordings. The UI never overlays two recordings in one 3D scene. When the
recording selector changes, the variance labels and all coordinates change.

## 2. Time alignment

For a saved landmark `t`:

- forecast origin = observed frame `t`;
- observed continuation = frames `t+1 … t+20`;
- model forecast step 1 = frame `t+1`;
- time axis = the recording's actual `tv` seconds;
- no future observation is an input to any model.

The artifact builder verified that `truth[i]` equals
`observed[t+1 : t+21]` after projection. The test suite verifies this on a
non-showcase landmark.

The performance endpoint is H20, computed from the 100-step innovation
rollout's first 20 steps. "H20" here means 20 forecast frames, not 20 seconds.
At ~2.8–3.1 Hz, that is about 6.5–7.1 seconds.

## 3. What is drawn

| Trace | Meaning | Not |
| --- | --- | --- |
| Observed history | measured PCA scores before origin | model output |
| Forecast origin | the last observed state | an intervention |
| Observed continuation | held-out measured scores after origin | model input or prediction |
| Model mean | deterministic zero-innovation rollout | the conditional mean of the data in a mechanistic sense |
| Sampled paths | draws from the frozen J5 block-bootstrap innovations | posterior samples or biological replicates |
| Ensemble shading | spread of the predictive ensemble | calibrated joint uncertainty |

The 3D explorer defaults to B3 and B5. B1m and B0 can be added. B3 is amber,
B5 violet, B1m grey dashed, B0 grey dotted. The observed continuation is cyan
dotted and explicitly labelled "evaluation only".

## 4. Animation

The Play/Pause controls are Plotly frames. The playhead moves along the
**observed continuation** after the origin. This is a temporal reading aid; it
does not generate the trajectory, and the model paths are static means. The
frame duration is user-selectable.

This choice avoids a misleading "the model is generating the future" animation.
A user who wants to inspect a sampled path can toggle the eight sampled paths;
the 100-member ensembles at the 15 showcase landmarks are available to the
figure builder if needed.

## 5. Uncertainty display

The Studio deliberately does not draw a confidence cone around the 3D mean:

- the saved 100-member ensembles are available only at showcase landmarks;
- the bootstrap is over innovations, not over animals;
- a 3D cone would imply a joint region that the marginal coverage results do
  not establish;
- the ensemble spread is a predictive spread, not a calibrated joint posterior.

Instead, uncertainty is displayed where its unit is explicit: the evidence
forest, the conditional and recording-bootstrap intervals, the horizon profile,
the marginal coverage bars, and the per-landmark score difference. The
per-landmark score is labelled "one landmark is not the population result".

## 6. Interaction contract

- Session and landmark selectors are shared with activity, models, evidence,
  and perturbations.
- Model multiselect changes only displayed traces; it does not refit.
- Observed history length and ensemble visibility are display controls.
- Reset camera changes the component key; it does not change data.
- 2D fallback uses the same `case` object and the same origin.
- Download exports the current figure as a self-contained HTML file.

## 7. Scientific interpretation limits

1. **No anatomy.** PC1–PC3 are not neuron positions. The projection mixes
   neurons; a coordinate is not a cell.
2. **No direct connectivity.** B3's `A` is a lag-one effective transition on
   standardized calcium signals. A coefficient is not a synapse, gap junction,
   or directed influence. The Studio never draws an edge from `A`.
3. **No animal-level generalization.** The recording selector is a view of one
   recording. Five recordings from one preparation do not establish a
   population of animals.
4. **No unique low-dimensional mechanism.** The 10-D PCA is a fixed summary;
   other projections or nonlinear embeddings could show different geometry.
   The result is conditional on this projection and this scoring rule.
5. **No calibration claim.** The 3D ensemble is not a calibrated joint
   predictive distribution. The marginal coverage of B5 is poor (0.725 at 95%),
   and coverage is per dimension.
6. **No causal reading of visual separation.** A model path can look close to
   the observed continuation and still be wrong at the population or animal
   level. The numerical score, not visual proximity, is the endpoint.

## 8. What would justify a stronger 3D claim?

- a projection pre-registered before seeing test outcomes;
- a joint, not marginal, calibration test;
- independent animals with documented identities;
- a model class selected without reusing the test tail;
- a capacity-matched residual model or an explicit measured common input;
- a perturbation with ground truth and a sham control.

Until then, the 3D view is an exploratory instrument for reading a bounded
result.
