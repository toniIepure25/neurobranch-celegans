# Perturbation identifiability — do observationally similar models agree?

Artifacts: `reports/simulator/failure_diagnostics/perturbation/`
Code: `src/neurobranch/diagnostics/perturbation.py`, `scripts/genfail_perturbation.py`

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**

This stage does not claim that any named neuron causes a behavioural or network effect. It asks whether the models’ **virtual** responses to the same mapped perturbation agree. They largely do not.

## 1. What was perturbed

- Five test frames per session, selected from valid test starts.
- Six named target neurons (alphabetically first available named predictors).
- A +0.5 standard-deviation impulse added to the target at the initial state.
- 20-frame response, evaluated at the final frame.
- Models: stable VAR, latent LDS, second-order latent LDS, causal calcium LDS, compositional latent LDS.
- The same target index and amplitude were mapped into each model’s state space using its observation map.

Initial-state support was high: mean fraction of neurons inside the train 0.5–99.5% range was 0.939 for the four direct models and 0.918 for the compositional model. The perturbation is therefore not mostly outside observed support, although a 0.5-SD coordinated state change is still a model construction, not an experimental intervention.

## 2. Cross-model agreement

Cosine similarity between the 20-frame response vectors of each model pair (150 comparisons for the four direct models; 30 for compositional):

| model pair | mean cosine | median cosine | mean relative disagreement |
|---|---:|---:|---:|
| latent LDS vs second-order | **0.976** | 0.982 | 0.261 |
| calcium vs compositional | 0.559 | 0.599 | 1.418 |
| stable VAR vs second-order | 0.520 | 0.604 | 1.300 |
| latent LDS vs calcium | 0.502 | 0.807 | 1.156 |
| stable VAR vs latent LDS | 0.498 | 0.579 | 1.278 |
| second-order vs calcium | 0.482 | 0.760 | 1.171 |
| stable VAR vs calcium | 0.236 | 0.402 | 1.659 |
| second-order vs compositional | −0.022 | −0.049 | 1.362 |
| stable VAR vs compositional | −0.018 | 0.032 | 1.434 |
| latent LDS vs compositional | **−0.073** | −0.099 | 1.400 |

Only the two closely related latent LDS variants agree strongly. The compositional model disagrees with the latent models and with the stable VAR; mean cosines are near zero, and relative disagreement exceeds 1.3. Stable VAR and calcium agree only weakly.

A cosine of 0.5 is not “moderate agreement” for a perturbation claim: it means half the response geometry is model-dependent. The near-zero compositional values mean the predicted response direction is essentially unrelated.

## 3. Target-ranking agreement

For each frame, models rank the six targets by their own response magnitude. Median Spearman correlation between model rankings is **0.371**. Models do not agree on which named neuron is most responsive. This is a direct warning against using any single model’s target ranking as a biological hypothesis.

## 4. Stability under resampling

Refitting the latent LDS and second-order model on 80% start subsamples (two independent subsamples) gives:

- median response cosine: **0.991**;
- AR(1) median cosine 0.985, AR(2) 0.997;
- relative disagreement ~0.15–0.19.

So the *same model class* is numerically stable under resampling. The disagreement above is therefore **structural/model-form disagreement**, not random seed noise.

## 5. Interpretation

**Supported:**

- Observationally similar models can produce incompatible virtual perturbation responses.
- The disagreement is largest for the compositional model and between VAR and calcium models.
- Resampling stability is high, so the disagreement is not a fitting instability.

**Not supported:**

- Any causal effect of a named neuron.
- That one model’s perturbation response is the “true” one.
- That the perturbations correspond to experimental interventions; no matched experimental perturbation exists.

The correct decision is `PERTURBATION_UNCERTAINTY_CHARACTERIZED` — PASS for characterising disagreement; `BIOLOGICAL_INTERVENTION_VALIDATION` remains **NOT_EVALUABLE** because there is no compatible experimental intervention dataset.

## 6. Why this matters

Even if a model forecasts acceptably, its perturbation predictions are not identified by observational data alone. Two models can fit the same observed trajectories and imply different responses to a state change. This is a fundamental limitation of purely observational virtual perturbation, not a bug that more training will necessarily fix.

The warning must travel with every perturbation table and dashboard view:

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.**
