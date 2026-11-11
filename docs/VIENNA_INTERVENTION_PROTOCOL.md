# Intervention protocol for a real `do()` test

Status: **not collected**. This is a pre-registered request package, not evidence
that the current NC-MCM `do()` is causally identified. The precomputed artifact
contains no perturbations, so the workbench can only propagate a forced state
through a fitted chain.

## 1. Estimand

For a target neuron or circuit state `S`, estimate the effect of an externally
delivered intervention `do(S = s)` on the distribution of the eight behaviours
over a fixed post-intervention window, relative to a randomized sham.

The estimand is **not** "what the Markov chain would do". It is a contrast in
held-out animals under an assigned intervention.

## 2. Minimum design

| element | requirement |
|---|---|
| assignment | randomized target vs sham, counterbalanced across animals and time |
| unit | animal; repeated windows are not independent animals |
| target | prespecified neuron(s) or circuit; no switching targets after inspection |
| sham | identical handling, light/solution/vehicle, no effective perturbation |
| sample | at least 6 animals per arm for a screening estimate; power calculation required before confirmatory claim |
| recording | neural trace and behaviour video with common hardware clock |
| labels | independent eight-behaviour annotation, or a prespecified subset with chance-corrected agreement |
| blinding | annotator blinded to arm; analysis code frozen before unblinding |
| negative control | untargeted neuron / off-target light / vehicle |
| positive control | a perturbation with a known behavioural direction, if available |

## 3. Required metadata

One row per trial:

```text
animal_id, session_id, arm, target, sham_type, intervention_start_frame,
intervention_end_frame, behavior_label_path, neural_trace_path,
frame_rate_hz, clock_offset_ms, annotator_id, notes
```

## 4. Analysis plan

1. Freeze preprocessing and the eight-behaviour mapping.
2. Fit the descriptive model on **non-intervention** sessions only.
3. Predict the intervention session's behaviour distribution from pre-window
   state, without using post-window behaviour.
4. Compare target vs sham on the prespecified post-window contrast.
5. Report effect size with animal-level bootstrap CI and a randomization test.
6. Compare against a matched **replay** and a **no-intervention** forecast; the
   replay is a negative control, not a causal model.
7. Report all animals, including failures and excluded trials with reasons.

## 5. Acceptance gates

- Common clock and frame convention documented before scoring.
- Target and sham differ only in the assigned perturbation.
- Behaviour labels have chance-corrected reliability reported.
- Analysis code and primary outcome frozen before unblinding.
- Effect is replicated in an independent cohort before any "causal" claim.

## 6. What this would and would not establish

A positive randomized result would support a **causal effect of that intervention
on that behaviour readout in that preparation**. It would not automatically
validate the NC-MCM state ontology, identify whole-brain mechanism, or justify a
digital-twin claim. Those require separate perturbation and validation work.

## 7. Current workbench boundary

`NCMCMWorm.intervention_horizon` is a model-based counterfactual: it forces a
state at `t=0` and propagates the fitted transition matrix. It has no
intervention data, no randomization, and no identified effect. The UI must keep
the label **"model-based counterfactual, not an identified causal effect"**.
