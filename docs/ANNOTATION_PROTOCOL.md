# Independent behaviour annotation protocol

**Status:** protocol only; no independent annotations exist in the repository.

## 1. Purpose

The released eight-state labels are not independent video annotations. This protocol produces a
small, auditable, blinded annotation set for the three Atanas validation recordings (or a new
cohort) so that behaviour claims can be evaluated against something other than the model's own
labels.

## 2. Sampling

- Unit: one recording, not one frame.
- At least **20% of frames per recording**, stratified by the released label and by time block.
- Include all reversal onsets/offsets and a random sample of ambiguous transitions.
- Two annotators per recording, blinded to condition, model output and each other.
- A third adjudicator resolves disagreements only after both annotations are locked.
- Frame boundaries and fps come from the recording metadata, not from the model.

## 3. Taxonomy

Use the eight released states, frozen before annotation:

```text
0  ... 7   # exactly the names in the release; do not rename silently
```

If a behaviour is not represented, annotate `unclear` and document it. Do not force every frame
into a state.

Rules:

- annotate at the same frame resolution as the neural recording;
- a state lasts at least the release's minimum bout length;
- transitions are labelled at the first frame of the new state;
- `unclear` frames are excluded from agreement, but their rate is reported.

## 4. Agreement

- Fleiss' kappa per behaviour and macro-average.
- Prevalence-adjusted bias-adjusted kappa (PABAK) as a secondary descriptive.
- Agreement on reversal onset within ±2 frames reported separately.
- Threshold for proceeding: **macro Fleiss kappa ≥ 0.60**.
- If below 0.60: review taxonomy and sampling; collapse to four pre-registered macro-states if
  needed. Do not compare models until the revised taxonomy passes.
- If agreement remains low, report that the eight-state endpoint is not reliably annotatable from
  video and stop; do not substitute model labels.

## 5. Outputs

```text
data/incoming/annotation/
  videos/<recording_id>.mp4
  annotations/<recording_id>_<annotator_id>.csv
  annotation_manifest.json
  agreement_report.json
```

CSV columns:

```text
recording_id, frame_start, frame_end, behavior_label, annotator_id, confidence, notes
```

Manifest:

```json
{
  "protocol_version": "vienna-annotation-v1",
  "taxonomy": ["..."],
  "annotators": ["a1", "a2"],
  "recordings": ["2023-01-23-15"],
  "sampling_fraction": 0.2,
  "video_sha256": {"2023-01-23-15": "..."},
  "frozen_before_annotation": true
}
```

## 6. Analysis freeze

Before unblinding:

1. freeze the annotation file hashes;
2. freeze the model outputs and the evaluation script;
3. specify the primary agreement metric and the handling of `unclear`;
4. specify one primary behavioural endpoint;
5. no model retraining on the annotation set.

## 7. What this does not establish

Independent annotation of three recordings improves measurement validity; it does not create a new
animal, a causal intervention or a population-level claim. Those remain G1/G3.
