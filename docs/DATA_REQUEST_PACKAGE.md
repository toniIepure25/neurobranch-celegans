# Data request package

**Status:** ready to send; **no data received**. This package asks for the minimum artifacts that
would close the gaps in `GAP_CLOSURE_MATRIX.md`. It is deliberately specific so a "yes" can be
acted on without another meeting.

## 1. Sixth held-out worm

Preferred: the exact released embedding.

```text
data/incoming/sixth_worm/bundlenet_consistent_embedding_worm_5.npz
```

Required keys:

| key | shape | meaning |
|---|---|---|
| `x` | `(T, 3)` float | consistent BunDLe-Net embedding, same coordinate system as worms 0–4 |
| `b` | `(T,)` integer | behaviour label in `0..7` |
| `worm_id` | scalar | `5` |
| `protocol` | string | recording/preparation description |
| `embedding_commit` | string | BunDLe-Net code commit and seed used to produce the embedding |
| `preprocessing` | string | alignment, filtering and frame conventions |
| `sha256` | string | hash of the source recording |

Alternative: raw recording plus the exact pipeline.

```text
data/incoming/sixth_worm/raw/
  traces.npy            (T, N) float, ΔF/F or raw with units documented
  neuron_names.json     trace index -> name or null
  states.npy            (T,) integer, 0..7, documented mapping
  metadata.json         fps, frame interval, preparation, animal id, protocol
  behaviour_video.*     optional but preferred
  pipeline_commit.txt   exact code commit + seeds + environment
```

**Acceptance gate:** `python scripts/validate_new_worm.py --worm-id 5 --data-dir <dir>`.
It must pass shape, finiteness, label range, metadata and non-overlap checks. A sixth worm is an
external-validity test, not a leaderboard entry; a worse score is reported.

**Not accepted:** a resampled/duplicated worm, a different embedding model presented as the same
embedding, or labels imputed from the model.

## 2. Independent eight-behaviour video annotation

Needed for the three Atanas validation recordings, or for a new cohort.

```text
data/incoming/annotation/
  videos/<recording_id>.mp4
  annotations/<recording_id>_<annotator_id>.csv
  annotation_manifest.json
```

CSV columns:

```text
recording_id, frame_start, frame_end, behavior_label, annotator_id, confidence, notes
```

Rules:

- two annotators, blinded to condition and to the other annotator;
- at least 20% of frames, stratified by behaviour and recording;
- taxonomy and frame mapping frozen before annotation;
- Fleiss' kappa per behaviour and macro-average reported;
- kappa < 0.6 triggers a taxonomy review (for example, collapse to four macro-states), not a model
  comparison.

**Acceptance gate:** `python scripts/validate_annotation.py --annotations <dir>`.

## 3. Randomized intervention

One row per trial:

```text
animal_id, session_id, arm, target, sham_type,
intervention_start_frame, intervention_end_frame,
behavior_label_path, neural_trace_path, fps, clock_offset_ms, annotator_id, notes
```

Required design:

- arm ∈ {target, sham, random-neuron, unperturbed} as applicable;
- randomization within animal; blinded scorer;
- common hardware clock between neural and behavioural streams;
- target neuron(s) named before data collection;
- positive control with a known behavioural direction;
- at least 6 animals per arm for screening; 12 per arm for confirmation;
- at least 30 trials per animal where possible;
- all exclusions and failed trials reported.

**Acceptance gate:** `python scripts/validate_intervention_trials.py --trials <file>`.
The trial analysis is frozen before unblinding.

After unblinding, add one binary `outcome` column (0/1) without changing the trial keys, then run:

```text
python scripts/analyze_intervention_trials.py \
  --trials data/incoming/intervention/trials_with_outcomes.csv \
  --out reports/vienna_workbench/intervention_analysis.json
```

The analysis treats animals, not trials or frames, as the independent unit. It requires at least two
animals per arm and reports an animal-level risk difference, bootstrap interval and permutation
p-value.

## 4. BCI/EEG/fMRI/MeDIL (only if selected by Denis)

A separate request is required: ethics/consent, BIDS-compatible data, session/subject identifiers,
and a documented online/offline split. None of these data are interchangeable with the worm
recordings. Do not send human data through this repository's public services.

## 5. Email template

```text
Subject: Minimal data package for the next NeuroBranch/Vienna step

Dear Denis,

We have frozen the current five-worm results and do not want to tune another model on the same
data. To make the next step decisive, could you confirm which of the following exist and can be
shared?

1. A sixth held-out worm with the exact BunDLe-Net consistent embedding (preferred), or the raw
   recording plus the pipeline commit.
2. Raw behaviour videos for the three validation recordings, for two blinded eight-behaviour
   annotations.
3. A randomized intervention dataset with target/sham/random-neuron controls and a common clock.
4. If the priority is BCI/NeuroAI rather than the worm work, the relevant human dataset and ethics
   path.

We have prepared exact schemas and validators. If a dataset does not exist, we will deliver the
protocol and power analysis instead of simulating a result. Could you also confirm the primary
endpoint, success criterion and stop rule in the attached one-page brief?

Best,
[Name]
```

## 6. Data handling

- Raw data stay outside Git; only hashes and manifests are committed.
- No redistribution without a licence that permits it.
- No private human data in this repository.
- Every incoming cohort gets an `incoming_manifest.json` before analysis.
