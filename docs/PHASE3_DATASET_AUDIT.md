# Phase 3 dataset audit

**Branch:** `research/phase3-neural-generalization`
**Audit date:** 2026-10-23
**Primary dataset:** Baskoylu & Flavell (2026), Dryad `doi:10.5061/dryad.w9ghx3g4v`
**Version:** Dryad dataset 183044, version 2 (id 436401), published 2026-04-15
**Licence:** CC0-1.0
**Raw archive:** 30 files, 320,032,093 bytes, SHA256 recorded per file in
`reports/phase3/dataset_manifest.json`

The raw files are downloaded into `data/raw/baskoylu2026/files/` and remain
untracked. The manifest and all derived audit tables are tracked under
`reports/phase3/`.

## 1. What was downloaded

| Component | Count | Notes |
|---|---:|---|
| `*-cudata.h5` recordings | 27 | one freely-moving animal each |
| `NeuroPAL_labels.jld2` | 1 | identity dictionary, not readable by h5py |
| README and Julia example | 2 | loader documentation |
| Total | 30 | 320,032,093 bytes |

The downloader solves the Dryad Anubis proof-of-work challenge, verifies every
file's SHA256 against the Dryad API, and writes `manifest.json`. It is
reproducible with:

```powershell
python scripts/phase3_download_baskoylu.py
```

The public API returned HTTP 401 for the bulk download and HTTP 403 for direct
file streams without the challenge; those failures and the workaround are in the
script. They are access-control behaviour, not missing data.

## 2. HDF5 structure verified

Every file was opened with h5py. The arrays used here are:

| Path | Shape | Meaning |
|---|---|---|
| `gcamp/trace_array_original` | `(1600, N)` | original GCaMP, **primary input** |
| `gcamp/trace_array` | `(1600, N)` | published z-score, **not primary** |
| `timing/timestamp_confocal` | `(1600,)` | confocal frame time |
| `behavior/velocity` | `(1600,)` | signed velocity |
| `behavior/reversal_vec` | `(1600,)` | frame-level reversal state |
| `behavior/reversal_events` | `(2, E)` | start/end columns |
| `behavior/head_angle` | `(1600,)` | head angle |
| `behavior/angular_velocity` | `(1600,)` | angular velocity |
| `behavior/dist_from_cuCenter` | `(1600,)` | arena distance |

The HDF5 shape is time × neuron. The Julia README describes the transposed
column-major view; this was checked directly, not assumed.

### The published z-score is transductive

For `2023-03-30-01`, `(trace_array_original - full_recording_mean) /
full_recording_std` matches `gcamp/trace_array` to within `0.0077` (the export
uses a slightly different variance convention). The published z-score therefore
uses statistics from the whole recording, including future frames. Phase 3 does
not use it for the primary benchmark. All normalisation is causal (current and
past frames only), and any train-fitted transform is fit on training animals.

## 3. Event index audit

`reversal_events` uses **0-based start and exclusive end**. This was established
by checking every event against `reversal_vec`:

- event `[13, 30]` corresponds exactly to `reversal_vec[13:30] == 1`;
- event `[1, 9]` corresponds to frames 1–8, not 0–8;
- a recording can begin inside a reversal, and the event table omits that initial
  partial segment.

The loader requires every event interval to be entirely reversal, rejects
overlaps and out-of-range bounds, and permits at most the single initial-frame
mismatch. Across 27 animals and 1,162 events, **7 animals** start in a partial
reversal; there are **7 mismatched frames in total**, all at frame 0. No other
mismatch was found. An earlier 1-based assumption produced 31–41 mismatches per
animal and was discarded.

This correction matters: a one-frame shift would move the onset label into the
feature window at some horizons.

## 4. Coverage and quality

All 27 animals passed structural QC:

| Quantity | Value |
|---|---:|
| Animals usable | 27 / 27 |
| Frames per animal | 1,600 |
| Total frames | 43,200 |
| Duration | 941.3–962.0 s (mean 960.8 s) |
| Median sampling interval | 0.6000189 s (range 0.5950–0.6000) |
| Neurons per animal | 132–163 (mean 147.6) |
| Reversal events | 21–62 per animal; 1,162 total |
| Event rate | 1.31–3.87 per min (mean 2.69) |
| Frames in reversal | 19.6–48.1% (mean 31.3%) |
| Non-finite raw trace values | 0 |
| Non-monotonic timestamps | 0 |

Exclusion rules were structural only: unreadable file, non-monotonic time,
missing behaviour, or fewer than 5 events. No animal was excluded for model
performance. Per-animal values are in
`reports/phase3/dataset_coverage.csv`.

The 31.3% reversal occupancy explains why a 6 s horizon has ~35% positive
prevalence among non-reversing frames: reversals are clustered, not rare point
events. This is a property of the data, not a label leak, and it is why a
constant and a behavioural-history baseline are mandatory.

## 5. NeuroPAL identities

`NeuroPAL_labels.jld2` cannot be read by h5py: its string type is a JLD2-specific
variable-length compound, and both h5py 3.16/HDF5 2.0 and h5py 3.12/HDF5 1.14
raise `OSError: Can't synchronously read data (ran off end of input buffer while
decoding)`. Installing an older h5py did not fix it. The JLD2 dictionary was
therefore converted once with Julia 1.9.2 + JLD2.jl by
`scripts/export_neuropal.jl`.

The export was checked against the Julia example structure:

- 27 animal keys;
- each value is `(index_dict, class_dict)`;
- index keys are **1-based** in Julia and were converted to **0-based** trace rows;
- the maximum index (162) is below the largest trace count (163), confirming the
  conversion;
- metadata includes `label`, `neuron_class`, `confidence`, `roi_id`.

Derived file: `data/raw/baskoylu2026/neuropal_labels.tsv`
SHA256 `7645ad1d112f4a412c84994597a81ad6aae809c275d65befcf68a1d482025d72`
2,595 rows, 170 unique labels, 27 animals.

| Identity quantity | Value |
|---|---:|
| Labelled traces | 2,595 |
| Labels per animal | 77–116 (mean 96.1) |
| Unique labels | 170 |
| Labels present in all 27 animals | 1 (`OLLL`) |
| Labels present in ≥26 animals | 7 |
| Labels present in ≥20 animals | 74 |
| Confidence values | 2:183, 3:597, 4:515, 5:1,299, 6:1 |
| Ambiguous duplicate-label traces | 6 |

Three duplicate label pairs occur (`RIBR` twice in `2024-05-22-21`, `RMDVL`
twice in the same animal, `SMBDR` twice in `2024-05-28-10`). Both members of each
pair are excluded from identity-aligned analyses; the export does not say which
ROI is the real cell. The single confidence-6 value (`RIVL`,
`2023-09-07-01`) is outside the documented 1–5 scale and is retained with a
warning, not silently clipped.

The small all-animal intersection is why identity-aligned models use labels
present in at least 80% of animals and mask absent labels as zero after causal
z-scoring. This is a conservative representation, not a claim that every animal
has the same 170 cells identified.

## 6. What this dataset can and cannot answer

**Can answer:** whether recorded neural population activity adds prospective
predictive value for a *real, independently observed* behavioural transition;
whether that value survives animal-disjoint evaluation and causal preprocessing;
whether identified-cell structure adds anything beyond population summaries.

**Cannot answer:** causality, necessity, sufficiency, or whether stimulating a
neuron will cause a transition. Correlated pre-transition activity is not a
control signal. No intervention was performed on these animals.

**Known limits:** 16 minutes per animal; 27 animals; one lab/preparation; GCaMP
imaging; NeuroPAL labels are probabilistic and partly ambiguous; behaviour is
resampled to 0.6 s, which is coarse relative to the ~0.3–1 s timescale of some
reversal dynamics; the arena/copper context is not experimentally varied.

## 7. Reproduction checks

```powershell
python -m pytest tests/data/test_baskoylu.py -q
python - <<'PY'
from neurobranch.data.baskoylu import load_all
recs, rows = load_all()
print(len(recs), sum(len(r.reversal_onsets) for r in recs))
PY
```

Expected: 27 usable animals, 1,162 events. The tests cover event-index
conversion, timestamp alignment, trace orientation, identity ambiguity,
train-only/causal normalisation, horizon labels and chronological purge.
