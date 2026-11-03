# Master Review: Data and Landmark Forensics

**Scope:** Kato et al. (2015) `WT_NoStim` only. No other biological dataset was loaded for the numerical audit.  
**Evidence directory:** [`reports/master_review/`](../reports/master_review/)  
**Primary artifacts:** `dataset_identity_audit.json`, `frame_landmark_bounds.json`, `provenance_checks.json`, `frame_landmark_bounds.csv`, `landmark_bounds_by_horizon.csv`.

---

## 1. Why this document exists

A predictive result is only as good as the object it was computed on. Before evaluating B3 versus B5, the audit had to establish:

1. that the local file is the public Kato `WT_NoStim` object;
2. how many recordings, frames, traces and predictors exist;
3. whether the five entries are five animals, five sessions, or something else;
4. whether the split boundaries are per-recording and purge-safe;
5. whether every forecast target is inside the recording and inside the test mask;
6. whether the documents describe those facts correctly.

The short answer: **the archive and the code pass; several documents do not.**

---

## 2. File identity and provenance

### 2.1 Direct file facts

| Field | Observed |
|---|---|
| Path | `data/raw/kato2015/WT_NoStim.mat` |
| Format | MATLAB v7.3 / HDF5 |
| Size | `56,390,975` bytes |
| SHA256 | `7531811c1718e26a0455f59cb2f8188cf25809278cfea85770dfe03315a0098a` |
| OSF project | `2395t`, <https://osf.io/2395t/> |
| OSF file size | `56,390,975` bytes |
| OSF SHA256 | matches local |
| Readme | `readme_Kato2015.txt`, SHA256 `9fbc04b06a5c59d13a7ce6a90a428bc49e2d4ef5a2ff21bd2efa6f67655bc3d6` |
| Licence | OSF metadata `node_license: null`; no explicit data licence |

The local hash matches both the expected hash in the repository and the OSF object. The file is the right object.

The documented size in `docs/DATA_PROVENANCE.md` is `128,511,040` bytes. That is false. The difference is not a different version with the same hash; it is a documentation error. The hash is the authoritative identity.

### 2.2 HDF5 structure

Top-level group `WT_NoStim` contains:

| Field | Shape / type | Meaning |
|---|---|---|
| `NeuronNames` | `(5,1)` object references | Per-recording name vectors |
| `Opts` | `(5,1)` object references | Bleaching-correction options and support data |
| `States` | `(5,1)` object references | Per-frame one-hot state labels |
| `dataset` | `(5,1)` object references | Acquisition name strings |
| `deltaFOverF` | `(5,1)` object references | Uncorrected ΔF/F, stored `N × T` |
| `deltaFOverF_bc` | `(5,1)` object references | Bleaching-corrected ΔF/F, stored `N × T` |
| `derivs` | `(5,1)` object references | Derivative traces |
| `fps` | `(5,1)` numeric | Stored frame rate |
| `stateParams` | `(5,1)` object references | Per-neuron state-annotation parameters, `N × 5` |
| `tv` | `(5,1)` object references | Time vector in seconds |

`Opts` contains only `bc_options` and `bc_suppdata` (bleach curves and tau). It contains **no animal identifier, genotype key, or session-to-publication mapping**.

The loader exposes traces as `T × N`; the HDF5 object stores `N × T`. This transpose is a code convention, not a data change. `docs/DATA_PROVENANCE.md` describes `T × N` without saying that this is the loader’s view; a reader inspecting HDF5 directly sees `N × T`. This is a minor documentation ambiguity.

`stateParams` is not a group; it is a per-recording `N × 5` numeric dataset. Its row counts do not always equal the trace counts:

| Session | Traces | `stateParams` rows | Match? |
|---:|---:|---:|---|
| 0 | 109 | 109 | yes |
| 1 | 135 | 107 | **no** |
| 2 | 131 | 131 | yes |
| 3 | 125 | 126 | **no** |
| 4 | 129 | 129 | yes |

The model never reads `stateParams`. The mismatch has no effect on the reported scores, but it shows that archive tables are not uniformly row-aligned. Any future use of `stateParams` must verify the join key first.

---

## 3. The five recordings

### 3.1 Counts and dimensions

| Session | Acquisition ID | Frames | Traces | Predictors | Named | Unnamed | Named fraction |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | `TS20140715e_lite-1_punc-31_NLS3_2eggs_56um_1mMTet_basal_1080s` | 3,137 | 109 | 103 | 38 | 71 | 0.349 |
| 1 | `TS20140715f_lite-1_punc-31_NLS3_3eggs_56um_1mMTet_basal_1080s` | 3,134 | 135 | 127 | 44 | 91 | 0.326 |
| 2 | `TS20140905c_lite-1_punc-31_NLS3_AVHJ_0eggs_1mMTet_basal_1080s` | 3,059 | 131 | 123 | 32 | 99 | 0.244 |
| 3 | `TS20140926d_lite-1_punc-31_NLS3_RIV_2eggs_56um_1mMTet_basal_1080s` | 3,311 | 125 | 118 | 47 | 78 | 0.376 |
| 4 | `TS20141221b_THK178_lite-1_punc-31_NLS3_6eggs_56um_1mMTet_basal_1080s` | 3,021 | 129 | 121 | 31 | 98 | 0.240 |
| **Total** | | **15,662** | **629** | **592** | **192 occurrences** | **437** | |

“Predictors” means traces remaining after excluding the eight behaviour-label-defining names:

```text
AVAL, AVAR, SMDDR, SMDDL, SMDVR, SMDVL, RIBR, RIBL
```

Not every recording contains all eight. Session 0 contains six: `AVAL, AVAR, SMDDR, SMDVR, SMDVL, RIBL`; `SMDDL` and `RIBR` are absent. The others contain all eight. The exclusion is name-based and does not remove every behaviour-correlated neuron, so it is a circularity reduction, not an independence guarantee.

The 192 “named occurrences” are occurrences across recordings, not 192 unique names. There are 58 unique alphabetic names; the same name appears in multiple recordings. The numeric labels are not cell identities and are not aligned across recordings.

### 3.2 Time axes and frame rates

| Session | `fps` stored | `fps` from `tv` | Duration (s) | Median `dt` (s) | Regular? |
|---:|---:|---:|---:|---:|---|
| 0 | 2.9046296 | 2.9046296 | 1079.656 | 0.344278 | yes |
| 1 | 2.9018519 | 2.9018519 | 1079.655 | 0.344608 | yes |
| 2 | 2.8324074 | 2.8324074 | 1079.647 | 0.353057 | yes |
| 3 | 3.0657407 | 3.0657407 | 1079.658 | 0.326214 | yes |
| 4 | 2.7972222 | 2.7972222 | 1079.648 | 0.357498 | yes |

All time axes are strictly increasing and regular to floating-point precision. There are no NaNs in either the corrected or raw traces. The recordings are ~18 minutes each, with session-specific frame rates around 2.8–3.1 Hz.

### 3.3 Behaviour labels

The eight states are `FWD`, `SLOW`, `DT`, `VT`, `REV1`, `REV2`, `REVSUS`, `NOSTATE`. Macro counts:

| Session | FWD | REV | TURN | NOSTATE |
|---:|---:|---:|---:|---:|
| 0 | 1,306 | 1,303 | 506 | 22 |
| 1 | 1,342 | 1,268 | 502 | 22 |
| 2 | 1,057 | 1,475 | 451 | 76 |
| 3 | 1,589 | 1,173 | 511 | 38 |
| 4 | 1,354 | 1,260 | 389 | 18 |

The labels are not independent measurements. The readme states that the states are behavioural categories; the project documentation correctly notes that they are inferred from command/interneuron calcium and behaviour. Excluding the label-defining neurons reduces direct circularity but does not make the remaining brain independent of the behaviour.

This matters for interpretation: `WT_NoStim` means **no optogenetic stimulus**, not **no behaviour** and not **no sensory input**. The worm is immobilised, but the recorded brain is still generating motor-related states. A “spontaneous” claim must be qualified.

---

## 4. Are the five entries five animals?

### 4.1 What the archive proves

The archive proves:

- five distinct `dataset` strings;
- five distinct acquisition timestamps/IDs;
- four distinct acquisition dates;
- two same-date acquisitions with different suffixes (`e`, `f`), egg counts (2 vs 3), frame counts (3,137 vs 3,134), trace counts (109 vs 135), and predictor dimensions (103 vs 127);
- one time axis, neuron panel and state vector per entry.

The archive does **not** prove:

- an animal identifier;
- that each acquisition is a different physical animal;
- that the five entries are the five animals reported in the paper;
- that the five animals are an independent cohort.

### 4.2 What the publication and secondary sources say

- Kato et al. (2015), *Cell* 163:656–669, DOI `10.1016/j.cell.2015.09.034`, PMID `26478179`, report a five-animal `WT_NoStim` cohort.
- Grosse-Wentrup et al. (2024), *PLoS Comput Biol*, DOI `10.1371/journal.pcbi.1012674`, PMID `39680605`, describe the same dataset as five worms with 107–131 neurons.
- BunDLe-Net (Kumar et al., bioRxiv 2023, DOI `10.1101/2023.08.08.551978`) also describes five worms.

The publication’s full methods text could not be fetched (publisher HTTP 403/CAPTCHA), so this is not a direct quotation. It is consistent across the publication record and two secondary analyses.

### 4.3 The defensible conclusion

```text
YES_WITH_HIGH_CONFIDENCE_FOR_FIVE_DISTINCT_PREPARATIONS
NO_EXPLICIT_ANIMAL_ID_IN_ARCHIVE
MODERATE_CONFIDENCE_FOR_ONE_TO_ONE_PUBLICATION_MAPPING
```

Recommended wording:

> Five Kato `WT_NoStim` recordings from five distinct acquisition preparations, consistent with the publication’s five animals. The OSF archive does not expose an explicit animal identifier, so the animal-level mapping is inferred rather than documented by a primary key. The five recordings are one laboratory cohort, not an independent replication.

This corrects both errors:

- “five repeated sessions of one preparation” is not supported by the acquisition IDs, dates, egg counts or trace panels;
- “five independent animals” overstates independence. They are five animals from one laboratory and one experimental protocol.

For inference, the five recordings remain the independent units. They are not a population sample, and five units give low power.

---

## 5. Split construction

### 5.1 The algorithm

For a recording with `T` frames and purge `p = 12`:

```text
train_end   = floor(0.70 * T)
train       = [0, train_end)
val_start   = train_end + p
val_end     = val_start + floor(0.15 * T)
val         = [val_start, val_end)
test_start  = val_end + p
test        = [test_start, T)
```

All bounds are half-open. The validation block length is `floor(0.15*T)`, not a second percentage of the remaining data. The test block absorbs rounding and therefore is slightly longer than 15% in some recordings.

### 5.2 Exact bounds

| Session | Train | Validation | Test | Test frames |
|---:|---|---|---|---:|
| 0 | `0:2195` | `2207:2677` | `2689:3137` | 448 |
| 1 | `0:2193` | `2205:2675` | `2687:3134` | 447 |
| 2 | `0:2141` | `2153:2611` | `2623:3059` | 436 |
| 3 | `0:2317` | `2329:2825` | `2837:3311` | 474 |
| 4 | `0:2114` | `2126:2579` | `2591:3021` | 430 |

There is no overlap between train/validation/test. The purge gaps are exactly 12 frames at each boundary.

### 5.3 What the code actually does

The code uses **per-recording** test starts. It does not use a common `2689:3137` boundary. The exact bounds above were recomputed independently from the raw frame counts and match the frozen split arrays.

The earlier concern that a shared boundary might silently truncate sessions 1–4 is therefore **not a computational defect**. It is a documentation defect: `docs/HANDOFF_METHODS_AND_LIMITATIONS.md` presents `2689:3137` as if it were common. Only session 0 has that test block.

### 5.4 Window validity

A window ending at `t` with history `h` and target `t+H` is admitted only if every frame from `t-h+1` to `t+H` is inside the same split. This is enforced by `valid_start_indices` and `split_starts` in `src/neurobranch/preprocessing/__init__.py`.

The audit checked the stronger condition directly:

- every target index is `< T`;
- every target index is in the test mask;
- no window crosses a purge gap;
- no padding, interpolation or resampling is used.

All pass for H1–H100.

---

## 6. Landmarks and bounds

### 6.1 The algorithm

The evaluation landmarks are generated inside the test mask with:

```text
history = 2
stride  = 10
horizon_max = 100
```

The first landmark is `test_start + 1` because the two-frame history must be inside the test mask. Landmarks are spaced 10 frames apart. For each horizon, the target `t+H` must be inside the recording and inside the test mask.

### 6.2 Actual H100 counts

| Session | Test start | Landmarks | First | Last | Last target | `T-1` |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2689 | 35 | 2690 | 3030 | 3130 | 3136 |
| 1 | 2687 | 35 | 2688 | 3028 | 3128 | 3133 |
| 2 | 2623 | 34 | 2624 | 2954 | 3054 | 3058 |
| 3 | 2837 | 38 | 2838 | 3208 | 3308 | 3310 |
| 4 | 2591 | 33 | 2592 | 2912 | 3012 | 3020 |
| **Total** | | **175** | | | | |

The counts are **not** 35 per session. They are 35, 35, 34, 38, 33. The total is 175. This is a confirmed documentation error, not an analysis error.

### 6.3 Boundary margin

At H100, the largest target is session 3’s `3308`, with `T-1 = 3310`: two frames of margin. Every H100 target is comfortably inside its recording.

At H5, session 1’s largest target is `3133` with `T-1 = 3133`: exactly the last observed frame. The target is valid, but there is **zero margin**. Any later preprocessing that drops the last frame, or any off-by-one change in the horizon convention, would invalidate that landmark. This is a boundary fragility to document, not a current leak.

### 6.4 Frozen-landmark identity

The frozen 175 landmarks in `reports/simulator/kato_common_state/model_per_landmark.csv` match the independently generated set exactly:

- session counts match;
- landmark values match;
- no extra or missing landmark;
- no target outside the test mask.

The H100 range is `2690..3030`, `2688..3028`, `2624..2954`, `2838..3208`, `2592..2912`.

---

## 7. Documentation defects and corrections

| ID | Document | Claim | Correct | Severity |
|---|---|---|---|---|
| X01 | `docs/DATA_PROVENANCE.md` | File size `128,511,040` | `56,390,975` bytes | High |
| X03 | `docs/ADAPTIVE_DYNAMICS_MODELS.md` | “175 test landmarks (35 per session)” | 35/35/34/38/33 | Medium |
| X04 | `docs/HANDOFF_METHODS_AND_LIMITATIONS.md` | Common test `2689:3137` | Session 0 only; per-recording bounds | Medium |
| X02 | Multiple docs | Five repeated sessions, not animals | Five animals inferred; one cohort | High |
| X13 | Archive metadata | `stateParams` rows align with traces | 107 vs 135, 126 vs 125; unused | Low |
| X14 | `docs/DATA_PROVENANCE.md` | Fields `IDs`, `timeVectorSeconds` | HDF5 `NeuronNames`, `tv` | Low |
| X16 | `docs/DATA_PROVENANCE.md` | Traces are `T × N` | HDF5 stores `N × T`; loader exposes `T × N` | Low |

The historical documents are preserved unchanged. Corrections are recorded in [`reports/master_review/corrections.json`](../reports/master_review/corrections.json) and in the claims ledger. Rewriting the old documents would destroy the audit trail.

---

## 8. What the data do not contain

- No explicit animal IDs or mapping table.
- No genotype metadata beyond the acquisition-name string.
- No measured common input or sensory stimulus in `WT_NoStim`.
- No trial-level optogenetic intervention.
- No cross-session neuron alignment for the numeric traces.
- No independent cohort.
- No licence declaration on OSF.

These are not defects in the analysis; they are limits on what the analysis can claim. In particular:

- the numeric traces cannot be treated as the same cells across recordings;
- the five recordings cannot be treated as five independent population samples;
- the absence of a stimulus does not mean the absence of behaviour or internal state;
- the absence of intervention data makes biological intervention validity `NOT_EVALUABLE`.

---

## 9. Forensic conclusions

1. **Object identity:** verified. Hash, size and OSF object match.
2. **Animal identity:** five distinct acquisitions; five animals inferred; no primary key; one cohort.
3. **Dimensions:** 15,662 frames, 629 traces, 592 predictors; variable per recording.
4. **Splits:** per-recording, purge-safe, non-overlapping.
5. **Landmarks:** 175 H100 landmarks; counts 35/35/34/38/33; all targets in bounds.
6. **Frozen artifacts:** match the independent landmark set.
7. **Documentation:** four material errors and three minor metadata ambiguities.
8. **Boundary fragility:** H5 session 1 reaches the last observed frame with zero margin.
9. **Data limitations:** no animal key, no intervention, no independent cohort, no licence.

The data are sufficient for a within-cohort, exploratory predictive analysis. They are not sufficient for an animal-level generalization claim or a biological mechanism claim.

---

## 10. Reproduction commands

```powershell
python scripts/master_review/kato_forensics.py
python scripts/master_review/build_audit_ledgers.py
python -m pytest -q tests/master_review/test_kato_bounds.py
```

The focused tests recompute the H100 bounds directly from HDF5 and compare them with the frozen table. They do not import the project’s split helper for the expected values.
