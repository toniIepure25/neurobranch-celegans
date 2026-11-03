# Data Provenance and Licensing

## 1. Primary dataset

| Field | Value |
|---|---|
| Study | Kato, Kaplan, Schrödel, Skora, Lindsay, Yemini, et al. (2015), "Global Brain Dynamics Embed the Motor Command Sequence of *Caenorhabditis elegans*", *Cell* 163(3):656–669 |
| DOI | <https://doi.org/10.1016/j.cell.2015.09.034> |
| Repository | OSF project `2395t`, <https://osf.io/2395t/> |
| File used | `WT_NoStim.mat` |
| Format | MATLAB v7.3 / HDF5 |
| Size | 56,390,975 bytes (verified 2026-11-03) |
| SHA256 | `7531811c1718e26a0455f59cb2f8188cf25809278cfea85770dfe03315a0098a` |
| Local path | `data/raw/kato2015/WT_NoStim.mat` |
| Readme | `data/raw/kato2015/readme_Kato2015.txt` |

The archive is stored read-only in spirit: code never modifies the `.mat` file.
Derived arrays live under `artifacts/` and carry the source hash.

### 1.1 What the file contains

Five wild-type, unstimulated, immobilised preparations recorded with whole-brain
two-photon calcium imaging. The publication reports five animals; the OSF
archive exposes no primary animal key, so the one-to-one animal mapping is
inferred from the publication and acquisition IDs, not documented in the file.
This is one preparation/cohort, not five independent cohorts.

Fields used (actual HDF5 names):

- `traces`: bleaching-corrected ΔF/F, `T × N`, float.
- `traces_raw`: uncorrected traces (kept for audit; not modelled).
- `NeuronNames`: cell identifiers for a subset of traces.
- `States`: one of eight integer codes per frame.
- `fps`, `tv`, `dataset`.

Historical notes used `IDs` and `timeVectorSeconds`; those names are aliases,
not the fields in this archive.

Frame counts and identified counts are in
[`RESULTS.md`](RESULTS.md#1-dataset) and `reports/recording_summary.csv`.

### 1.2 Behaviour labels are not independent measurements

The original readme states the eight states as FWD, SLOW, DT, VT, REV1, REV2,
REVSUS, NOSTATE. These states were inferred from calcium activity in command
interneurons and motor circuits (notably AVA, SMD, RIB) combined with
behavioural scoring. **They are not an external behavioural ground truth and they
are not independent of the neural predictors.** NeuroBranch excludes the direct
label-defining neurons and reports only observational consistency. See
[`METHODOLOGY.md`](METHODOLOGY.md#12-behaviour-labels-and-circularity-control).

### 1.3 Licensing status — read before redistributing

The OSF API metadata for project `2395t` reports `"node_license": null` as of
2026-10-23. **No explicit licence is declared in the repository metadata.**
Therefore:

- The raw `.mat` file is **not** committed to Git (see `.gitignore`).
- NeuroBranch does not redistribute the raw data.
- Downstream users must obtain it from OSF and check the terms themselves.
- The article is published under the publisher's standard terms; consult the
  *Cell* article for reuse of figures/text.

This is a factual description of the metadata, not legal advice.

### 1.4 Integrity checks

```bash
neurobranch data audit
```

verifies the file exists, computes SHA256, lists worms, frames, named/unnamed
counts, missing named cells, NaNs, and macro/fine state counts, and writes:

- `reports/dataset_audit.json`
- `reports/recording_summary.csv`
- `reports/neuron_identity_table.csv`
- `reports/macro_transitions.csv`
- `reports/fine_state_transitions.csv`

A mismatch between the recorded and expected SHA256 is a hard error in the audit
report.

---

## 1A. Phase 3 independent dataset

| Field | Value |
|---|---|
| Study | Baskoylu & Flavell (2026), freely-moving whole-brain calcium imaging and reversal behaviour |
| DOI | <https://doi.org/10.5061/dryad.w9ghx3g4v> |
| Repository | Dryad, dataset id 183044, version 2 id 436401, published 2026-04-15 |
| License | CC0-1.0 |
| Files | 30, total 320,032,093 bytes |
| Animals | 27 |
| Local path | `data/raw/baskoylu2026/` (not committed) |
| Manifest | `data/raw/baskoylu2026/manifest.json` with per-file SHA256 |
| Identity export | `data/raw/baskoylu2026/neuropal_labels.tsv`, SHA256 `7645ad1d112f4a412c84994597a81ad6aae809c275d65befcf68a1d482025d72` |

Fields used: `gcamp/trace_array_original` (raw, `T × N`),
`gcamp/trace_array` (published whole-recording z-score; **not** used as a
predictor), `behavior/velocity`, `reversal_vec`, `reversal_events`,
`head_angle`, `angular_velocity`, `dist_from_cuCenter`, and
`timing/timestamp_confocal` / `timestamp_nir`.

Event convention is **0-based start, exclusive end**. Seven recordings begin in a
partial reversal at frame 0; the adapter permits only that frame-0 initial
mismatch. The full audit, leakage finding and label coverage are in
[`PHASE3_DATASET_AUDIT.md`](PHASE3_DATASET_AUDIT.md).

The NeuroPAL JLD2 strings are not readable by h5py 3.16/HDF5 2.0; a one-time
Julia 1.9.2 + JLD2.jl export produces the TSV. The analysis pipeline reads only
the TSV and has no Julia dependency.

## 1B. Phase 4 directed-turning dataset

| Field | Value |
|---|---|
| Study | Kramer et al. (2026), "Neural sequences underlying directed turning in *Caenorhabditis elegans*", *Nature Neuroscience* |
| DOI | <https://doi.org/10.1038/s41593-026-02257-5> |
| Repository | Dryad `doi:10.5061/dryad.8sf7m0cz2`, dataset id 135272, version 5 id 433175 |
| License | CC0-1.0 |
| Files | 82, total 1,455,335,690 bytes |
| Downloaded | 80 files, 1,444,081,660 bytes; all SHA256 match |
| Recordings | 73 HDF5 files (32 WT octanol, 24 WT diacetyl, 17 tdc1 octanol) |
| Local path | `data/raw/kramer2026/files/` (not committed) |
| Manifest | `data/raw/kramer2026/manifest.json` with the full 82-file SHA256 list |
| Checksums | `reports/phase4/data_checksums.csv` |

Fields used: `gcamp/trace_array_original` (raw, `T × N`),
`gcamp/trace_array` (published whole-recording z-score; **not** used as a
predictor), `behavior/velocity`, `reversal_vec`, `reversal_events`,
`head_angle`, `dv`, `worm_angle`, `angular_velocity`, `pumping`, and
`timing/timestamp_confocal`.  The MATLAB `wbData` structures carry the neuron
identities and odour indices.

Event convention is 0-based start, exclusive end.  `dv` is the
orientation-corrected dorsal/ventral head angle; raw `head_angle` has a
per-recording camera sign and is not used across animals.  The two omitted
files are MATLAB `.chxTracks.mat` behaviour-tracking files, not used by the
analysis.  There are no explicit per-event D/V labels and no trial-level
optogenetic recordings.  Full audit:
[`PHASE4_DATASET_AUDIT.md`](PHASE4_DATASET_AUDIT.md).

---

## 2. Reference code and artifacts

All reference repositories are cloned under `references/external/` and are **not
committed** (they are third-party code).

| Repository | Purpose | Licence status in checkout |
|---|---|---|
| BunDLe-Net (`kumar` et al.) | architecture and official 3-D embeddings | no `LICENSE` file found at repository root in this checkout |
| NC-MCM (`Grosse-Wentrup` et al.) | distributed `.npz` states and Markov test | no `LICENSE` file found at repository root in this checkout |
| `Celegans_premotor` | auxiliary reference | MIT License (Copyright 2024 mmtree) |

Because BunDLe-Net and NC-MCM have no licence file in the cloned checkout, their
code is **not copied into** `src/neurobranch`; only independent reimplementations
and artifact readers are included. If you redistribute this repository, verify
the upstream licences at the canonical repositories first.

### 2.1 Artifacts read (not retrained)

| Artifact | Used for | Caveat |
|---|---|---|
| `bundlenet_consistent_embedding_worm_{1..5}.npz` | official 3-D latent geometry, kNN and silhouette | fitted upstream on all windows → **embedding-level leakage** for kNN; silhouette fully in-sample |
| `nc_mcm_model_worm_{1..5}.npz` | NC-MCM cluster labels, behaviour probabilities, p-value curves | precomputed; not retrained here |

The artifact IDs are mapped to local worm indices 0–4 in the order documented in
`representation/bundlenet.py` and `reproduction/ncmcm.py`. Worm numbering is a
local convention and is stated in every output.

---

## 3. What was not acquired

- No `WT_Stim.mat` or `AVA_HisCl.mat` was downloaded or analysed.
- No new animals were imaged.
- No optogenetic, chemogenetic, or mechanical perturbation was performed.
- No external behavioural tracking data were used in Phase 1/2. Phase 3 uses the
  independently observed Baskoylu & Flavell (2026) behaviour, but still no
  perturbation and no causal intervention.
- Phase 4 used the Kramer et al. (2026) observational archive.  It contains no
  trial-level single-cell optogenetic recordings; tdc1 is a mutant cohort, not a
  perturbation experiment.  The two `.chxTracks.mat` files were not downloaded.

The intervention engine is therefore fitted only to observational data and is
labelled accordingly. The `EmpiricalEvidence` gate would require a matched,
provenance-checked intervention dataset before any output could be called
validated; none is present.

---

## 4. Provenance of derived outputs

Every training suite writes `run_manifest.json` with:

- Git commit, branch, dirty flag;
- dataset path, size, SHA256, and the hash recorded by the loader;
- full resolved config and a config hash;
- per-worm split manifest (frame counts, first/last frame, fractions);
- predictor and excluded-neuron lists;
- model kinds and seeds;
- Python/platform/NumPy/SciPy/scikit-learn/PyTorch versions and CUDA availability;
- exact command line, UTC timestamp, output paths, metric version.

The committed `reports/results_*.csv` and `figures/*` are generated from the
artifacts by `neurobranch report`. They are **not** hand-edited. The
`reports/epistemic_manifest.json` lists the standing caveats in machine-readable
form.


## Phase 11 input

The Phase 11 sensitivity reads the derived Randi table at
`reports/phase3/randi_trial_table_phase3.csv.gz`:

- rows: 237,639
- animals: 110
- SHA-256: `8186c96bcb33cdce15dc9ae88691e09fc596e42c5513310b6d6a636e4d5dd515`

The table is not tracked in Git and is not redistributed until its upstream
licence is confirmed (see [`LICENSE_AUDIT.md`](LICENSE_AUDIT.md)). The hash is
also recorded in `reports/phase11/summary.json` and `manifest.json`.
