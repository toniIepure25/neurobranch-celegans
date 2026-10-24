# Phase 4 dataset audit: Kramer et al. (2026)

## Provenance

- Dryad: `doi:10.5061/dryad.8sf7m0cz2`
- Dataset id: `135272`
- Version: 5, version id `433175`
- Licence: CC0-1.0
- Archive: 82 files, 1,455,335,690 bytes
- Downloaded for analysis: 80 files, 1,444,081,660 bytes
- SHA-256: all 80 downloaded files match the Dryad manifest
  (`reports/phase4/data_checksums.csv`)

The two files not downloaded are `03072024_N2_2.chxTracks.mat` and
`03072024_N2_3.chxTracks.mat`, MATLAB behaviour-tracking files.  They are not
used by the neural/behaviour analysis.  Their manifest checksums are retained
in `data/raw/kramer2026/manifest.json`.

## Recording inventory

`load_all()` validates every MATLAB row against an HDF5 raw-trace file.  All
73 rows map exactly; no audit row failed.

| cohort | recordings | frames | reversals | median neurons | labelled traces |
|---|---:|---:|---:|---:|---:|
| WT octanol | 32 | 48,877 | 990 | 159 | 3,278 |
| WT diacetyl | 24 | 34,490 | 1,013 | 161.5 | 2,061 |
| tdc1 octanol | 17 | 23,764 | 667 | 162 | 1,686 |
| **total** | **73** | **107,131** | **2,670** | — | **7,025** |

At the median `dt = 0.6000188 s`, this is approximately 17.9 hours of
recorded behaviour.  Neuron counts range from 133 to 182.

## Event inventory and class imbalance

Events are 0-based, end-exclusive reversal intervals.  The primary turn-
direction set keeps reversals of at least 12 frames with a valid two-frame
post-offset mean.

| cohort | long events | dorsal | ventral | near-zero | animals with both classes |
|---|---:|---:|---:|---:|---:|
| WT octanol | 596 | 100 | 496 | 18 | 27/32 |
| WT diacetyl | 349 | 86 | 263 | 11 | 23/24 |
| tdc1 octanol | 282 | 83 | 199 | 13 | 15/17 |
| **WT total** | **945** | **186** | **759** | **29** | **50/56** |

The WT dorsal fraction is 19.7%.  Six WT animals contribute only one class and
therefore cannot contribute a per-animal AUROC.  This imbalance is why the
primary metric is AUROC/AUPRC with animal-cluster uncertainty, not accuracy.

## Direction convention

The archive does not provide explicit dorsal/ventral labels.  The field `dv` is
the orientation-corrected D/V head angle; positive `dv` is dorsal.  The raw
`head_angle` field has an arbitrary camera sign that is constant within a
recording: across recordings, `dv` is either `+head_angle` or `-head_angle`.
`orientation_sign` is retained in the inventory.  Using raw `head_angle` across
animals would silently mix D/V.

This mapping is supported by marker-neuron activity (causal z-score, last four
frames before offset, WT diacetyl):

| marker | pre mean dorsal | pre mean ventral | difference (D−V) | observed sign |
|---|---:|---:|---:|---|
| SAAV | 0.841 | 1.135 | −0.294 | ventral-higher |
| RMDV | −0.037 | −0.225 | +0.188 | dorsal-higher |
| RMDD | −0.191 | −0.344 | +0.153 | dorsal-higher |
| SMDD | −0.051 | −0.679 | +0.628 | dorsal-higher |
| SMDV | −0.672 | −0.505 | −0.168 | ventral-higher |
| RIV | −0.544 | +0.246 | −0.790 | ventral-higher |

The observed pattern is qualitatively compatible with previously reported
roles for these head-steering neurons.  This is a sanity check, not a decoding
result and not proof that every event is labelled correctly; the audit does
not independently establish those roles.

## Neuron coverage

- 115 unique identified labels; 7,025 labelled traces.
- Only 10 labels are present in at least 80% of WT animals.
- Prespecified markers are sparse: SAAV 27 WT animals (14 octanol, 13
  diacetyl), RMDV 12, RMDD 16, SMDV 9, SMDD 31, RIV 12, SMBV 34, SMBD 35.
- The full identified-neuron model therefore uses fold-level labels with
  training-only coverage; missing test labels are zero after causal
  z-scoring.  This is not equivalent to measuring those neurons in every
  animal.

## Trace source

The adapter reads `gcamp/trace_array_original` from each HDF5 file.  The MATLAB
`gcamp_trace` equals the HDF5 `gcamp/trace_array` (whole-recording z-score) and
is not treated as raw.  This distinction matters because causal z-scoring must
not use future frames.

## What the archive does not contain

- No trial-level single-cell optogenetic recordings.
- No explicit per-event D/V labels; labels are inferred from `dv`.
- No pre-extracted SAAV/HC CSV files used by the published decoding code.
- No neural identities in the HDF5 files; identities come from the MATLAB
  structures.
- tdc1 is a mutant cohort, not a perturbation experiment.

## Audit outputs

- `reports/phase4/dataset_inventory.csv`
- `reports/phase4/cohort_inventory.csv`
- `reports/phase4/neuron_coverage.csv`
- `reports/phase4/behavior_event_inventory.csv`
- `reports/phase4/marker_sanity.csv`
- `reports/phase4/data_checksums.csv`
- `reports/phase4/dataset_audit_summary.json`
