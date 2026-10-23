# Intervention dataset audit: Randi et al. (2023)

**Status:** downloaded, checksum-verified, parsed, and used for a held-out
response benchmark. Raw files are **not committed**.
**Data:** `data/raw/randi2023/exported_data/`
**Adapter:** `src/neurobranch/data/randi.py`
**Coverage artifact:** `reports/phase2b_optogenetic_atlas/randi_coverage.json`

> This dataset is experimental. It is used only to test whether a model fitted
> on observational Kato data, or a simple response model, predicts independently
> measured optogenetic responses. It is **not** used to fit NeuroBranch dynamics,
> and it does **not** make any virtual perturbation causally validated.

## 1. Provenance and integrity

| Item | Value |
|---|---|
| OSF project | [`e2syt`](https://osf.io/e2syt/) — “Neural signal propagation atlas of C. elegans” |
| Paper | Randi et al., *Nature* (2023), [doi:10.1038/s41586-023-06683-4](https://www.nature.com/articles/s41586-023-06683-4) |
| Archive | `exported_data.tar.gz`, OSF file GUID `9mecf` |
| Size | 523,093,816 bytes |
| SHA256 (computed) | `d6e7b3d93175b40b7ae17bde2182835e9c2144388142c522ee9be3832f6ce836` |
| SHA256 (OSF API) | `d6e7b3d93175b40b7ae17bde2182835e9c2144388142c522ee9be3832f6ce836` |
| OSF metadata hash checked on | 2026-10-23 |
| Dataset 0 GCaMP SHA256 | `45519b52ab43a9e64d8dbcf105ea0852bba4582e24609d3a0832ed4788b30345` |
| Dataset 0 labels SHA256 | `e6ea5c5452035b746ae5e8e7ecf42d7696ab2dff78c51809e5a5782811791a7e` |
| Reference code | [`leiferlab/pumpprobe`](https://github.com/leiferlab/pumpprobe), GPL-3.0 |

A previously recorded candidate hash `3da1f0…` did **not** match the downloaded
archive. The OSF API independently reports the computed hash above, so the
mismatch was in the earlier note, not in the download. Only this verified hash
should be cited.

The OSF node API does not expose a license field. The archive is publicly
downloadable, but raw files are kept out of Git. `pumpprobe` itself is GPL-3.0;
this repository does not vendor it.

## 2. Files and schema

Each dataset `<id>` has:

| File | Shape / meaning |
|---|---|
| `<id>_gcamp.txt` | frames × traces; processed GCaMP signal; may contain NaN |
| `<id>_labels.txt` | one label per trace; blank, numeric, and ambiguous entries exist |
| `<id>_t.txt` | frame times, seconds; median Δt = 0.5 s |
| `<id>_stim_neurons.txt` | trace index of the stimulated neuron per trial |
| `<id>_stim_volume_i.txt` | frame index of each stimulation |
| `<id>_ds_name.txt` | original acquisition path |

Negative sentinels in `stim_neurons`: `-2` means no targeted hit / discarded and
`-3` means a complementary/ambiguous label. They are **not** neuron indices and
are excluded. Blank, numeric, `merge`, `RIG` (ambiguous between RIGL/RIGR),
unnumbered ventral-cord classes, and duplicate labels are also excluded or
flagged.

## 3. Coverage actually parsed

| Quantity | Value |
|---|---:|
| Source datasets | 113 |
| Datasets with ≥1 valid downstream event | 110 |
| Datasets with no valid event | 3 (`11`, `24`, `29`) |
| Total frames | 394,084 |
| Total segmented traces | 14,379 |
| Raw unique labels | 357 |
| Canonical unique labels | 314 |
| Duplicate-label traces excluded | 212 (34 datasets) |
| Stimulation events in files | 5,808 |
| Valid events with unambiguous target | 3,371 |
| Long-form downstream rows | 237,639 |
| Unique animals | 110 |
| Unique stimulated cells | 259 |
| Unique recorded cells | 305 |
| Unique target–neuron pairs | 40,341 |

The 3,371 / 5,808 ratio is the cost of requiring a named, unambiguous stimulated
cell. This is reported, not hidden by imputation.

## 4. Response target

For each valid event at frame `i0`:

- baseline = median of frames `i0-8 … i0-1` (4 s);
- post window = frames `i0+1 … i0+12` (0.5–6 s after onset);
- response = signed value at the maximum absolute deviation from baseline;
- the stimulated neuron's own trace is **excluded** (direct optogenetic /
  fluorescence effect is not a downstream propagation measurement);
- traces with non-finite baseline or post samples are skipped, never imputed;
- `response_residual` subtracts the event's median downstream response, a
  sensitivity target for global/illumination effects.

This is a deliberately simple, auditable detector. It is **not** the authors'
response model, and it is not claimed to reproduce their effect sizes.

## 5. Behavioural / identity caveats

- Labels are trace annotations from the export, not a within-animal cell
  registration. Same string across datasets means same cell **type**, not the
  same individual cell.
- 57 of the 58 Kato named predictor cells occur in Randi. `VB03` is absent.
- Duplicate labels within a dataset (e.g. two `SAAD` traces) are ambiguous; all
  such traces are excluded rather than merged.
- No behavioural labels are present in this archive. It cannot answer the
  behaviour-prediction question.

See [`NEURON_IDENTITY_MAPPING.md`](NEURON_IDENTITY_MAPPING.md).

## 6. What this dataset can and cannot test

**Can test**

- Whether pre-stimulus activity / target identity / connectome features predict
  a held-out downstream response.
- Generalization to new stimulation events, new stimulated cells, and new
  animals.
- Repeatability of responses to repeated stimulation of the same target.

**Cannot test**

- Whether NeuroBranch's virtual perturbations match biological effects. The
  model was not fitted to these trials, and the benchmark is not an
  intervention-validated simulator.
- Behaviour, because no behaviour labels are included.
- Causal sufficiency or necessity, because the benchmark is predictive only.
