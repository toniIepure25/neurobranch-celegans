# Neuron identity mapping: Kato ↔ Randi

**Status:** audited; no within-animal registration is claimed.
**Code:** `canonical_neuron_name()` in `src/neurobranch/data/randi.py`
**Artifacts:** `reports/phase2b_optogenetic_atlas/randi_coverage.json`

## 1. What “mapping” means here

Kato et al. (2015) and Randi et al. (2023) are different animals, different
preparations, and different microscopes. A shared string such as `AVAL` means
the same **cell type / name**, not the same individual cell tracked across
sessions. There is no affine or non-rigid registration between the two datasets
in this repository. Every cross-dataset statement is therefore at the level of
named cell types.

This matters for the optogenetic benchmark: a model that uses `AVAL` as a
feature is using the label, not a registered cell identity.

## 2. Canonicalization rules

1. Strip whitespace.
2. Reject blank and purely numeric labels.
3. Reject placeholders and ambiguous strings: `smthng else`, `unknown`, `none`,
   `nan`, `merge`, `check`, `mah`, `target`, `RIG` (ambiguous RIGL/RIGR).
4. Reject unnumbered ventral-cord / AS classes (`DA`, `DB`, `DD`, `VA`, `VB`,
   `VC`, `VD`, `AS`) because the index is missing.
5. Zero-pad numbered ventral-cord names to two digits:
   `VB1 → VB01`, `DB2 → DB02`, `DA1 → DA01`, `VA1 → VA01`, `DD1 → DD01`,
   `AS1 → AS01`.
6. Otherwise preserve the label exactly. No fuzzy matching or synonym guessing
   is performed.

Aliases observed in the export (counts are trace occurrences):

| Raw | Canonical | Occurrences |
|---|---|---:|
| `VB1` | `VB01` | 75 |
| `VB2` | `VB02` | 52 |
| `DB2` | `DB02` | 32 |
| `DA1` | `DA01` | 26 |
| `VA1` | `VA01` | 19 |
| `DD1` | `DD01` | 19 |
| `DB1` | `DB01` | 16 |
| `AS1` | `AS01` | 8 |

## 3. Coverage

| Quantity | Value |
|---|---:|
| Raw unique labels | 357 |
| Canonical unique labels | 314 |
| Kato named predictor cells | 58 |
| Kato names present in Randi | 57 |
| Kato name absent from Randi | `VB03` |
| Duplicate-label traces excluded | 212 |
| Datasets containing ambiguous duplicates | 34 |

The Kato named set is only 58 cells; the recordings contain many more segmented
and numeric traces. The 57/58 overlap is therefore a statement about the small
Kato predictor set, not about the full 302-neuron atlas.

## 4. Duplicate labels are ambiguity, not replication

Example: a dataset can contain two traces both labelled `SAAD`, or three labelled
`SMB`. The export does not say which is left/right or which segmentation is
which. Treating both as the same cell would silently duplicate rows and make a
neuron-mean baseline look better than it is.

The adapter therefore:

- keeps the duplicate traces in the raw recording;
- marks every trace whose canonical label occurs more than once in that dataset
  as `ambiguous`;
- excludes ambiguous traces from response targets;
- excludes events whose stimulated trace is ambiguous;
- reports the excluded count.

This is why the usable trial table has 3,371 events rather than 5,808.

## 5. Remaining ambiguity

Even after canonicalization, the following are **not** resolved:

- left/right or segment identity for any duplicate label;
- whether a label refers to a cell body, a process, or a merged segment;
- developmental or preparation differences between animals;
- one-to-one correspondence to the Witvliet connectome used as a feature.

The connectome features in `randi_benchmark.py` are therefore **cell-type-level
edge counts**, not measured connectivity for the recorded animal.
