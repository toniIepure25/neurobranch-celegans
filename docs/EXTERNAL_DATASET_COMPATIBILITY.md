# External dataset compatibility

**Status:** readiness audit, 2026-10-31. This page records what can be checked
from the public files and what cannot. It does not promote any external result
to validation.

## Scope and evidence rules

The question is whether NeuroBranch can be evaluated on additional *C. elegans*
recordings without choosing models on the already reused Kato NoStim test block.
Three protocols are distinguished:

- **A — frozen transfer:** apply an existing fitted model, scaler and support
  model without refitting anything.
- **B — per-animal refit:** fit on an animal's training segment and evaluate on
  its own untouched future segment.
- **C — cross-animal model-family generalization:** fit a shared model family
  across animals with an explicit, one-to-one neuron mapping and
  leave-one-animal-out evaluation.

A dataset is not independent merely because it has a different filename. Oxygen
concentration is an environmental variable. It is **not** targeted neuronal
stimulation and cannot support a causal intervention claim.

Evidence was read from the raw OSF HDF5 files, the associated OSF metadata, the
Kato and Nichols papers, and the dLDS paper that reuses the Kato data. The dLDS
reference is Yezerets et al., *Communications Biology* 2025,
[10.1038/s42003-025-08599-3](https://doi.org/10.1038/s42003-025-08599-3),
PMID 40804171. Raw files remain outside Git; hashes are in
`reports/external_validation/dataset_inventory.json`.

## Inventory (actual files)

| Dataset | Recording sessions | Duration | Sampling | Traces/session | Labelled traces | Unique trace labels | Behaviour states | Oxygen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Kato 2015 `WT_Stim` | 7 | 719.6–719.7 s | 2.80–3.06 fps | 107–151 | 37–51 | 45–52 | 4 coarse: FWD, REV, REVSUS, TURN | 21% for 0–360 s; 4/21% steps every 30 s from 360 s |
| Kato 2015 `WT_NoStim` | 5 | 1079.6–1079.7 s | 2.80–3.07 fps | 109–135 | 31–49 | 46 common names/IDs | 8 fine states | constant hyperoxia, no switches |
| Nichols 2017, four OSF files | 44 | 1079.6–1079.8 s | 2.81–4.58 fps | 80–130 | 23–44 | 27–47 | 6 five-state labels including quiescence | 10/21% steps at 360 s and 720 s |

The Nichols count is **44 sessions**, not 40: `npr1_prelet` 10,
`npr1_let` 11, `n2_let` 12, `n2_prelet` 11. The OSF filenames identify
recording dates/letters, not animals.

### Kato 2015 `WT_Stim`

- Source: <https://osf.io/2395t/>; paper DOI
  [10.1016/j.cell.2015.09.034](https://doi.org/10.1016/j.cell.2015.09.034).
- Seven sessions, all about 720 s, whole-brain calcium imaging. The dLDS paper
  describes the preparation as immobilized worms and confirms 12 Kato worms in
  total: 5 NoStim and 7 Stim.
- The first 360 s are constant 21% oxygen. The second half alternates 4% and
  21% every 30 s. The oxygen schedule is stored in the file and is measured
  environmental input, not a targeted perturbation of identified neurons.
- Trace labels are not cell identities. A trace may have no label, one label,
  or several labels; a label can recur on several traces. Across the seven
  sessions there are 19 common trace labels and 82 in the union; pairwise
  Jaccard overlap is 0.468–0.655.
- Behaviour is inferred from AVA/SMD/RIB calcium activity. It is a supporting
  readout, not an independent behavioural observation.
- Animal independence is not documented. The sessions are different recording
  entries from NoStim, but OSF provides no animal IDs and no explicit
  non-overlap statement.
- Published dLDS Table 3 trace counts disagree with the raw HDF5 shapes for
  some worms. The raw shapes are authoritative in this repository and the
  discrepancy is recorded in the inventory.

### Kato 2015 `WT_NoStim`

- Five sessions, 1080 s, constant hyperoxia and eight fine behavioural states.
- Already used for Phase 5/6 training, validation and test. Test block 2 was
  inspected in Phase 5, so all Phase 6 test and gate evidence is exploratory.
- No untouched test segment remains for an external claim on these five
  sessions.

### Nichols 2017

- Source: <https://osf.io/kbf38/>; paper DOI
  [10.1126/science.aam6851](https://doi.org/10.1126/science.aam6851).
- 44 recording sessions during `PreLet` and `Let` lethargus, with
  `npr-1` and N2 genetic backgrounds and several reporter/imaging constructs.
- Six five-state labels include `quiescence`; the dataset is about a global
  sleep-like state, not the Kato pirouette assay.
- Oxygen is a binary 10%/21% arousing stimulus at 360 s and 720 s. It is not a
  targeted neuronal stimulation and is confounded with developmental state and
  genotype.
- Across all 44 sessions only **AVAL and RIS** are common named labels; union is
  87 names and pairwise Jaccard is 0.234–0.762. This is not a common neuron
  panel.
- OSF metadata does not document immobilization or a microfluidic preparation;
  the paper methods were not fully accessible during this audit. Do not fill
  that gap by assumption.
- The 44 sessions are not proven to be 44 independent animals and are not proven
  disjoint from Kato.

## Alignment and missing data

- Kato and Nichols both provide time vectors and calcium traces, but neither
  provides a shared ordered neuron panel suitable for frozen transfer.
- Numeric trace labels in Kato are worm-specific. Equal numbers do not imply
  equal neurons.
- Multi-neuron trace labels prevent a one-to-one cell mapping without an
  explicit deconvolution/assignment model.
- The Kato Stim and NoStim behaviour vocabularies differ (4 versus 8 states), so
  behaviour accuracy is not directly comparable.
- No explicit missing-data mask was found; varying frame counts reflect different
  sampling rates/durations. Any adapter must reject non-finite traces rather
  than impute silently.

## Compatibility decision

| Dataset | A frozen transfer | B per-animal refit | C cross-animal family | Oxygen-response claim |
| --- | --- | --- | --- | --- |
| Kato Stim | **BLOCKED** — no shared ordered feature space; checkpoints are 103–127-dimensional worm-specific VARs | **GO (limited)** — same lab/modality, constant 21% first 360 s, chronological split; executed as a pilot | **BLOCKED** — only 19 common labels across seven sessions and multi-label traces | **LIMITED** — oxygen is measured environmental input; no input model |
| Nichols 2017 | **BLOCKED** — different lab, stage, preparation and feature space | **LIMITED / not executed** — possible only with a separate quiescence/arousal protocol and metadata QC | **BLOCKED** — only AVAL/RIS common across 44 sessions | **BLOCKED/LIMITED** — arousing oxygen is not neuronal intervention |
| Kato NoStim | not applicable; already in-sample | no untouched test segment | **BLOCKED** | not applicable |

### Why frozen transfer is blocked

Phase 6 checkpoints contain coefficients for each NoStim worm's own ordered trace
panel (103, 127, 123, 118, 121 predictors). Kato Stim has 102–145 non-behaviour
traces in different orders, and Nichols has 75–125 after behaviour exclusion.
There is no common ordered vector, no common scaler, and no one-to-one identity
mapping. Applying the coefficients would be a numerical coincidence, not a
transfer test. The 12 common Kato Stim/NoStim names are not enough to define an
ordered panel; selecting them would be a new model and would require its own
protocol and test data.

### Why the Kato Stim B pilot is allowed

The first 360 s provide a constant-21%-oxygen segment with the same laboratory,
signal modality and general preparation as NoStim. A per-animal chronological
refit avoids the reused NoStim test block and does not require cross-animal
identity. It tests autonomous future prediction only. It does not test the
response to oxygen and cannot be described as sensory-input validation.

### What would change the decision

- A documented one-to-one neuron mapping with a shared annotation version and
  pre-specified missing-neuron handling could make C testable.
- A controlled, randomized sensory-input dataset with an identified input channel
  could make an input-response question testable, but only after the simulator
  has an exogenous-input path and a separate protocol.
- Animal IDs or a cohort manifest could establish independence; filenames alone
  cannot.

## Reproduction

Files and SHA256 values are in
`reports/external_validation/dataset_inventory.json`; session-level rows are in
`reports/external_validation/dataset_inventory_sessions.csv`. The inventory
script is `scripts/external_validation_inventory.py`. No raw data are committed.
