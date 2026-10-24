# Phase 5A: Kramer SAAV decoding source and reconstruction audit

## Status

**`BLOCKED_MISSING_INPUTS` for exact reference reproduction.**  
**`PARTIAL_REPRODUCTION` for the raw-Dryad reconstruction.**

The four aligned CSVs used by the official code are not in the public code
repository and are not in the Dryad archive. A method reconstruction from raw
HDF5 data is possible, but it cannot be called an exact reproduction.

## Primary sources inspected

| source | version / identifier |
|---|---|
| Kramer et al., *Nature Neuroscience* (2026) | <https://doi.org/10.1038/s41593-026-02257-5> |
| Kramer et al., bioRxiv preprint | <https://doi.org/10.1101/2024.08.11.607076> |
| Official code | <https://github.com/flavell-lab/SAAV_Decoding>, commit `c7e7e18a84cde5dc02df9d63492293b38402eb2c` |
| Dataset | Dryad `doi:10.5061/dryad.8sf7m0cz2`, dataset `135272`, version `5`, version id `433175` |

The official code is **GPL-3.0**. It is kept outside the tracked NeuroBranch
tree during source inspection. NeuroBranch contains an independent
architecture-compatible reimplementation; no GPL source file is copied into the
tracked package.

## Source hashes

| file | bytes | SHA-256 |
|---|---:|---|
| `load_data.py` | 859 | `7b472f5906394f274521a69954bcc8195660b256e85e20e4850b979c7289e38f` |
| `model.py` | 1,191 | `a5a08ab99fefc797b25781f09f19f8c36ed0ab28fd646048d69d3eadaee41181` |
| `train.py` | 5,712 | `2af6982a966437693cea1deec313592aaf34ddc4a2d914464f19af86a71691f9` |
| `kfold.py` | 1,875 | `e32b4fb44222bc5098517055e156bc9ce9ee8c8aa200a7ae45f55db25809281d` |
| `util.py` | 2,941 | `bd96a53a00aa0e4883ab89cee0fa4d4619c47060dcefa73b45680b3894b8b807` |
| `DecodingMain.ipynb` | 2,348,729 | `68c68069da94acd0081f2139ce5580c1994db7dac2052c83e572740371120543` |
| `LICENSE` | 35,823 | `230184f60bae2feaf244f10a8bac053c8ff33a183bcc365b4d8b876d2b7f4809` |

## Missing inputs

`load_data.py` reads, without fallback:

```text
/store1/prj_rim/Decoding_Data/FlossieAligned/SAAV_dorsal.csv
/store1/prj_rim/Decoding_Data/FlossieAligned/HC_dorsal.csv
/store1/prj_rim/Decoding_Data/FlossieAligned/SAAV_ventral.csv
/store1/prj_rim/Decoding_Data/FlossieAligned/HC_ventral.csv
```

The repository contains only the notebook, five Python files, two EPS figures
and the licence. No CSV, archive, checksum, generation script, animal identifier
table or fold manifest is public. Dryad contains raw HDF5 recordings and cohort
MATLAB structures, not these aligned arrays.

A concise request for the missing files is in
[`PHASE5_MISSING_DATA_REQUEST.md`](PHASE5_MISSING_DATA_REQUEST.md).

## Exact official contract, read from source

### Data shape and channels

`load_data()` creates:

```text
positive: (n_dorsal_events, n_frames, 3)
negative: (n_ventral_events, n_frames, 3)
channel 0 = SAAV
channel 1 = head curvature (HC)
channel 2 = unused zeros
```

The notebook's display uses `data[:, 38:50]` and plots the reversal end at
frame 49. The displayed 12 frames are **not** the full model input: `train.py`
passes the full event trace through `extract_all_sections`, generating all
contiguous four-frame sections. For a 50-frame trace this is 47 sections per
event. A single event therefore contributes many highly overlapping samples.

### RNN

- `eqx.nn.GRUCell(in_size=2, hidden_size=3)`
- hidden state initialised to zeros
- `lax.scan` over the four input frames
- dropout `p=0.5` applied to every hidden state in the scan
- `LayerNorm(3)` applied to each hidden state
- `Linear(3 → 1, bias=True)` plus a separate zero-initialised output bias
- sigmoid at every timestep
- output shape `(time, 1)`, not one pooled event output

The notebook uses `hidden_size=3`, `time_range=4`, `epochs=250`, `batch_size=16`,
`learning_rate=1e-3`, `k_folds=5`, and `channel_selection=[[0,1],[2,1]]`
(`SAAV+HC` and `HC-only`; in the second channel set, channel 0 is the zero
channel).

### Training

- Adam, learning rate `1e-3`.
- Each epoch shuffles positive and negative event arrays and takes
  `min(n_positive, n_negative)` events from each class.
- All four-frame sections of the selected events are expanded.
- Sections are concatenated, shuffled, and batched at size 16.
- Validation uses the same class-balanced construction.
- The training script records train/validation loss and accuracy at every epoch.
- The notebook later chooses the epoch with minimum validation loss.

There is a source-level label/feature shuffle bug: `establish_batches` uses
`lax.select(shuffle, train_ys, train_ys[perm])`, which aligns labels with
features only when `shuffle=False`. The notebook uses `shuffle=False`, so the
published run is on the correct branch. NeuroBranch uses an explicit aligned
shuffle and documents this as a source-code defect, not as a result-changing
difference for the published setting.

### Cross-validation

- Five outer folds are created by rotating each class array by
  `floor(n/5) * fold_index`.
- Within each outer training partition, four inner rotations create 75:25
  train/validation splits.
- This yields 20 train/validation/test models, each outer test set evaluated by
  four models.
- The test set retains the natural class ratio; training and validation are
  class-balanced.

### Evaluation

- Training and validation loss: binary cross-entropy, labels broadcast over the
  four timesteps.
- Test accuracy: for each section, predictions at all four timesteps are
  compared with the event label and averaged; section accuracies are then
  averaged within dorsal and ventral classes and the two class means are
  averaged.
- The reported value is therefore **section-level accuracy**, not event-level
  AUROC, not event-level log loss and not animal-disjoint accuracy.
- The bootstrap resamples test sections for the empirical P value. The code does
  not cluster by animal.

This distinction is central. The published positive result is compatible with a
small section-level accuracy gain even if event-level or new-animal performance
is weak. Phase 5 tests that directly.

## Why exact reproduction is blocked

1. The four aligned CSVs are absent.
2. Their column order, frame count, NaN policy, z-score fit and event-to-animal
   mapping are not documented in the repository.
3. The notebook's display subset is not the model input; no exported split
   manifest, seed history or per-fold predictions are available.
4. The code's NaN handling is unspecified. Raw event traces with missing neurons
   cannot be passed to the published loss without an undocumented complete-case
   or imputation rule.
5. The paper reports group-level results, not trial-level data or code outputs.
6. No author response or archived input has been located.

## Raw-Dryad reconstruction

### Data used

- `gcamp/trace_array_original`, not the published whole-recording z-score.
- `behavior/velocity`, `behavior/angular_velocity`, `behavior/pumping`,
  `behavior/head_angle`, MATLAB `dv` and reversal events.
- SAAV identity from the MAT `wbData` labels, ambiguous labels excluded.
- WT octanol and WT diacetyl cohorts. tdc1 is excluded from reconstruction
  because the published decoding analysis is wild type.

### Head-curvature phase alignment

1. Smooth `dv` with a one-frame Gaussian filter.
2. Detect sign changes and linearly interpolate each zero-crossing time.
3. Reject crossings whose adjacent half-cycle is shorter than 0.25 s or longer
   than 20 s.
4. Assign phase `0, π, 2π, ...` to successive crossings.
5. At the last reversal frame, compute the instantaneous phase.
6. Build 48 target phases at `π/2` spacing backwards from that phase
   (four samples per cycle, 11.75 cycles).
7. Invert the phase track by linear interpolation between crossings and sample
   SAAV and HC at those times.

This follows the paper's description of aligning to dorsal→ventral and
ventral→dorsal crossings at a uniform 4.8 s head-swing period, with four samples
per cycle. The exact author interpolation, frame count and endpoint convention
are unknown, so this is a reconstruction.

### Outcome and inclusion

- Reversal duration ≥12 frames (1.5 head swings at the archive's ~0.6 s frame
  interval).
- Valid two-frame post-turn window.
- Unique unambiguous SAAV identity.
- Finite 48-frame phase-resampled SAAV and HC.
- `|post-turn mean dv| ≥ 0.05`; near-zero outcomes are excluded rather than
  assigned a direction.
- Dorsal = positive post-turn `dv`; ventral = negative.

### Normalisation

For the original-style reconstruction, SAAV and HC are z-scored over the full
recording. This mimics the published "z-scored activity" display but is
retrospective: the normalisation uses frames after the prediction point. It is
used only for offline protocol comparison and is never used in the online
benchmark. The online benchmark fits scaling statistics on training animals only.

### Architecture-compatible implementation

NeuroBranch implements the same GRU/LayerNorm/dropout/linear/sigmoid sequence
contract in PyTorch. Differences from the official JAX/Equinox code:

- framework and GRU kernel implementation;
- batch size and epoch budget (reduced for CPU feasibility; documented in the
  results);
- explicit random section sampling per event per epoch to control compute;
- optimisation/early-stopping details;
- no claim of bitwise or numerical equivalence.

These differences are sufficient to require `PARTIAL_REPRODUCTION`, even if the
qualitative pattern matches.

## What would upgrade the status

To reach `EXACT_REFERENCE_REPRODUCTION`, obtain the four CSVs plus:

- the preprocessing script that generated them;
- animal, recording and event identifiers for every row;
- the exact event inclusion and NaN rules;
- z-score fit scope and head-curvature definition;
- the original fold assignments and random seeds;
- per-event and per-section model predictions from the published run.

Without those, Phase 5 can test protocol sensitivity but cannot adjudicate
whether the published numerical result is reproducible.
