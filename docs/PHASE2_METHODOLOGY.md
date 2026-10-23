# Phase 2 methodology

**Scope:** robustness of Phase 1 negative findings, corrected branching
statistics, intervention specificity, real optogenetic response prediction, and
incremental behavioural prediction.
**Baseline:** [`PHASE2_BASELINE_AUDIT.md`](PHASE2_BASELINE_AUDIT.md).
**Branch:** `research/phase2-intervention-validation`.

## 1. Principles

1. **Preserve the Phase 1 result, then challenge it.** New analyses are additive;
   old reports and artifacts are not overwritten.
2. **No test tuning.** Hyperparameters, candidate cells, splits and metrics are
   fixed before test evaluation. No test-fold model selection.
3. **No imputation of biological signal.** Missing or ambiguous traces are
   excluded and counted.
4. **Every virtual result is labelled.** The exact string
   `MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED` is attached to
   virtual-intervention outputs.
5. **Negative results are first-class.** A failed replication is reported with the
   same precision as a positive one.

## 2. Data

### Kato 2015

- `WT_NoStim.mat`, 5 worms, 15,662 frames, 629 segmented traces, 192 named
  traces, ~2.9 Hz.
- SHA256 `7531811c1718e26a0455f59cb2f8188cf25809278cfea85770dfe03315a0098a`.
- Behaviour is decoded from AVA/SMD/RIB; those eight cells are excluded from
  predictors.

### Randi 2023

- OSF `e2syt`, archive `9mecf`, SHA256
  `d6e7b3d93175b40b7ae17bde2182835e9c2144388142c522ee9be3832f6ce836`.
- 113 datasets, 110 with a valid event, 3,371 valid events, 110 animals.
- Response = signed peak ΔGCaMP over frames +1…+12 from a −8…−1 median baseline;
  stimulated trace excluded; ambiguous duplicate labels excluded.
- Details: [`INTERVENTION_DATASET_AUDIT.md`](INTERVENTION_DATASET_AUDIT.md).

## 3. Splits and leakage controls

### Dynamics and behaviour

- **Blocked:** five contiguous blocks; test = block 2, validation = block 4,
  train = blocks 0,1,3; 12-frame purge. This is temporal interpolation, not
  forecasting.
- **Strict forward:** train on the first contiguous fraction, test on the last;
  no future frames in training or preprocessing.
- Windows are generated **after** the split, so no window crosses a boundary.
- PCA, scaler, probe and behaviour classifier are fitted on train frames only.
- No test frame influences feature selection, early stopping or model choice.

### Branching

- PCA(3) fitted on train frames only.
- Candidate frames are held-out test frames in macro-state REV.
- kNN uses `k = 40` and excludes candidates within ±12 frames in time.
- Outcome horizon = 20 frames; outcome = first different macro-state.
- 500 permutations/shifts; block sizes 10/30/60; moving-block bootstrap for the
  mean-entropy CI.

### Randi benchmark

- **heldout_events:** one GroupShuffleSplit, 20% of events.
- **heldout_targets:** 5-fold GroupKFold by stimulated cell.
- **heldout_animals:** 5-fold GroupKFold by dataset.
- Groups are disjoint at the event/target/animal level as named.
- Encoders, scalers and shrinkage means are fitted on training rows only.
- Ridge `alpha = 10.0`; one-hot target/recorded cell + standardized
  pre-activity; connectome features are standardized edge counts.
- Cluster bootstrap resamples **events**, not rows, for CIs.

## 4. Branching statistic

For candidate frame `i`, let `N_k(i)` be its `k` nearest candidate frames in the
3-D projection, excluding temporal neighbours. Let `p_i` be the empirical
distribution of the first post-horizon macro-state among `N_k(i)`, and

```
H_i = - sum_c p_i(c) log2 p_i(c)
H   = mean_i H_i
```

Nulls decouple outcomes from geometry while preserving:

| Null | Preserved |
|---|---|
| iid permutation | outcome counts |
| block permutation | local runs of length 10/30/60 |
| circular shift | full outcome sequence and autocorrelation |

The two-sided p is `2 * min(P(null ≥ observed), P(null ≤ observed))`, capped at 1.
Synthetic scenarios with known ground truth test the detector before it is
applied to recordings. See [`BRANCHING_STATISTICAL_AUDIT.md`](BRANCHING_STATISTICAL_AUDIT.md).

## 5. Intervention specificity

- 13 candidate cells, amplitudes ±2 standardized units.
- 6 models (MLP seeds 17/23/31, residual MLP 17, VAR 17, GRU 17).
- 19 non-candidate controls per model, matched on amplitude, initial states,
  seeds, horizon and target behaviour.
- Readout: ΔP(TURN) vs paired no-intervention rollout.
- Uncertainty: cluster bootstrap over initial frames.
- Specificity: empirical percentile of the candidate against the control
  distribution, plus a selection-adjusted max-statistic test.
- Amplitude sensitivity: −3…+3 for six cells.
- Rank stability: Spearman and top-5 overlap across models.

## 6. Optogenetic response benchmark

Targets and baselines:

| Model | Information used |
|---|---|
| `zero` | none |
| `global_mean` | training mean only |
| `neuron_mean` | shrunk recorded-cell mean |
| `target_mean` | shrunk stimulated-cell mean |
| `additive` | shrunk target + recorded-cell effects |
| `onehot_ridge` | target + recorded cell + pre-activity |
| `connectome_ridge` | Witvliet edge counts + pre-activity |

The connectome is a **cell-type-level** feature table from the pumpprobe
reference, not a measured connectome for each animal. This is a baseline, not a
model reproduction.

A `response_residual` target subtracts each event's median downstream response
to probe global/illumination effects.

## 7. Behaviour benchmark

- Current-state one-hot logistic regression.
- State-only Markov transition model.
- Neural-only PCA(20) + logistic regression.
- State + neural logistic regression.
- Metrics: accuracy, balanced accuracy, macro F1, log loss, multiclass Brier,
  and accuracy on the transition subset.
- PCA and scaler fitted on training frames only.

## 8. Uncertainty and multiple comparisons

- Mean-entropy CI: moving-block bootstrap (block 18).
- Intervention CI: cluster bootstrap over initial frames.
- Randi CI: cluster bootstrap over events.
- Behaviour deltas: paired per-worm differences; no across-worm
  independence is claimed.
- Branching p-values are **not** corrected across worms/nulls; the pattern across
  worms is reported, not a single significant p.
- The intervention audit explicitly reports a selection-adjusted max-statistic
  p because 26 candidate cells were scanned.

## 9. Software

Python 3.13.5, PyTorch 2.7.1+cpu, NumPy 2.3.1, SciPy 1.16,
scikit-learn 1.9.1, pandas 2.x, Streamlit 1.66.1. Tests:
`python -m pytest`. Lint: `python -m ruff check src tests scripts`.

Summary figure: `figures/fig09_phase2_summary.png` (generated by
`scripts/phase2a_make_figures.py`).

## 10. What would falsify the conclusions- A causal-filter latent model that beats persistence under a strict forward
  split, with a properly held-out encoder.
- A branching statistic that shows significant **excess** over a circular-shift
  null and replicates across worms and projections.
- An intervention effect that survives matched controls and a pre-registered
  candidate set, with dose–response rather than a threshold at +3.
- An optogenetic benchmark where a pre-stimulus model beats the constant mean on
  held-out animals with a paired CI excluding zero.
- An external behaviour dataset where neural activity improves a state-only
  baseline.
