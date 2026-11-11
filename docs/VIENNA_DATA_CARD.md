# Vienna Neuroinformatics Studio — data card

## Inputs

| Dataset | Content | Unit | Source / license |
|---|---|---|---|
| BunDLe-Net consistent embeddings | 3-D latent state + 8 behaviour labels per frame | worm (5) | NC-MCM `precomputed_results/bundlenet_consistent_embedding_worm_*.npz`; upstream license unspecified |
| NC-MCM precomputed models | activity, 20 cluster solutions, behaviour, pairwise probabilities, Markov p-values | worm (5) | NC-MCM `precomputed_results/nc_mcm_model_worm_*.npz`; GPL-3.0 |
| Celegans_premotor `avg_model_weights.jld2` | 205×205 chemical `A`, drive `d` | network (1) | Morrison & Young release; MIT |
| Celegans_premotor connectome table `df5.csv` | chemical/gap support and neuron names | network (1) | same release; MIT |
| Atanas recordings | calcium traces, NeuroPAL labels, frame interval | recording (30) | same release; MIT |

## Preprocessing

- BunDLe-Net: published consistent embeddings used as distributed; no refitting in the UI.
- NC-MCM: published cluster assignments and behaviour labels; 1-based NC-MCM labels mapped to the
  canonical 0-based BunDLe-Net order.
- Premotor: `A`/`d` used as published. `W` is reconstructed from the release connectome and the
  accepted 205-name order; 913/913 non-zero `A` entries lie inside the chemical support.
- Atanas: Savitzky–Golay `window=11, polyorder=5`; first and last smoothed frame removed
  (`ts_tmp[2:end-1]`); frame interval rounded to three decimals as in the notebook; documented
  AVBR/RIBR/AIB/RID/twin replacements only. No generic nearest-neighbour fill.
- Tier-3 behaviour target: neural readout proxy exactly as in notebook cell 20:
  `F = mean(AVBL, AVBR, RIBL, RIBR, RID)` and
  `R = mean(AVAL, AVAR, RIML, RIMR, AVEL, AVER)`, threshold 0.5. It is not an independent
  video annotation; an external check against the release's `reversal_events` is reported
  separately (readout mean AUC 0.848, 75.9% binary agreement; clamped model lag-0 AUC 0.633,
  retrospective +7.7-frame AUC 0.834, prospective AUC 0.388 (exploratory negative-lag max
  0.526–0.627), persistence AUC 0.939). The
  `reversal_events` interval convention and the five-frame trace offset are inferred, not
  documented, so offset 3–7 and closed/half-open intervals are reported as sensitivity analyses.
  Three validation recordings (84, 85, 86 measured core neurons) are used; only the canonical one
  has the published 84/15/69 split.

## Splits and leakage

- Tier 1: leave-one-worm-out. Fitting, feature standardisation, behaviour readout, and residual
  noise calibration use training worms only.
- Tier 2: no supervised train/test split; transport is held-out-worm behaviour statistics only.
  Cluster IDs are not aligned across worms and are never compared directly.
- Tier 3: the canonical recording is used as a replay input, not as a training label. The recovered
  `W` is built from the published connectome, not from the recording being simulated.

## Identifiers and privacy

All data are public research data from non-human animals. No human subjects, personal data, or
clinical identifiers are present.

## Redistribution

Third-party artifacts are **not** committed. Licenses: NC-MCM GPL-3.0; Celegans_premotor MIT;
BunDLe-Net unspecified. Review before redistribution. The fetch script records SHA-256 hashes in
`data/processed/vienna_artifacts_manifest.json`.
