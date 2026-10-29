# Benchmark card: BoundedBench (NeuroBranch 0.2.0)

A benchmark card states what a benchmark measures, what it does not measure,
and how a reader can reproduce it. This card covers the Phase 6-9 evaluation
layer.

## Dataset

| Field | Value |
| --- | --- |
| Source | Randi et al., *Nature* 2023, optogenetic whole-brain atlas |
| Accession | `10.1038/s41586-023-06683-4` |
| Local table | `reports/phase3/randi_trial_table_phase3.csv.gz` |
| Trials | 237,639 |
| Animals | 110 |
| Targets | 259 |
| Neurons | 305 |
| License | see the source publication and repository terms |

The table is derived, not raw: each row is one stimulation event with a
pre-computed response window, baseline and z-score. The derivation is recorded
in `docs/DATA_PROVENANCE.md`.

## Tasks

1. **Pair-level profile prediction.** Given a target and a responder neuron,
   predict the mean response from training animals and rank pairs. Scored with
   Spearman correlation against held-out animals.
2. **Animal-held-out evaluation.** All events from an animal are held out
   together. No event from a test animal appears in training.
3. **Trajectory prediction.** Predict the response time course for a held-out
   animal; scored by correlation and empirical coverage.

## Splits and leakage controls

- Folds are **animal-disjoint**. Event-level random splits are not used and are
  considered invalid for this benchmark.
- Targets with fewer than six animals are excluded from across-animal claims.
- Responder sets are computed inside the training fold only; a responder list
  derived from all animals would leak test information.
- Hyperparameters are selected inside the training fold. Phase 8 used five
  folds; the fold assignment is seeded and recorded in the Phase 8 manifest.
- The reliability ceiling is estimated on the full atlas, which is itself a
  leakage risk for a strict benchmark: the ceiling should ideally be estimated
  on training animals only. This is a known limitation, documented in
  `docs/PHASE9_RESULTS.md`.

## Baselines

| Baseline | What it controls for |
| --- | --- |
| Static CRO | Pair-level association without adaptation |
| Per-responder template | Animal-independent response shape |
| Persistence | No-change forecast for trajectories |
| Permutation | Chance level for rank correlation |

A model must beat the strongest baseline, not the weakest, and must be compared
against the measurement ceiling.

## Metrics

- Spearman correlation at pair level (primary).
- Trajectory correlation and empirical coverage (secondary).
- Reliability and attenuation ceiling (context, not a model score).
- MDE and TOST equivalence (design resolution, not a model score).

## Uncertainty

- Phase 8 used animal-level folds and reported 95% intervals.
- Phase 9 reports target-level intervals for aggregate reliability; these
  ignore animal sharing and are labelled descriptive.
- No per-target reliability interval is reported because the estimator is a
  variance ratio with few animals per target.

## Known failure modes

1. **Low ceiling.** Single-animal reliability is 0.223; a held-out correlation
   above ~0.47 is not attainable with this atlas.
2. **Few events per target.** Targets with few stimulations have unstable
   profiles; the reliability table reports counts so readers can filter.
3. **Target selection.** Exclusion of low-coverage targets changes the target
   population and must be reported.
4. **Response-window dependence.** Reliability is estimated for the mean
   response in a fixed window; other summaries may differ.
5. **Stimulation artefact.** Optogenetic stimulation can produce direct
   artefacts; the pipeline excludes the stimulation frame but not all
   artefactual dynamics.

## Intended use

- Comparing models under an explicit measurement ceiling.
- Designing the next experiment (how many animals, stimulations, targets).
- Teaching leakage-safe evaluation on a real perturbation atlas.

## Out of scope

- Causal claims about the worm's nervous system.
- Clinical or translational claims.
- Claims about individual animals beyond the estimated reliability.
- Any statement that a model is a digital twin or a foundation model of
  *C. elegans*.

## Reproduce

```bash
python scripts/phase9_reliability.py
python scripts/phase9_claims.py --check
python scripts/validate_reports.py --strict
```

The Phase 8 evaluation is reproduced by `scripts/phase8_run.py`; see
`docs/PHASE8_RESULTS.md`.
