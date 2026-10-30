# Benchmark card: BoundedBench (NeuroBranch 1.3.2)

> **Phase 14 update.** The Phase 14A reference audit supports the coverage-aware
> ceiling **0.560** under a robust one-event variance treatment (V1 0.3006 versus
> V0 0.3136; delta 0.0131). The Phase 14B exact zero/spike-and-slab model is
> **rejected**: 8/12 frozen gates failed, including SBC rank, held-out PPC, both
> null q95 tails, prior sensitivity, reference agreement, and spike calibration.
> The machine-readable Phase 11 card at `reports/phase11/benchmark_card.json`
> remains authoritative for the descriptive ceiling. The aligned Phase 12R fit
> gives 0.304 / 0.551, but its interval is **not calibrated**. The Phase 12
> animal-cluster benchmark comparison is withdrawn because baseline/model truth
> was not paired on events (C11). See `docs/PHASE14_RESULTS.md`.
>
> **Phase 12R note.** The Phase 12R fit is retained as a historical negative
> result; it is not a replacement ceiling.

A benchmark card states what a benchmark measures, what it does not measure,
and how a reader can reproduce it. This card covers the Phase 6-10 evaluation
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
  `docs/PHASE10_RESULTS.md`.

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
- Phase 10 reports target-level intervals for aggregate reliability; these
  ignore animal sharing and are labelled descriptive.
- The animal jackknife interval is a design diagnostic. Its synthetic coverage
  study failed nominal coverage, so it is **not** a calibrated confidence
  interval.
- No per-target reliability interval is reported because the estimator is a
  variance ratio with few animals per target.
- The Phase 11 two-way percentile bootstrap failed its own diagnostic: the
  point estimate (0.314) fell outside the interval ([0.497, 0.583]) because
  resampling sparse responder rows inflates reliability. It is **not** a
  calibrated interval.
- The Phase 12 hierarchical posterior is also not calibrated. SBC coverage was
  0.923 (gate [0.93, 0.97]), rank uniformity failed (KS p = 1.4e-06),
  held-out coverage was 0.896 with median absolute error 0.191, and the
  animal-wise permutation null gave median R(22) = 0.176. The point estimate
  is reported as a model-based sensitivity result, not a calibrated interval.
- Phase 12 benchmark comparisons use an animal-cluster bootstrap with Holm
  adjustment. The largest paired delta was +0.189 (global mean vs static CRO),
  but MDE was 0.238 against a 0.05 SESOI and Holm p was 0.185. No
  configuration is supported as superior; fold-level leaderboard values remain
  descriptive.

## Known failure modes

1. **Facet-dependent ceiling.** Single-animal reliability is 0.162 (ceiling
   0.403); a per-animal correlation above ~0.40 is not attainable with this
   atlas. Pooled over 22 animals at 80% responder coverage, reliability is
   0.314 and the ceiling is 0.560 (the Phase 10 sparse-coverage values were
   0.693 and 0.833). Comparing a pooled metric to the single-animal ceiling is
   an error.
2. **Few events per target.** Targets with few stimulations have unstable
   profiles; the reliability table reports counts so readers can filter.
3. **Target selection.** Exclusion of low-coverage targets changes the target
   population and must be reported.
4. **Response-window dependence.** Reliability is estimated for the mean
   response in a fixed window; other summaries may differ.
5. **Stimulation artefact.** Optogenetic stimulation can produce direct
   artefacts; the pipeline excludes the stimulation frame but not all
   artefactual dynamics.
6. **Unstable aggregate.** The delete-one-animal jackknife spans [0.165, 1.000]
   for the fold-level mean; a single animal can move the aggregate enough that
   the interval is uninformative. Report it, do not lean on it.
7. **Unstable target ranking.** Per-target reliability rankings do not
   replicate across animal halves (median Spearman about 0.00). Do not
   prioritise targets by estimated reliability. Phase 12 did not produce a
   ranking; the hierarchical model's target-level posterior is unpublished and
   its planning table is conditional.
8. **Positive null bias.** Permuting responder labels within each animal still
   yields median R(22) = 0.176. The model can manufacture apparent reliability
   from coverage and marginal structure, which is why the negative control is a
   required gate rather than a formality.

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
python scripts/phase10_reliability.py
python scripts/phase10_robustness.py
python scripts/phase10_design.py
python scripts/phase10_claims.py --check
python scripts/phase12_hierarchy.py
python scripts/phase12_benchmark.py
python scripts/phase12_manifest.py
python scripts/validate_reports.py --strict
python scripts/check_reproducibility.py
```

The Phase 8 evaluation is reproduced by `scripts/phase8_run.py`; see
`docs/PHASE8_RESULTS.md`.
