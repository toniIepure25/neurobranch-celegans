# Phase 2 next experiments

Ordered by scientific value per unit of risk. Each item states a falsifiable
prediction and a stop condition. None of these has been executed.

## 1. Independent behavioural generalization (highest priority)

**Why:** Phase 2 behaviour is still Kato, still AVA/SMD/RIB-derived, still
per-worm. The original request was independent behavioural data; it was not
acquired.

**Data:** an external *C. elegans* whole-brain recording with behaviour labels
not derived from the predictor neurons (e.g. published Zhang/Atanas/Kim
datasets, if licensing permits).

**Protocol:**

- freeze the Phase 1 preprocessing and model before touching the new dataset;
- map neuron names with the audited canonicalization, excluding ambiguous
  duplicates;
- evaluate state-only vs state+neural on the new lab/preparation;
- report calibration (log loss, Brier) and transition accuracy;
- no fine-tuning on the target dataset in the primary analysis.

**Falsifiable prediction:** state+neural beats state-only on log loss with a
paired animal-level CI excluding zero.

**Stop:** if the external dataset has <2 transition frames per animal, the
behavioural question is underpowered; report that and stop.

## 2. Causal-filter sequence model with a held-out encoder

**Why:** Phase 1 dynamics models collapse under forward split; BunDLe official is
leaky and BunDLeLite is partial.

**Protocol:**

- train the encoder only on training blocks, then freeze it;
- causal filtering only (no `filtfilt`), with filter warm-up excluded;
- strict forward split, multiple seeds, equal-compute persistence/VAR baselines;
- report one-step R² and rollout error vs persistence;
- pre-register the architecture and early-stopping rule.

**Falsifiable prediction:** a causal sequence model beats persistence by >0.02 R²
with a paired CI excluding zero on ≥4/5 worms.

**Stop:** if the model is worse than persistence on 2 consecutive seeds, stop and
report the negative result.

## 3. Branching with a prospective intervention

**Why:** the current statistic is a projection of observational data and has low
power for stochastic branching.

**Protocol:**

- choose candidate states from the corrected audit;
- apply a targeted optogenetic or temperature stimulus at those states;
- measure the outcome distribution with and without the stimulus;
- test whether the stimulus changes the **conditional outcome distribution**,
  not just the mean;
- use a pre-registered outcome code and a held-out animal.

**Falsifiable prediction:** stimulation at a candidate state changes the
outcome distribution relative to yoked sham stimulation.

**Stop:** no candidate state survives the corrected entropy audit with a stable
projection; do not proceed to intervention.

## 4. Intervention generalization with a pre-registered cell set

**Why:** the RIVL effect is nonspecific; the scan is post hoc.

**Protocol:**

- pre-register 5–10 cells and a dose–response range before scanning;
- include matched controls selected by anatomy/class, not by effect;
- test on a new animal/preparation;
- primary endpoint: effect at the pre-registered dose, with a hierarchical
  model across animals;
- report all cells, including nulls, and a max-statistic correction.

**Falsifiable prediction:** the pre-registered cell's effect exceeds the
pre-registered control distribution at the primary dose in a new cohort.

**Stop:** if the effect disappears at ±1 SD, it is an off-support artifact; stop.

## 5. Reproduce the Randi response analysis

**Why:** our response detector is simple and may be the limiting factor.

**Protocol:**

- run the authors' `pumpprobe` response pipeline at a pinned commit on the same
  archive;
- compare event-level responses and reliability;
- then rerun the benchmark with the authors' responses;
- if the authors' responses are more repeatable, re-evaluate predictability.

**Falsifiable prediction:** authors' responses have higher repeated-stimulation
correlation than our signed peak, and a pre-stimulus model then beats the
constant mean.

**Stop:** if reliability remains <0.2, prediction is ceiling-limited; report that
rather than tuning the model.

## 6. Hierarchical response model with animal-level partial pooling

**Why:** the current benchmark treats all rows as exchangeable within a split and
uses a global ridge.

**Protocol:**

- partial pooling of target, recorded cell and animal effects;
- reliability-weighted likelihood (down-weight noisy events);
- animal-held-out evaluation;
- compare to the constant-mean baseline with a paired event bootstrap.

**Falsifiable prediction:** hierarchical shrinkage improves held-out-animal MAE
over the global mean.

**Stop:** if the improvement is <0.01 MAE or the CI includes zero, stop.

## 7. Connectome-informed propagation with sign and path features

**Why:** raw edge counts may be too weak; the atlas has directed sign and
path-length structure.

**Protocol:**

- build directed, signed features (chemical/electrical, 1–3 hop paths);
- predict response sign and magnitude separately;
- test whether path features beat target identity alone;
- evaluate on held-out animals.

**Falsifiable prediction:** path features improve sign accuracy over target
identity by >0.05.

**Stop:** if target identity alone explains the signal, connectome features add
nothing.

## 8. Cross-dataset neuron registration

**Why:** same name is not same cell; duplicate labels are excluded.

**Protocol:**

- use the Funatlas / anatomical coordinates to register Randi and Kato neurons;
- resolve left/right duplicates with positional priors;
- quantify registration uncertainty and propagate it into the benchmark.

**Falsifiable prediction:** registered identities reduce duplicate exclusions and
improve held-out-animal prediction beyond name-only features.

**Stop:** if registration uncertainty exceeds the effect size, keep name-level
analysis.

## 9. Reporting and app

- Add the Randi benchmark tab to the app (done in Phase 2).
- Add a “negative results” page that shows the corrected branching deficit and
  the failed intervention specificity.
- Add a reproducibility check that recomputes all Phase 2 summary tables from
  artifacts and fails on checksum mismatch.

These are engineering tasks, not scientific priorities.
