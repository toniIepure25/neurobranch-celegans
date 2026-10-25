# Phase 6 external transfer status

> **MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.** No transfer
> result below is an interventional or cross-dataset causal claim.

## Bottom line

No external cohort with the **same trial-level optogenetic stimulation task** is
present in this checkout. Phase 6 therefore does **not** claim cross-dataset
transfer. The transfer contract and validator are implemented, but no result is
fabricated from a dataset with a different task or from unlabelled spontaneous
activity.

## What is available locally

| Cohort | Local status | Why it cannot be the primary transfer test |
|---|---|---|
| Randi 2023 | 110 animals, 3,371 stimulation events | Training cohort; animal-disjoint split is internal, not external |
| Kato 2015 | 5 wild-type immobilised worms | Spontaneous activity, no controlled single-cell stimulation |
| Kramer 2026 | Behaviour/turning cohort | No trial-level single-cell optogenetic perturbation |
| Baskoylu 2026 | Behaviour cohort | No controlled perturbation; external-only |
| Simeon 2024 | Not downloaded | Spontaneous calcium; different task and standardisation |

## What would count as transfer

A candidate table must contain the same minimal schema:

```text
dataset_id, event_id, target, neuron, response_mean
```

and preferably `response`, `pre_activity`, `pre_sd`, `prev_stim_interval`,
`prev_same_target`, `event_order`. It must have repeated stimulations, identified
targets and responders, and a licence that permits evaluation. The validator is:

```python
from neurobranch.benchmark.external import audit_external_table

audit = audit_external_table("data/raw/external/trials.parquet")
print(audit.as_dict())
```

`audit.compatible` is necessary but not sufficient: the cohort must also pass the
connectome-coverage, label-ambiguity and licence checks before any claim.

## Why the Kato “forecasting transfer” is not run as a headline

The preregistered secondary analysis asks whether the CRO effect matrix improves
Kato one-step forecasting over persistence/VAR/PCA/GRU. That is a legitimate
falsification test, but it is a **different endpoint** and would not validate the
optogenetic response operator. It is left as an explicit next experiment rather
than silently relabelled as transfer. Prior Phase 5 results already showed that
behavioural/neural influence adds little to animal-disjoint turn prediction; the
expected result is negative.

## Required next data

To make a real cross-dataset claim, we need one of:

1. an independent lab's trial-level optogenetic atlas with the same target and
   responder identifiers;
2. a held-out genotype/condition in Randi that was not used for any Phase 1–6
   fitting, with the same stimulation protocol;
3. a preregistered prospective replication in a new cohort.

Until then the correct statement is: **the CRO generalises across held-out
targets within Randi, not across datasets.**
