# Phase 14 literature and methods scan

**Review type:** rapid, non-systematic methods scan. It is **not** a systematic
review and has no PRISMA screening counts. It was used to position the Phase 14
negative result, not to establish prevalence or effect sizes.

**Question:** given a repeated-stimulation atlas with sparse per-cell events, is
an exact zero/spike-and-slab measurement model a defensible way to calibrate
pooled reliability, and what must be reported when it fails?

**Decision supported:** whether to adopt M4 and release `v1.4.0`, or publish a
negative `v1.3.2` result. The answer was negative.

**Search date:** 2026-10-30 (UTC). **Databases:** PubMed, Europe PMC, OpenAlex,
Crossref. **Language:** English. **Date range:** no fixed lower bound; methods
sources from 2005--2025 and atlas sources from 2023.

## Search log

| # | Index | Query | Hits | Used |
|---|---|---|---:|---|
| 1 | Europe PMC | `simulation-based calibration Bayesian hierarchical model rank uniformity` | 179 | screened top 8; methods hits were mostly unrelated applied papers |
| 2 | PubMed | `spike-and-slab prior variance components hierarchical model` | 1 | one unrelated rare-variant paper; not used |
| 3 | Europe PMC | `C. elegans whole-brain calcium imaging optogenetic atlas reliability` | 53 | Randi 2023, Atanas 2023 identified |
| 4 | OpenAlex | `measurement error reliability hierarchical model posterior predictive check` | 9,486 | Williams 2021, hierarchical robustness work |
| 5 | OpenAlex | `Simulation-based calibration Talts Betancourt` | 452 | Modrak 2023 SBC sensitivity paper |
| 6 | OpenAlex | `spike and slab prior variable selection George McCulloch` | 1,130 | Ishwaran & Rao 2005 |
| 7 | PubMed | `zero-inflated hierarchical model false positive calibration` | 0 | no direct hit; no claim from absence |
| 8 | Europe PMC record | PMID 37914938 | 1 | Randi metadata and abstract verified |
| 9 | Europe PMC record | PMID 37607537 | 1 | Atanas metadata and abstract verified |
| 10 | Europe PMC record | PMID 41049725 | 1 | Modrak abstract and PMCID verified |
| 11 | Crossref | DOI `10.1038/s41586-023-06683-4` | 1 | Randi DOI verified |
| 12 | Crossref | DOI `10.3758/s13428-021-01646-x` | 1 | Williams DOI verified |
| 13 | PubMed status | PMIDs 37914938, 37607537 | 2 | no indexed notices; not proof of clean status elsewhere |

**Deduplication:** by DOI, PMID and title. The Randi paper appeared in both
Europe PMC and PubMed and was counted once. Preprints were not used as
decision evidence. The SBC paper is peer-reviewed in *Bayesian Analysis*; its
PMID/PMCID record was checked. Citation counts were recorded only incidentally
and were not used as quality ratings.

## Evidence table

| ID | Source | Status | Design / contribution | What it supports here | Limits |
|---|---|---|---|---|---|
| E14-01 | Williams, Martin & Rast (2021), *Behavior Research Methods*, [doi:10.3758/s13428-021-01646-x](https://doi.org/10.3758/s13428-021-01646-x) | Peer-reviewed; Crossref verified | Bayesian hierarchical model for within-person variance heterogeneity | Reliability depends on explicit variance assumptions; motivates the one-event variance audit | Human behavioural/psychometric context, not *C. elegans* |
| E14-02 | Modrák et al. (2023), *Bayesian Analysis*, [doi:10.1214/23-ba1404](https://doi.org/10.1214/23-ba1404), PMID 41049725 | Peer-reviewed; Europe PMC record verified | SBC theory and test-quantity sensitivity; rank-based checks | Equal-tailed coverage alone is insufficient; rank uniformity and test choice matter | Does not prescribe our exact gates or thresholds |
| E14-03 | Ishwaran & Rao (2005), *Annals of Statistics*, [doi:10.1214/009053604000001147](https://doi.org/10.1214/009053604000001147) | Peer-reviewed; OpenAlex metadata verified | Spike-and-slab variable selection, frequentist and Bayesian | Exact zero mass is a legitimate modelling device, not evidence it will calibrate | Variable selection, not reliability; no direct transfer |
| E14-04 | Randi et al. (2023), *Nature*, [doi:10.1038/s41586-023-06683-4](https://doi.org/10.1038/s41586-023-06683-4), PMID 37914938 | Peer-reviewed dataset; Crossref and Europe PMC verified; no indexed notice | 23,433 optogenetically probed neuron pairs with whole-brain calcium imaging | Source atlas and the repeated-stimulation structure analysed here | Derived event table, not raw recordings; stimulation and preparation specific |
| E14-05 | Atanas et al. (2023), *Cell*, [doi:10.1016/j.cell.2023.07.035](https://doi.org/10.1016/j.cell.2023.07.035), PMID 37607537 | Peer-reviewed dataset; Europe PMC verified; no indexed notice | Brain-wide behavioural representations in freely moving animals | Context for behavioural atlas work and why behaviour is a different estimand | Not a repeated-stimulation transport set; not used as validation |

## Synthesis

**Agreement.** The methods literature treats reliability as a variance
decomposition whose estimate depends on the assumed measurement-error structure
(E14-01), and treats SBC as a distributional check in which the choice of test
quantity matters (E14-02). Both points are consistent with the Phase 14
experience: changing the one-event variance treatment changed the reference by
0.0131, while coverage 0.967 coexisted with rank KS p = 3.35e-192.

**Disagreement / gap.** There is no direct peer-reviewed study of calibration for
this atlas's reliability estimand. The sparsity is extreme (most cells have one
event), and no source supplies an external repeated-stimulation calibration set.
The literature therefore constrains method choice but cannot adjudicate the
biological value of `R(22)`.

**What the scan does not show.** It does not show that spike-and-slab models are
generally invalid; E14-03 shows the opposite as a modelling device. It does not
show that SBC rank tests are the only valid calibration check; E14-02 argues the
test quantity should be chosen deliberately. The negative result is specific to
M4 on this atlas with the frozen priors and grid.

## Inference versus finding

- **Finding:** M4 fails 8/12 pre-registered gates on the local data.
- **Finding:** the Phase 11 `0.560` reference is robust to a robust one-event
  variance replacement (V1 delta 0.0131).
- **Inference:** a zero median can hide a heavy false-positive tail; full null
  distributions should be reported.
- **Inference:** a new model, not a retune of M4, is needed before any ranking or
  animal-budget claim.

## Limitations of this scan

- Rapid and non-systematic; no PRISMA counts, no dual screening, no exhaustive
  snowballing.
- PubMed returned few methods hits; OpenAlex recall is broad but noisy.
- Citation counts were not used as evidence quality.
- No full text was read for the statistical methods papers beyond indexed
  abstracts and metadata; claims are limited to their stated contributions.
- The atlas papers were read through abstracts and metadata; the derived table's
  provenance is documented separately in `docs/DATA_PROVENANCE.md`.

## Next tests

1. Pre-register a two-part/detection model with an explicit false-positive gate
   on the null q95, not just the median.
2. Obtain an independent repeated-stimulation dataset before any transport
   claim; status remains `not_evaluable`.
3. Pre-register ranking-stability and animal-budget gates before producing a
   target ranking.
4. Report the full null distribution and spike calibration for every candidate
   model.
