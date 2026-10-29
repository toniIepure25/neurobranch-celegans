# Phase 10 research synthesis: reliability, ceilings and design

**Review type:** rapid scoping review, **not** a systematic review. It supports
an engineering decision (how to bound and design measurements) and does not
claim exhaustive coverage. Search date: 2026-10-29. No PRISMA screening counts
are reported because no systematic search was performed.

## Question

Which quantities should bound a model's correlation with held-out neural
responses, how should uncertainty in those bounds be expressed, and what design
variables limit a *C. elegans* perturbation atlas?

## Search record

| Index | Query | Hits | Screened | Used |
| --- | --- | ---: | ---: | --- |
| Europe PMC | `split-half reliability neural response correlations` | 6,148 | 5 titles | 0 |
| OpenAlex | `attenuation correction reliability correlation ceiling` | 5,706 | 5 titles | 0 |
| PubMed | `simulation based calibration Bayesian computation` | 279 | 5 titles | 1 (SBC methods) |
| PubMed | `generalizability theory measurement error reliability` | error | 0 | 0 |
| Europe PMC | DOI lookups for the seven records below | 0-1 each | 7 | 7 |
| Crossref | DOI verification for five records | 1 each | 5 | 5 |

The PubMed generalizability query failed inside the local MCP client with a
`UnicodeEncodeError` on a non-cp1252 title, not because the index had no
results. The failure is recorded rather than hidden. DOI lookups were used to
confirm metadata; title-level search results were not treated as evidence.

## Evidence table

| Source | Type | What it supports here | Identifier | Status |
| --- | --- | --- | --- | --- |
| Shrout & Fleiss 1979, *Psychological Bulletin* | Peer-reviewed methods paper | Intraclass correlations separate raters, targets and error; reliability is a variance decomposition | [10.1037/0033-2909.86.2.420](https://doi.org/10.1037/0033-2909.86.2.420) | DOI verified; no PMID indexed in Europe PMC |
| Button et al. 2013, *Nature Reviews Neuroscience* | Peer-reviewed review | Small samples make neuroscience estimates unstable and inflate apparent effects; design resolution must be reported | [10.1038/nrn3475](https://doi.org/10.1038/nrn3475), PMID 23571845 | PubMed record has an **Erratum** (2013;14(6):451); no retraction |
| Nosek et al. 2018, *PNAS* | Peer-reviewed perspective | Pre-declaring analyses separates confirmatory from exploratory claims | [10.1073/pnas.1708274114](https://doi.org/10.1073/pnas.1708274114), PMID 29531091 | Verified; no recorded notice checked here |
| Benjamini & Hochberg 1995, *JRSS-B* | Peer-reviewed methods paper | Multiplicity control for the Phase 8 inferential family | [10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) | DOI verified |
| Randi et al. 2023, *Nature* | Peer-reviewed dataset paper | Source of the 110-animal optogenetic atlas; responses vary across animals | [10.1038/s41586-023-06683-4](https://doi.org/10.1038/s41586-023-06683-4), PMID 37914938 | Verified; no recorded notice |
| Kato et al. 2015, *Cell* | Peer-reviewed dataset paper | Source of the five-worm calcium imaging data; population dynamics are behaviour-embedded | [10.1016/j.cell.2015.09.034](https://doi.org/10.1016/j.cell.2015.09.034), PMID 26478179 | Verified |
| Modrák et al. 2025, *Bayesian Analysis* | Peer-reviewed methods paper | Simulation-based calibration checks whether an inference procedure is calibrated; test quantities affect sensitivity | [10.1214/23-ba1404](https://doi.org/10.1214/23-ba1404), PMID 41049725 | Verified; no recorded notice |

Citation counts were not used as a quality signal. Full texts were not
systematically read; the table records metadata and the specific methodological
role of each source.

## Synthesis

**Agreement.** Reliability is a property of a measurement design, not of a
model. It decomposes into target signal, sampling noise and target x unit
interaction (Shrout & Fleiss). Correlations with a noisy measurement are
attenuated, so a model should be compared with a reliability-derived ceiling
rather than with 1. Small samples both widen intervals and bias estimates
(Button). Multiplicity and pre-specification determine which differences may be
interpreted (Benjamini & Hochberg; Nosek).

**Disagreement / gap.** The literature rarely states the **facet** at which a
ceiling is computed. Studies mix within-subject repeatability, between-subject
generalisation and pooled-mean reliability. This is exactly the error Phase 9
made: a valid single-animal ceiling was applied to a pooled metric. No retrieved
source provides a ready-made ceiling for an animal-pooled optogenetic atlas;
the Phase 10 formula is a standard variance decomposition applied to this
design, not a novel statistical result.

**Calibration.** SBC is the appropriate framework for checking whether an
interval procedure is calibrated (Modrák et al.). The Phase 10 coverage study is
a simple frequentist analogue; it failed, which means the animal-jackknife
interval should not be called a confidence interval.

## What this changed in NeuroBranch

1. Estimands are declared per claim (`single_animal`, `repeat`, `pooled_fold`,
   `superiority`, `descriptive`).
2. The pooled ceiling at 22 animals is reported next to pooled metrics;
   single-animal claims use the lower ceiling.
3. Phase 8 inferential claims carry unadjusted and Holm-adjusted p-values.
4. The design calculator reports an animal budget instead of an EIG portfolio.
5. Negative results (unstable target ranking, failed interval coverage) are
   reported rather than suppressed.

## Confidence and limitations

- **High confidence** in the arithmetic: the pooled reliability formula and its
  inversion are algebra, checked by synthetic recovery tests.
- **Moderate confidence** in the fitted components: they are moment estimates
  from 110 animals with unbalanced target coverage.
- **Low confidence** in per-target rankings and in the jackknife interval; both
  failed stability or coverage checks.
- The review is non-exhaustive and did not search IEEE, ACM, bioRxiv or Google
  Scholar. It should not be cited as a systematic review.

## Testable next steps

1. Two-way animal x responder resampling with nominal coverage on synthetic
   atlases.
2. Prospective test of the animal-budget calculator on a new cohort.
3. Shrinkage or hierarchical reliability to stabilise target rankings.
4. Held-out validation of the extrapolated region of the ceiling curve.
