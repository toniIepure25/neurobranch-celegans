# Search log

Search date: 2026-10-29. This is a rapid scoping search, not a systematic
review.

| Index | Query | Hits | Screened | Used |
| --- | --- | ---: | ---: | ---: |
| Europe PMC | `split-half reliability neural response correlations` | 6,148 | 5 titles | 0 |
| OpenAlex | `attenuation correction reliability correlation ceiling` | 5,706 | 5 titles | 0 |
| PubMed | `simulation based calibration Bayesian computation` | 279 | 5 titles | 1 |
| PubMed | `generalizability theory measurement error reliability` | error | 0 | 0 |
| Europe PMC | DOI lookups for the seven evidence records | 0-1 each | 7 | 7 |
| Crossref | DOI verification for five records | 1 each | 5 | 5 |

The failed PubMed query raised `UnicodeEncodeError: 'charmap' codec ...` inside
the local MCP client on a non-cp1252 title. It was not an empty result. DOI
lookups were used to confirm metadata. Verified records are listed in
`research/evidence.csv`.

## Phase 11 coverage audit

No new literature search was performed for the coverage sensitivity analysis;
it is a re-analysis of the existing Randi atlas. The methods sources above are
the only external evidence used.

## Phase 12 hierarchical audit

No new literature search was performed for Phase 12. The analysis reuses the
methods records already verified for the reliability and calibration work:
Shrout & Fleiss (1979; `10.1037/0033-2909.86.2.420`) and Modrak et al. (2025;
`10.1214/23-ba1404`, PMID 41049725). The hierarchical model, SBC, held-out
predictive check and negative control are implemented from those methods
descriptions, not from a new systematic review. The calibration failure is an
internal result and must not be presented as a literature-supported claim.
