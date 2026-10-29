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
