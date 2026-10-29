# Phase 11 evidence protocol

**Review type:** rapid scoping review, not a systematic review. It supports an
internal measurement-audit decision and does not claim exhaustive coverage.

## Question

Is the Phase 10 pooled reliability ceiling robust to sparse responder coverage,
and which methods papers justify the sensitivity and calibration checks?

## Inclusion

- Peer-reviewed methods papers on reliability, multiplicity, calibration or
  research design.
- Peer-reviewed dataset papers for the Kato and Randi atlases.
- Records with a resolvable DOI; PMID recorded when indexed.

## Exclusion

- Preprints without peer review.
- Title-only search hits that were not read.
- Blog posts, documentation pages and vendor material.

## Indexes and queries

Europe PMC, OpenAlex, PubMed and Crossref were used. Queries and hit counts are
recorded in [`docs/SEARCH_LOG.md`](../docs/SEARCH_LOG.md). DOI lookups were used
to verify metadata; a title-level hit was never treated as evidence.

## Verification

Each row in [`evidence.csv`](evidence.csv) records DOI, PMID where available,
verification date and the claim it supports. PubMed notices are checked where a
PMID exists; absence of a notice is not proof of a clean record.

## Limitations

The search is not exhaustive, screening was title-level for most queries, and
one PubMed query failed inside the local client with a Unicode encoding error.
That failure is recorded, not excluded silently.
