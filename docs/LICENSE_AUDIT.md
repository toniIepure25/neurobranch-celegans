# Licence and data-availability audit

This is a factual inventory, not legal advice. It records what is in the
checkout and what may be redistributed; it does not resolve upstream terms.

## Repository code

| Item | Status |
|---|---|
| NeuroBranch source | MIT (`LICENSE`) |
| Packaged schemas | MIT, same repository |
| Figures and tables generated from public data | MIT for the code; upstream data terms still apply |

## Kato et al. 2015

| Field | Value |
|---|---|
| Source | OSF project `2395t`, `WT_NoStim.mat` |
| Licence metadata | OSF `node_license: null` when checked (2026-10-23) |
| Redistributed in Git | **No** (`data/raw/*` ignored) |
| Action before redistribution | Obtain explicit terms from the OSF project owner |

## Randi et al. 2023 optogenetic atlas

| Field | Value |
|---|---|
| Source | Public atlas used to build the derived trial table |
| Derived table | `reports/phase3/randi_trial_table_phase3.csv.gz` |
| Redistributed in Git | **No** (ignored; regenerate from the upstream atlas) |
| Licence status | **Unresolved in this checkout**; do not redistribute the table until confirmed |

The Phase 11 coverage analysis reads this derived table. A reader without the
table can reproduce the code path but not the exact numbers; the committed
aggregate artifacts and hashes are the audit trail.

## Third-party code

| Repository | Use | Licence status in checkout |
|---|---|---|
| BunDLe-Net | architecture reference | no `LICENSE` file found at repository root |
| NC-MCM | distributed states / Markov test | no `LICENSE` file found at repository root |
| `Celegans_premotor` | auxiliary reference | MIT (Copyright 2024 mmtree) |

No code from BunDLe-Net or NC-MCM is copied into `src/neurobranch`; only
independent reimplementations and artifact readers are included. Verify upstream
terms before redistributing those artifacts.

## Data-availability statement

The repository ships code, schemas and aggregate reports, not the raw Kato or
Randi data. To reproduce the analyses, obtain the data from the original
sources, verify the hashes in [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md), and
check the upstream terms. The Phase 11 input hash is recorded in
`reports/phase11/summary.json` and `reports/phase11/manifest.json`.
