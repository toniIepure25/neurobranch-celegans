| Gate | Status | Requirement |
|---|---|---|
| `G1_estimand_completeness` | **pass** | all three estimands with target counts and extrapolation flags |
| `G2_jackknife_integrity` | **pass** | >=95% valid LOO replicates, status ok |
| `G3_calibration` | **fail** | coverage 0.93--0.97 for each estimand |
| `G4_null_control` | **fail** | median <= 0.05 and q95 <= 0.15 for both controls |
| `G5_extrapolation` | **fail** | half-split MAE <= 0.10 and abs bias <= 0.05 |
| `G6_ranking_stability` | **fail** | Spearman lower CI > 0.50 and top-10 Jaccard > 0.60 |
| `G7_variance_robustness` | **pass** | abs(V1 - V0) <= 0.05 for each estimand |
| `G8_reproducibility` | **checked at release** | manifest, hashes, validator, clean worktree |
