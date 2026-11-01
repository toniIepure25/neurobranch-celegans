# dLDS reference and independent implementation

## 1. Reference

- Yezerets, Mudrik, Charles, and colleagues. “Decomposing dynamics in neural populations with dynamic latent linear dynamical systems.” *Communications Biology* (2025).
- DOI: [10.1038/s42003-025-08599-3](https://doi.org/10.1038/s42003-025-08599-3)
- PMID: `40804171`; PMCID: `PMC12350842`
- Code archive: [10.5281/zenodo.15866270](https://doi.org/10.5281/zenodo.15866270)
- Outputs: <https://osf.io/35hy2/>; data: <https://osf.io/2395t/>

The published model writes observations through a dictionary and evolves a latent state with a time-varying convex combination of operators:

```
y_t = D x_t
x_{t+1} = (Σ_m c_{m,t} F_m) x_t
```

with coefficients on the simplex and operators intended to be stable. The official Python repository inspected at commit `dc4889fa37bda8a0d851d93b09379ccc97e071a1` contains **no LICENSE file**. Its code was therefore not copied or vendored. The MATLAB development repository inspected at commit `3822905ad13848f61c3f561dc66d97a8ff06711d` contains external CVX files with their own licenses; those were not vendored either.

This branch contains an **independent implementation of the mathematical form**, written from the paper and public description. It is not a reproduction of the authors' software, numerical results, preprocessing, or biological conclusions.

## 2. What this implementation does

`CompositionalLatentLDS` in `src/neurobranch/dynamics/context_models.py`:

- fits a train-only PCA observation map `D` (latent dimension selected by Stage B, 15);
- initializes `M` stable operators `F_m`;
- alternates between:
  - non-negative least squares for simplex coefficients on each training transition;
  - ridge regression for each operator given the current coefficients;
- enforces an exact spectral cap on every operator;
- fits a **causal linear coefficient policy** from the current latent state and current known input only;
- rolls out by evaluating the policy at each step; it never receives future coefficients or future observations.

The implementation is deliberately small (M ∈ {1,2,3}) and is not an architecture search.

## 3. Differences from the published work

| Aspect | Published reference | This branch |
|---|---|---|
| Code | authors' Python/MATLAB repositories | independent Python implementation; no copied code |
| Observation model | dictionary learning / dLDS fitting | train-only PCA, fixed dictionary |
| Coefficient inference | published optimization | alternating NNLS + causal linear policy |
| Forecast | one-step reconstruction is the primary published use | explicit autonomous rollout and known-schedule rollout, reported separately |
| Stability | operator stability intended | exact spectral cap verified per operator |
| Biological interpretation | authors' analyses | none: operators are model-internal, not synapses |
| Datasets | authors' benchmarks | Kato 2015 sessions, with the limitations in the protocol |

## 4. Results

Validation selected **2 operators** (median H20 skill −0.577) over 1 (−0.594) and 3 (−0.646). All operators satisfied the exact 0.995 cap.

Held-out test:

| Horizon | Mean skill | Median skill |
|---|---:|---:|
| 20 | −1.771 | −1.087 |
| 100 | −0.808 | −0.343 |

All five sessions were worse than the selected latent LDS at H20. Coefficient mean entropy was 0.25–0.28 nats against a maximum of 0.693, and the causal policy RMSE was 0.375–0.393. This is evidence of **operator/coefficient-policy collapse**, not evidence of a compositional neural mechanism. No synaptic claim is made.

See `operator_validation.csv`, `operator_test.csv`, `operator_coefficient_trajectories.csv`, and `operator_summary.json`.

## 5. Reproduction boundary

- The official Python code was used only to inspect the model form and repository metadata.
- No official code or license file was copied.
- The official outputs were not claimed to be reproduced.
- The paper's DOI, PMID, and PMCID should be verified against PubMed/Crossref before external citation; this document records the identifiers used during the audit, not a claim of independent bibliographic verification in this branch.
