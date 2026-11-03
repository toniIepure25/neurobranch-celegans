# Scientific evidence and claims

**Current verdict:** `READY_FOR_METHODOLOGICAL_SUPERVISOR_REVIEW; NOT_READY_FOR_CONFIRMATORY_OR_BIOLOGICAL_CLAIMS`.

The Studio's job is to present this bounded result without promoting it. This
document is the human-readable evidence summary; the machine-readable sources
remain authoritative.

## 1. Dataset

Kato et al. (2015), *Cell* 163:656–669, DOI
[10.1016/j.cell.2015.09.034](https://doi.org/10.1016/j.cell.2015.09.034).

| Property | Value |
| --- | --- |
| File | `data/raw/kato2015/WT_NoStim.mat` |
| Actual size | 56,390,975 bytes |
| SHA256 | `7531811c1718e26a0455f59cb2f8188cf25809278cfea85770dfe03315a0098a` |
| OSF | https://osf.io/2395t/ |
| Recordings | 5 |
| Frames | 15,662 |
| Predictors | 592 (103, 127, 123, 118, 121) |
| Named-trace occurrences | 192 |
| Sampling | ~2.80–3.07 Hz, one 1,080 s acquisition each |

The five entries are five distinct acquisition preparations and are consistent
with the publication's five-animal cohort. The OSF archive exposes **no animal
key**. The one-to-one animal mapping is inferred, not a documented primary key.
They are one laboratory/preparation, not an independent cohort.

Historical documentation said 128,511,040 bytes and used field names `IDs` and
`timeVectorSeconds`. Those are corrections X01 and X14. The actual HDF5 fields
include `NeuronNames` and `tv`. `stateParams` rows do not align with all trace
rows and are unused by the model (X13).

## 2. Question and endpoint

**Question.** Does a full population lag-one VAR (B3) retain predictive
structure beyond a ten-factor common-state model with each neuron's own lag
(B5-10)?

**Primary endpoint.** H20 PCA-10 energy score, paired at 175 frozen test
landmarks. Energy is a proper score for a probabilistic forecast. Positive
contrast `B5 − B3` means B3 has lower energy.

**Models.**

- B0 persistence.
- B1m independent AR(1), 200-frame window, matched to the historical baseline.
- B3 full population VAR, 500-frame ridge, spectral cap 0.995.
- B5-10 own AR plus ten training-only PCA factors, 500-frame ridge.
- B4m is an independent AR with the 500-frame window; it is a capacity
  comparator, not a second population model.

All are lag-one models on standardized calcium ΔF/F. The coefficient matrix is
an effective imaging-timescale transition, not a synaptic weight.

## 3. Primary result

| Quantity | Value |
| --- | ---: |
| Paired landmarks | 175 |
| B3 mean H20 energy | 2.7001497504 |
| B5 mean H20 energy | 2.9645877431 |
| Weighted paired mean difference | **+0.2644379927** |
| Relative to B3 mean | 9.79% |
| Equal-session mean | +0.2616577452 |
| Positive recordings | 4 / 5 |
| Positive landmarks | 64.0% |
| Paired dz (descriptive) | 0.2704639172 |
| Median paired difference | +0.1860157735 |

Recording means: `+0.3003, +0.3292, −0.1931, +0.4075, +0.4645`.

### Intervals (do not collapse them)

| Object | Interval |
| --- | --- |
| Conditional landmark block-5 | `[+0.0750, +0.4031]` |
| Recording resampling block-5 | `[−0.0408, +0.4793]` |
| Recording resampling block-20 | `[−0.1307, +0.4403]` |
| Recording resampling block-100 | `[+0.0257, +0.4189]` |
| Five-recording t interval | `[−0.0641, +0.5874]` |
| Sign test, one-sided | p = 0.1875 |

The conditional interval describes landmarks within these recordings. The
recording bootstrap is the appropriate check for extending beyond these five
units; it includes zero at block 5 and block 20. With five units, power is very
low.

**Strongest defensible statement:** *In these five recordings, B3 had lower H20
PCA-10 energy than B5-10 in four of five sessions (equal-session mean +0.26);
the recording-level t interval includes zero and the sign test is not
significant. The result is exploratory and horizon-dependent.*

## 4. Horizon and proper scores

| Horizon | Mean Δ (B5 − B3) | Conditional interval |
| ---: | ---: | --- |
| H1 | **−0.0637** | `[−0.0967, −0.0311]` |
| H5 | +0.0835 | `[+0.0339, +0.1145]` |
| H10 | +0.1471 | `[+0.0629, +0.2037]` |
| H20 | +0.2644 | `[+0.0750, +0.4031]` |

H1 reverses: B3 is worse than B5 at one step. Any claim of a general advantage
is false. The four horizons are descriptive; they were not treated as four
independent confirmatory tests.

**MSE.** Mean difference `+0.0545`, block-5 interval
`[−0.1574, +0.2409]`: the interval crosses zero. B3 mean MSE 1.7067, B5 1.7612
(B5 about 3.1% worse). Energy rewards calibration and sharpness as well as the
conditional mean, so the energy result does not establish a mean advantage.

## 5. Calibration

| Nominal | B3 marginal | B5 marginal |
| ---: | ---: | ---: |
| 50% | 0.5463 | 0.3491 |
| 80% | 0.8297 | 0.5629 |
| 95% | 0.9383 | 0.7246 |

These are averages of **marginal per-dimension** coverage. They are not the
probability that a whole 10-D vector lies in a joint region. B5 is severely
undercovered; B3 is close to nominal on this marginal statistic but is not
thereby "calibrated". Calibration is **not established** (X10, X17).

## 6. Capacity and identifiability

B3 has `n(n+1)` nominal coefficients; B5-10 has `12n + 110`. The ratio is
**7.96–9.95×** across recordings. B5 captures 54.7–71.0% of training variance
and leaves up to 45.3% residual variance. Ridge shrinkage changes effective
capacity but does not make the models capacity-matched.

The ordered decomposition from B4m is 51.0% common + 49.0% residual. Against
B1m it is 23.5/76.5; against B0 it is 72.4/27.6. The split depends on the
ordering and the baseline. It is arithmetic, not a structural or biological
partition (X06, X12).

B3 can represent arbitrary residual cross-neuron coordinates that B5 cannot.
The residual advantage is therefore **not identifiable as cross-neuron
information** from this design. No VAR coefficient is a directed connection.

## 7. Inference unit

The 175 landmarks are overlapping windows from five recordings. They are not
175 animals. Approximately 25 non-overlapping H100 proxies exist; the landmark
inflation factor is about 7×. Four of five recording means are positive, but
five units cannot support an animal-level generalization claim (X08, X15).

## 8. Protocol boundary

- The test tail was reused across earlier project phases; this is not a fresh
  confirmatory test (X09).
- B3/B5 are refitted at every landmark on the trailing 500 observed frames,
  including validation and earlier test frames. There is no future leakage,
  but the estimand is an **online-refit** one (X16).
- G4's numerical pass was not a confirmatory PASS (X05).

## 9. Claim ledger

23 claims: **9 INVALIDATED, 6 UNSUPPORTED, 3 LIMITED, 3
SUPPORTED_EXPLORATORY, 2 NOT_EVALUABLE**. There is no `CONFIRMED` tier.

Examples of invalidated/unsupported wording:

- "B3 generally forecasts better." — invalidated by H1 and the null MSE.
- "175 independent landmarks." — invalidated by overlap and effective units.
- "The 51/49 split is a unique structural partition." — invalidated by the
  ordered decomposition.
- "The residual advantage isolates cross-neuron information." — unsupported by
  capacity.
- "The ensemble is calibrated." — unsupported; B5 marginal coverage is 0.725.
- "The test tail was untouched." — invalidated by reuse.
- "The model is a validated digital twin." — unsupported.
- "A virtual perturbation is a real stimulation." — invalid.

## 10. Virtual perturbation boundary

The perturbation lab edits a standardized ΔF/F coordinate in a fitted B3 model
and rolls forward under zero innovation. It displays the exact banner
**MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.** It is not
optogenetics, electrophysiology, chemogenetics, behaviour, or a therapeutic
effect. It does not rank targets. The B3/B5 disagreement quantifies model
dependence; it does not validate the perturbation. An edit outside the training
marginal is flagged as OOD.

The input matrix is already train-standardized per recording (training mean 0,
training SD 1). Adding one unit to one native coordinate is therefore one
training z unit at the origin; the model refits its own within-window
standardization internally. No cross-session cell registration exists, so a
label identifies a cell-type label in one recording, not the same identified
cell across recordings.

## 11. What would change the conclusion

1. Independent animals with documented IDs and a pre-registered endpoint.
2. A capacity-matched residual model or an explicit measured common input.
3. A fresh test tail that has not been reused.
4. Joint calibration, not marginal coverage.
5. A prospective, sham-controlled perturbation with held-out targets/animals.

Falsification for the next study: the animal-level B3-minus-B1 energy interval
includes zero, or the sign reverses in most animals.

## 12. Source map

| Claim | Source |
| --- | --- |
| archive hash/size | `reports/master_review/dataset_identity_audit.json` |
| paired scores | `reports/simulator/kato_common_state/model_per_landmark.csv` |
| primary statistics | `reports/master_review/statistical_audit.json` |
| intervals | `reports/master_review/cluster_bootstrap.csv`, `.../kato_final_audit/uncertainty_summary.json` |
| claims wording | `reports/master_review/claims_ledger.json` |
| corrections | `reports/master_review/corrections.json` |
| capacity | `reports/master_review/capacity_audit.json` |
| quality gate | `reports/master_review/quality_gate.json` |
