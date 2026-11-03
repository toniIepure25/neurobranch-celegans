# Kato final audit: results and required answers

**Scope:** five Kato 2015 `WT_NoStim` recordings; exploratory reanalysis. The
test tails have been inspected in previous phases. This is not a systematic
review and not a confirmatory replication.

The clean-rerun status and exact comparison are in
[KATO_FINAL_AUDIT_REPRODUCTION.md](KATO_FINAL_AUDIT_REPRODUCTION.md). The claims
ledger is `reports/simulator/kato_final_audit/claims_ledger.json`.

## Required answers

### 1. Did the original main experiment reproduce from start to finish?

The historical committed artifacts were **not** produced by an uninterrupted main
run: the prior phase disclosed a late-stage failure followed by a resume script.
That historical run therefore did not complete end-to-end. A separate clean run
from the committed entry point was executed for this audit and completed normally
after 2,005.9 s with empty stderr. Its status is
`FULL_END_TO_END_REPRODUCTION_PASSED`. The distinction matters: post-processing a
saved table is not an end-to-end reproduction, and the clean rerun is the
reproduction evidence.

### 2. Did the independently regenerated scores match the committed results?

Yes, far inside the declared `1e-8` deterministic tolerance. Across 17 compared
artifacts the maximum absolute difference was `4.44e-16` and the maximum relative
difference `9.84e-16`; no value exceeded `1e-8`. Individual model predictions
(2,275 rows × 29 numeric columns), B3/B5 energies, paired differences, bootstrap
intervals, win rates, per-session estimates, horizon profiles, 4,200 surrogate
draws and all synthetic rows matched. All eight gate statuses matched. See
[KATO_FINAL_AUDIT_REPRODUCTION.md](KATO_FINAL_AUDIT_REPRODUCTION.md).

### 3. Was the sign convention a harmless notation issue or a genuine protocol ambiguity?

**Genuine ambiguity affecting the prespecified rule.** The lower-is-better target
was physically clear, but the only committed protocol was added with the results,
contains an explicit “sign-convention clarification”, and has no independent
pre-evaluation commit. The numerical result is retained; the confirmatory label is
not. See [KATO_FINAL_AUDIT_SIGN_CONVENTION.md](KATO_FINAL_AUDIT_SIGN_CONVENTION.md).

### 4. Is the G4 PASS claim still defensible under the original frozen criterion?

Only as **`PASS_UNDER_CORRECTED_SIGN`**, not as a fully prespecified confirmatory
PASS. The three numerical criteria are met: mean `+0.2644 > 0`, block-5 interval
lower bound `+0.0750 > 0`, and fraction better `0.64 > 0.5`. The evidential tier is
`SUPPORTED_EXPLORATORY`.

### 5. Is the 51% / 49% decomposition technically correct, and what does it mean?

The arithmetic is exact for the ordered sequence B4m→B5-10→B3:

```text
common    B4m - B5 = +0.275079  (51.0%)
residual  B5 - B3  = +0.264438  (49.0%)
total     B4m - B3 = +0.539517
```

It is a descriptive sequential decomposition. With B1m as baseline the shares
become 23.5%/76.5%; with B0 they become 72.4%/27.6%. The split is not unique, not
a variance-component estimate, and not a biological partition. See
[KATO_FINAL_AUDIT_COMPARABILITY.md](KATO_FINAL_AUDIT_COMPARABILITY.md).

### 6. Were B3 and B5 compared with fair estimation assumptions?

They were matched on recordings, 175 H20 landmarks, targets, J5 innovations,
500-frame causal window, ridge `1.0`, unpenalized intercept, initialization and
the PCA-10 metric. They were **not** matched on capacity or information:
B3 has 7.96–9.95× more parameters and arbitrary other-neuron residual
coordinates; B5 is a restricted diagonal-plus-ten-factor model. The comparison is
a valid restricted-model contrast, not a capacity-matched causal ablation.

### 7. Does the ten-factor control discard historically relevant predictive information?

The evaluation PCA and common-factor PCA are the **same training-only transform**;
B5 therefore retains all ten evaluation dimensions. It is not discarding the
evaluation space by compression. What it cannot represent is arbitrary
other-neuron residual coordinates: its residual-source operator is restricted to
`diag(own_scale) @ Q`, one own-row direction per output neuron. B3's residual
rows are 98.6–99.9% outside that one-dimensional pattern (median 99.6%).

This proves B3 has access to residual cross-neuron structure that B5 lacks. It
does **not** prove that this structure causes the score advantage, because B3 also
has far more parameters. Distinguishing information from capacity would require a
prespecified restricted model or new data; no new model was selected on the reused
test tails.

### 8. Does the population advantage survive defensible bootstrap block lengths?

Yes in point estimate and interval sign for blocks 3, 5, 10 and 20:

| Block | Mean `B5−B3` | 95% interval |
| ---: | ---: | --- |
| 3 | +0.2644 | `[+0.0972, +0.4018]` |
| 5 | +0.2644 | `[+0.0750, +0.4031]` |
| 10 | +0.2644 | `[+0.0350, +0.3994]` |
| 20 | +0.2644 | `[+0.0921, +0.2939]` |

All remain positive, so the result is not an artifact of choosing block 5.
Block 20 has only two blocks per session and is not evidence of greater precision.
The intervals are repeated-session/overlapping-landmark intervals, not animal-level
intervals. See [KATO_FINAL_AUDIT_UNCERTAINTY.md](KATO_FINAL_AUDIT_UNCERTAINTY.md).

### 9. How heterogeneous are the effects across sessions?

Four of five session means are positive:

| Session | Mean `B5−B3` | Fraction B3 better |
| ---: | ---: | ---: |
| 0 | +0.3003 | 0.686 |
| 1 | +0.3292 | 0.686 |
| 2 | **−0.1931** | 0.500 |
| 3 | +0.4075 | 0.579 |
| 4 | +0.4645 | 0.758 |

Session 2 reverses. Within-session block-5 intervals include zero in sessions
1, 2 and 3; only sessions 0 and 4 exclude zero. Leave-one-session-out pooled
means remain positive (`+0.218` to `+0.375`), so no single session drives the sign.
The sessions are repeated recordings, not independent animals.

### 10. Which results are robust across prediction horizons?

H1 reverses: B5 is better by `−0.0637` (`[−0.0967, −0.0311]`). B3 is better at
H5 (`+0.0835`), H10 (`+0.1471`) and H20 (`+0.2644`), with intervals excluding
zero. The finding is a medium/long-horizon energy result, not a general
short-horizon forecasting improvement. MSE and neuron-space results are mixed.

### 11. Which findings are specific to the ten-dimensional PCA representation?

The positive interval is specific to PCA-10 **energy**:
`+0.2644` with interval excluding zero. Neuron-space energy is positive
(`+0.1617`) but its interval includes zero. PCA-10 MSE is `+0.0545` and neuron
MSE is `−0.0354`, both intervals including zero. The result should be called a
fixed-PCA energy result, not a general forecast improvement.

### 12. What exactly can be concluded about cross-neuron predictive information?

A modest, horizon-dependent, repeated-session **conditional predictive
association** survives the specified own-history and ten-factor controls under
the corrected energy orientation. It is compatible with residual cross-neuron
information, unmeasured common input, nonlinearity, slow state or estimator
capacity. The current design cannot distinguish these.

### 13. What cannot be concluded about biological neuron-neuron coupling?

No stable pairwise interaction graph, synapse, gap junction, command neuron,
causal driver, functional-connectivity network or stimulation target. The
off-diagonal operator is unstable across chronological segments (coefficient
correlation `0.052`), and observational prediction cannot identify direct
coupling. G7 is `FAIL`; G8 is `NOT_EVALUABLE`.

### 14. Why are mathematical perturbations still non-causal and unstable?

The perturbation is a `+1 SD` displacement inside a fitted linear operator, not a
calibrated biological dose. Cross-fit response directions have median cosine
`0.556`, with sign reversals across permitted fits. No matched sham-controlled
intervention data exist. The correct label is:

```text
MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.
```

### 15. Which historical scientific claims should remain in the final handoff?

- Kato WT_NoStim population history gives exploratory H20 predictive improvement.
- B3 beats matched own-history baselines on energy, with mixed MSE/neuron results.
- Common factors explain a substantial ordered increment, but not a unique share.
- Negative findings remain: unstable off-diagonal operators; Markov-null failure;
  calibration/covariance failure; unstable virtual perturbations.
- Biological intervention validity is `NOT_EVALUABLE`.

### 16. Which claims need to be downgraded or corrected?

- “G4 is a fully prespecified confirmatory PASS” → numerical pass under clarified
  sign, confirmatory status `NOT_ESTABLISHED`.
- “51% common / 49% residual” as a unique partition → ordered descriptive split.
- “Five animals” → five repeated recording sessions with undocumented independence.
- “Residual advantage isolates neuron-specific interactions” → unresolved
  information-versus-capacity ambiguity.
- Any “validated digital twin”, “connectome”, “causal network” or “stimulation
  target” claim → unsupported.

### 17. Is NeuroBranch ready for supervisor review?

See `reports/simulator/kato_final_audit/decision_record.json`. The decision is
**`READY_FOR_SUPERVISOR_REVIEW`**: the clean end-to-end rerun passed, the claims
are correctly tiered, and the scientific limitations are disclosed. “Ready” means
ready for supervisory judgement of an exploratory result, not ready for
publication as a confirmed mechanism.

### 18. What is the most scientifically useful question for Denis Kovačević?

> For an MSc, is an independent-cohort, animal-level replication of the
> B5-controlled H20 energy contrast sufficient, or does the project need matched
> intervention data before the predictive result is scientifically useful?

This asks him to choose the evidential standard and scope, rather than asking him
to endorse a mechanism the data cannot identify.

## Standing labels

```text
PROBABILISTIC CALIBRATION NOT ESTABLISHED
MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.
Biological intervention validation: NOT_EVALUABLE.
```

## Status

- Starting HEAD: `358107a7a55470d48f8092b2a2dc7f704dbd37c1`.
- Branch: `research/kato-final-reproducibility-audit`.
- Clean rerun: see reproduction document.
- No raw data or large model checkpoints are committed.
