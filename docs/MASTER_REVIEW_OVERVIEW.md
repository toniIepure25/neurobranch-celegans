# Master Review Overview

**Audited branch:** `research/kato-final-reproducibility-audit`  
**Audited HEAD:** `96a8dd333643de0564247e2c01c98a85511dac82`  
**Base commit:** `358107a7a55470d48f8092b2a2dc7f704dbd37c1`  
**Audit date:** 2026-11-03  
**Reviewer stance:** independent computational-neuroscience, statistical, dynamical-systems and research-software audit.

---

## 1. What this review is, and is not

This is an independent forensic review of an existing research repository. It asks four different questions that are often conflated:

1. **Is the archive the object it claims to be?**  
   File identity, hash, dimensions, splits, landmark bounds, and metadata consistency.
2. **Does the committed code compute what the documents say it computes?**  
   Equations, causal windows, leakage, bootstrap construction, scoring rules.
3. **Does the numerical result support the scientific claim?**  
   Effect size, dependence, calibration, capacity, heterogeneity, multiplicity, and the history of test reuse.
4. **What may be communicated, to whom, and with what caveats?**  
   A claims ledger, a peer-review decision, an educational account, and a supervisor briefing.

It is **not** a new model-selection exercise. No architecture search was run. No hyperparameter was tuned on the test block. No raw data were modified. No Git history was rewritten. The prior audit package is preserved.

It is also **not a systematic review**. There was no PRISMA screening, no exhaustive citation chasing, and no meta-analysis. Literature is used for methodological orientation only. Calling this a systematic review would be false.

---

## 2. Source hierarchy

When sources disagreed, the review used this order:

| Rank | Source | Use |
|---:|---|---|
| 1 | Raw HDF5 archive `data/raw/kato2015/WT_NoStim.mat` | Counts, hashes, dimensions, state vectors, frame axes |
| 2 | OSF API metadata and readme | File identity, public size, hash, absence of animal keys |
| 3 | Peer-reviewed publication record (Kato et al. 2015, DOI `10.1016/j.cell.2015.09.034`, PMID `26478179`) | Cohort description, experimental context |
| 4 | Peer-reviewed secondary analysis (Grosse-Wentrup et al. 2024, DOI `10.1371/journal.pcbi.1012674`, PMID `39680605`) | Independent description of the same five-worm cohort |
| 5 | Committed code at HEAD | Actual equations and data flow |
| 6 | Frozen artifacts | Numerical results and their provenance |
| 7 | Project documentation | Claims to be checked, never treated as evidence by itself |

The publication’s full methods text could not be read directly because the publisher returned HTTP 403/CAPTCHA. The cohort statement is therefore taken from the published record and from two independent descriptions of the same dataset; it is not a direct quotation of the methods section. This is an explicit limitation.

---

## 3. The archive is real; the documentation is not fully consistent

The local file is:

- **size:** `56,390,975` bytes;
- **SHA256:** `7531811c1718e26a0455f59cb2f8188cf25809278cfea85770dfe03315a0098a`;
- **OSF object:** the same size and hash.

`docs/DATA_PROVENANCE.md` records `128,511,040` bytes. That number is wrong. The hash is right, so the scientific object is intact; the documentation is not.

The five recordings are:

| Session | Acquisition ID | Frames | Traces | Predictors after label exclusion | Named | Unnamed |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `TS20140715e` | 3,137 | 109 | 103 | 38 | 71 |
| 1 | `TS20140715f` | 3,134 | 135 | 127 | 44 | 91 |
| 2 | `TS20140905c` | 3,059 | 131 | 123 | 32 | 99 |
| 3 | `TS20140926d` | 3,311 | 125 | 118 | 47 | 78 |
| 4 | `TS20141221b` | 3,021 | 129 | 121 | 31 | 98 |
| **Total** | | **15,662** | **629** | **592** | **192 occurrences** | **437 occurrences** |

The archive has no explicit animal identifier. The publication reports a five-animal cohort; the five acquisition IDs are distinct; the two same-date recordings differ in suffix, egg count, frame count and trace panel. The defensible statement is:

> Five Kato `WT_NoStim` recordings from five distinct acquisition preparations, consistent with the publication’s five animals. The animal-level mapping is inferred, not keyed by the archive. They are one laboratory cohort, not an independent replication.

Both earlier extremes are wrong: “five repeated sessions of one preparation” is not supported, and “five independent animals” overstates independence. The correct phrase is **five animals (inferred), one cohort, not independent replication**.

---

## 4. The split and landmark computation is sound; three documents are wrong

The code computes per-recording split boundaries, not a shared boundary:

| Session | Train | Validation | Test |
|---:|---|---|---|
| 0 | `0:2195` | `2207:2677` | `2689:3137` |
| 1 | `0:2193` | `2205:2675` | `2687:3134` |
| 2 | `0:2141` | `2153:2611` | `2623:3059` |
| 3 | `0:2317` | `2329:2825` | `2837:3311` |
| 4 | `0:2114` | `2126:2579` | `2591:3021` |

The actual H100 landmark counts are `35, 35, 34, 38, 33`, not “35 per session.” Every H1–H100 target lies inside its recording and inside the test mask. There is no padding, interpolation or resampling. The frozen landmarks match the committed analysis exactly.

The H=5 case in session 1 reaches target index `3133` with `T=3134`: it is the last observed frame. That is valid but has zero margin. It is a boundary condition, not leakage.

Three documentation defects are confirmed:

- `docs/ADAPTIVE_DYNAMICS_MODELS.md` says “175 test landmarks (35 per session).” False.
- `docs/HANDOFF_METHODS_AND_LIMITATIONS.md` presents `2689:3137` as a common test boundary. It is session 0 only.
- `docs/DATA_PROVENANCE.md` gives the wrong byte size and uses field names (`IDs`, `timeVectorSeconds`) that do not match the HDF5 fields (`NeuronNames`, `tv`).

The code is not wrong here. The documents are.

---

## 5. The numerical result is reproducible; its inferential status is weaker than its presentation

The primary contrast is:

```text
B5-10 energy - B3 energy = +0.2644379926683563
```

Positive means B3 has lower H20 PCA-10 energy. The frozen block-5 interval is `[+0.0750, +0.4031]`; B3 is better at 64.0% of the 175 paired landmarks.

An independent reference implementation reproduced three landmarks with maximum absolute error `7.46e-7`. The arithmetic is not in doubt.

The inferential problems are:

1. **The five recordings are the independent units.** A recording-level bootstrap gives `[-0.041, +0.479]` at block 5, and `[-0.131, +0.440]` at block 20. A t interval across the five session means is `[-0.064, +0.587]`. The sign test is `p = 0.1875` one-sided. The animal-level result does **not** exclude zero.
2. **The 175 landmarks are not independent.** They are spaced 10 frames apart but forecast 20–100 frames; adjacent windows share future observations. The approximate number of non-overlapping H100 windows is 25, a roughly 7× inflation.
3. **The effect is small.** Descriptive paired `dz` is `0.27` for energy and `0.04` for MSE. The median advantage is `+0.186`, with an IQR from `-0.189` to `+0.804`.
4. **The effect reverses at H1.** B5 is better by `-0.0637` at one step. The result is medium/long-horizon only.
5. **Session 2 reverses.** Session means are `+0.300, +0.329, -0.193, +0.407, +0.465`. Four of five are positive; one is not.
6. **The mean forecast is not better.** PCA-10 MSE is `+0.0545` with an interval `[-0.157, +0.241]`; neuron-space MSE is `-0.0354` with an interval crossing zero.
7. **Calibration differs sharply.** B3’s 95% coverage is `0.938`; B5’s is `0.725`. B5 is severely undercovered. The energy score rewards a distribution that is both sharp and calibrated, so the energy advantage is partly a spread/calibration contrast.
8. **Capacity is not matched.** B3 has `7.96–9.95×` more parameters than B5-10 and arbitrary residual coordinates that B5 cannot represent. The contrast is a valid restricted-model comparison, not a capacity-matched causal ablation.
9. **The test block was reused.** Earlier phases inspected the same tails. The final audit is exploratory, not fresh confirmation.
10. **The sign rule was clarified in the same commit as the results.** G4 is a numerical pass under the corrected sign, not a fully prespecified confirmatory PASS.

The correct summary is:

> In these five Kato recordings, B3 has a modest, horizon-dependent H20 PCA-10 energy advantage over the ten-factor control. The effect is not established at the animal level, is partly confounded with ensemble spread/calibration and estimator capacity, reverses at H1, and is absent for mean MSE.

That is narrower than “population dynamics contain cross-neuron information,” and much narrower than “the model identifies interactions.”

---

## 6. What is verified, unresolved, and invalidated

### Verified

- File identity, SHA256 and OSF hash.
- Five recordings, frame counts, trace counts, predictor dimensions.
- Per-recording split boundaries and absence of split overlap.
- H1–H100 landmark containment; frozen landmark identity.
- B3/B5 arithmetic at three independent landmarks.
- The B3 advantage in the frozen sample, under the conditional block-5 interval.
- The H1 reversal, session-2 reversal, MSE null, and coverage difference.
- The off-diagonal instability (`r = 0.052`) and the failed structural gate.

### Unresolved

- Whether the five acquisition IDs map one-to-one to five publication animals. The archive has no key.
- Whether B3’s residual advantage is information or capacity.
- Whether the energy advantage survives a capacity-matched, calibration-matched, animal-level design.
- Whether an unmeasured common input, behaviour, nonlinearity or slow state explains the residual.
- Whether a different neuron alignment or representation would change the result.
- Whether the exact Kato methods used any additional preprocessing not represented in the archive.

### Invalidated

- “35 landmarks per session.”
- “Common test boundary 2689:3137.”
- “128,511,040 bytes.”
- “Five repeated sessions, not five animals” as a categorical statement.
- “G4 fully prespecified confirmatory PASS.”
- “51/49 is a unique partition.”
- “175 independent replicates.”
- “Test tails untouched.”
- “B3 improves general forecast accuracy.”
- “Probabilistic calibration established.”

---

## 7. Readiness verdict

The prior decision `READY_FOR_SUPERVISOR_REVIEW` is defensible only in an administrative sense: the code reruns, the claims are tiered, and the limitations are documented.

It is **not** ready for:

- a confirmatory mechanistic claim;
- a biological interpretation of coupling or command neurons;
- a publication abstract that says the population model “captures cross-neuron interactions”;
- an animal-level generalization claim;
- a digital-twin or intervention claim.

The appropriate verdict is:

```text
READY_FOR_METHODOLOGICAL_SUPERVISOR_REVIEW
NOT_READY_FOR_CONFIRMATORY_OR_BIOLOGICAL_CLAIMS
```

**For Denis Kovačević:** yes to a methodological discussion of the evidence and the next experiment; no to final scientific sign-off or a mechanistic claim.

The most useful supervisor question is not “is this significant?” It is:

> For an MSc, is an independent-cohort, animal-level, capacity-matched replication of the B5-controlled H20 energy contrast sufficient, or does the project require matched intervention data before the predictive result is scientifically useful?

That question asks for an evidential standard, not an endorsement of a mechanism the data cannot identify.

---

## 8. Reading map

| Document | Question it answers |
|---|---|
| [Data forensics](MASTER_REVIEW_DATA_FORENSICS.md) | What exactly is in the archive, and are the bounds real? |
| [Research history](MASTER_REVIEW_RESEARCH_HISTORY.md) | How did the project arrive at this claim, including dead ends? |
| [Code audit](MASTER_REVIEW_CODE_AUDIT.md) | What does the code compute, line by line? |
| [Statistics](MASTER_REVIEW_STATISTICS.md) | What can be inferred from five recordings and 175 overlapping windows? |
| [Biological interpretation](MASTER_REVIEW_BIOLOGICAL_INTERPRETATION.md) | What can and cannot be said about the worm? |
| [Literature comparison](MASTER_REVIEW_LITERATURE_COMPARISON.md) | How does this compare with Kato, BunDLe-Net and NC-MCM? |
| [Claims ledger](MASTER_REVIEW_CLAIMS_LEDGER.md) | Which sentences are allowed, limited, unsupported or invalid? |
| [Peer review](MASTER_REVIEW_PEER_REVIEW.md) | Would a skeptical reviewer accept this? |
| [Oral defense](MASTER_REVIEW_ORAL_DEFENSE.md) | Can the student defend every claim under questioning? |
| [Educational course](MASTER_REVIEW_EDUCATIONAL_COURSE.md) | What must the student learn to own this work? |
| [Final verdict](MASTER_REVIEW_FINAL_VERDICT.md) | What is the disposition, and what comes next? |

Machine-readable evidence is in [`reports/master_review/`](../reports/master_review/). The claims ledger is the authoritative list of permitted statuses; this overview is a narrative entry point, not a substitute for it.
