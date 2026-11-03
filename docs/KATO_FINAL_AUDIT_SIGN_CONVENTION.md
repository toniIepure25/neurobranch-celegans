# Kato final audit: sign convention

**Decision: `B_GENUINE_RULE_AMBIGUITY_NOT_FULLY_PRESPECIFIED`.**

The observed numbers are retained. This is an evidential-status decision, not a
change to any score, threshold, interval or gate arithmetic.

## 1. What the frozen record shows

The only committed common-state protocol was added in:

```text
358107a7a55470d48f8092b2a2dc7f704dbd37c1
2026-11-03 01:04:34 +02:00
feat(kato): add common-state predictive controls
```

The results, code and protocol were added in that same commit. There is no
earlier committed version of `docs/KATO_COMMON_STATE_PROTOCOL.md`:

```text
git log --follow -- docs/KATO_COMMON_STATE_PROTOCOL.md
358107a 2026-11-03 01:04:34 +0200 feat(kato): add common-state predictive controls
```

The committed protocol states, at lines 97–101:

> Energy is lower-is-better, so the positive advantage is defined as
> `B5 energy - B3 energy`; positive means B3 is better.
>
> This is a sign-convention clarification; the target quantity (whether B3 has
> lower energy than B5) is unchanged.

Its G4 row, line 159, uses `B5-10 - B3` with positive mean, positive interval lower
bound and fraction better `> 0.5`.

The prior working record reports that an earlier draft wrote `B3 - B5`. That
text is not recoverable from Git, has no independent timestamp, and cannot be
treated as a frozen pre-analysis document.

## 2. Why this is a genuine rule ambiguity, not merely a typo

The underlying physical target is clear: lower energy is better, and B3 must have
lower energy than B5. If the only question were “which model has lower energy?”,
the notation error would be harmless.

The problem is the decision rule and its provenance:

1. The written contrast in the earlier draft pointed in the opposite direction
   from the lower-is-better target.
2. The only committed version is the corrected version, and it shares a commit
   with the results; no pre-evaluation version is present.
3. No independently timestamped pre-evaluation protocol exists in Git.
4. The committed file itself calls the change a “sign-convention clarification”,
   which is evidence that the direction was not fixed unambiguously in the
   original record.

A reader cannot reconstruct, from the frozen repository alone, whether the
pre-evaluation success rule was “B3 lower” or “B3 higher”. That is a genuine
inconsistency in the prespecified decision rule, even though the numerical target
is physically obvious after correction.

## 3. Numerical result retained

| Quantity | Value |
| --- | ---: |
| B3 H20 PCA-10 energy | 2.7001497504411485 |
| B5-10 H20 PCA-10 energy | 2.9645877431095045 |
| Positive B3 advantage (`B5 - B3`) | 0.2644379926683560 |
| Block-5 95% interval | `[0.07499291957903519, 0.40311649156808815]` |
| Fraction of 175 landmarks with B3 lower | 0.64 |

These values are unchanged and reproduce the committed result.

## 4. Gate status

| Label | Status |
| --- | --- |
| Numerical G4 under the corrected committed rule | `PASS` |
| Fully prespecified confirmatory G4 | `NOT_ESTABLISHED` |
| Final evidence tier for the residual claim | `SUPPORTED_EXPLORATORY` |

This follows the audit rule: if the success direction was genuinely ambiguous
before evaluation, G4 must not be described as a fully prespecified confirmatory
pass. The numerical pass is real; the confirmatory label is not.

## 5. What should be said in the handoff

Allowed:

> Under the corrected orientation, B3 has a positive exploratory H20 energy
> advantage over B5-10 (`+0.264`, block-5 interval `[+0.075, +0.403]`), but the
> sign rule was clarified during analysis and was not frozen in an independent
> pre-evaluation commit. The result is therefore exploratory, not confirmatory.

Not allowed:

> G4 is a fully prespecified confirmatory PASS.

No historical file is rewritten. The corrected committed protocol remains as the
historical record; this audit adds the provenance caveat.
