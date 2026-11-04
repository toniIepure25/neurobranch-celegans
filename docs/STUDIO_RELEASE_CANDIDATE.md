# Scientific Studio release candidate

**Branch:** `research/neurobranch-release-candidate`  
**Starting HEAD:** `1989b5a2cc5d7739d6e43604255ad1e5dc5203b1`  
**Scope:** provenance repair, release verification, browser/presentation QA, and
bounded documentation. No new model, dataset, hyperparameter search, or
scientific claim.

## 1. Release position

The application remains a read-only presentation layer over frozen Kato 2015
`WT_NoStim` artifacts. The numerical result is unchanged:

- primary B5−B3 H20 PCA-10 energy contrast: `+0.2644379926683563`;
- conditional block-5 interval: `[+0.0749929, +0.4031165]`;
- recording-resampling interval: `[-0.0408009, +0.4792966]` — includes zero;
- 4/5 sessions positive; sign-test p `0.1875`;
- H1 reverses; MSE improvement is not established;
- B3/B5 capacity is not matched;
- no causal, biological, digital-twin, or animal-level claim.

## 2. What changed in this pass

1. Added `scripts/studio_provenance_audit.py`, a read-only, version-aware
   historical snapshot validator. It validates each frozen manifest entry
   against its recorded source commit and classifies line-ending and
   pre-commit worktree differences instead of comparing to today's worktree.
2. Added `scripts/studio_release_checks.py`, which verifies the corrected PCA
   denominator, primary contrast, interval boundaries, H1 reversal, capacity,
   numerical root-cause control, and 3D alignment from frozen artifacts.
3. Hardened `scripts/studio_browser_qa.py` to wait for the presentation slide
   marker and to use a robust `Next` locator.
4. **Fixed a real presentation crash found by browser interaction.** Slide 2
   called `snapshot.sessions.values()` even though `StudioSnapshot.sessions` is a
   tuple. The route rendered an `AttributeError` and lost its navigation
   controls. It now iterates the tuple directly. A regression test renders all
   ten slides through `AppTest`.
5. Added `docs/STUDIO_HISTORICAL_PROVENANCE_VALIDATION.md`,
   `docs/STUDIO_HUMAN_VISUAL_REVIEW_CHECKLIST.md`, and
   `docs/STUDIO_FINAL_DEMO_SCRIPT.md`.
6. No visual restyling was performed without image inspection. The seven
   sections and presentation architecture are unchanged.

## 3. Provenance outcome

| Item | Result |
| --- | --- |
| Final-audit manifest, recorded head `2b42ec3` | 71 exact, 5 line-ending representations, 7 at `96a8dd3`, 2 unverifiable audit-script worktree snapshots |
| Interaction manifest, recorded head `4896ab1` | 34 exact; `app/handoff_demo.py` and `mkdocs.yml` match pre-commit `358107a` |
| Unexplained scientific-artifact modification | **none** |
| Current Studio manifest | **124/124 entries pass** |
| Historical provenance | **INCOMPLETE** — two historical audit-script blobs are not recoverable from Git |

The old validators still fail by design because they compare historical
manifests to the current worktree. They are historical-only checks; the new
command is the contemporary integrity gate.

## 4. Current integrity command

```powershell
python -X utf8 scripts/studio_provenance_audit.py --pretty
```

Exit code is `1` while the two unverifiable historical script snapshots remain.
The command does not rewrite a manifest and does not hide the two entries.

```powershell
python -X utf8 scripts/studio_release_checks.py
```

This must exit `0` and report `status=PASS`.

## 5. Browser QA position

The automated run visits all seven sections at `1920×1080`, `1366×768`, and
`1280×800`, presentation slides 01, 02, and 06 at each viewport, and one
light-mode 3D view. It records console errors and horizontal overflow and
writes viewport captures plus `browser_qa.json`.

**The agent cannot perform genuine image-based visual inspection.** Therefore:

- `BROWSER_AUTOMATED_QA` may be PASS if the capture run is clean;
- `HUMAN_VISUAL_SIGNOFF` remains **PENDING**.

Contact sheets and the human checklist are the handoff for the visual reviewer.

## 6. Release decision

| Status | Value | Basis |
| --- | --- | --- |
| `NUMERICAL_VALIDATION` | **PASS** | 11 extended cases; H100/H20 negative control |
| `SCIENTIFIC_CLAIMS` | **PASS** | bounded wording and intervals preserved |
| `HISTORICAL_PROVENANCE` | **INCOMPLETE** | two audit-script snapshots lack Git bytes |
| `CURRENT_ARTIFACT_INTEGRITY` | **PASS** | current manifest entries match |
| `3D_VISUALIZATION_INTEGRITY` | **PASS** | train-only PCA; denominators and semantics checked |
| `BROWSER_AUTOMATED_QA` | **PASS** | 31 captures, 0 console errors, 0 overflow |
| `HUMAN_VISUAL_SIGNOFF` | **PENDING** | agent cannot inspect images |
| `SUPERVISOR_DEMO_READINESS` | **READY_WITH_CAVEATS** | demo viable, caveats explicit |

The release candidate is scientifically and numerically bounded. It is not a
visual PASS and it is not a fully Git-verifiable historical package. The two
unverifiable historical scripts are audit utilities, not model or result data.
No scientific artifact is unexplained.

The presentation crash on slide 2 is fixed; all ten slides now render in the
AppTest matrix, and slides 01, 02, and 06 were exercised in Chromium at all
three viewports.
