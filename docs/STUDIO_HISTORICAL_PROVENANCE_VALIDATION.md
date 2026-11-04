# Historical provenance validation

**Purpose:** explain, rather than suppress, the two frozen-manifest hash failures
reported by the old Kato validators. The old validators compare a historical
manifest to the current worktree. That is the wrong comparison after legitimate
later commits, so this release adds a read-only snapshot-aware audit.

## 1. Definitions

| Term | Meaning |
| --- | --- |
| `HISTORICAL_SNAPSHOT_INTEGRITY` | A manifest entry checked against its recorded source Git commit, permitting a documented newline representation and, when necessary, the commit where the recorded bytes actually occur. |
| `CURRENT_WORKTREE_INTEGRITY` | The current Studio manifest checked against current working-tree bytes. |
| `EXPECTED_VERSION_DIFFERENCE` | An authorized later code/docs edit, or a line-ending representation difference. |
| `UNVERIFIABLE_WORKTREE_SNAPSHOT` | The manifest records a path as dirty at generation, but its exact bytes are absent from every Git commit and from the object database. This is not an unexplained scientific-artifact change, but it is not a Git-verifiable snapshot. |
| `UNEXPLAINED_ARTIFACT_MODIFICATION` | No Git tree, line-ending variant, or recorded dirty path accounts for the recorded bytes. |

`HISTORICAL_SNAPSHOT_INTEGRITY` is not a claim that a historical manifest equals
today's files. It is a claim about the source revision and the recorded
generation state.

## 2. Final-audit manifest

Manifest: `reports/simulator/kato_final_audit/manifest.json`  
Recorded head: `2b42ec3f4eeaf8c09c1bed08ffcb0dec5ce5320a`  
Generated: `2026-11-03T17:32:19+02:00`  
Next commit: `96a8dd333643de0564247e2c01c98a85511dac82`, `2026-11-03T17:32:30+02:00`

The manifest recorded nine dirty/untracked paths at generation. The audit
resolves them as follows:

| Class | Count | Explanation |
| --- | ---: | --- |
| Exact Git blob at recorded head | 71 | Historical snapshot matches its source. |
| CRLF representation of source blob | 5 | Manifest was generated from a Windows worktree; Git stores LF. |
| Exact blob at next commit | 7 | The manifest was generated from a worktree immediately before `96a8dd3`; docs, figures, and `layout_report.json` landed in that commit. |
| Unverifiable worktree snapshot | 2 | Two audit scripts were dirty at generation and their exact bytes never entered Git. |
| Unexplained scientific-artifact modification | 0 | No scientific data artifact is unexplained. |

### `.gitattributes`

- Historical Git blob at `2b42ec3`: `1,562` bytes, SHA-256 `879a83e2…`.
- Manifest: `1,602` bytes, SHA-256 `e5608358…`.
- Converting the historical LF blob to CRLF gives exactly the manifest bytes and
  hash. This is **line-ending normalization**, not a content edit.

### The two unverifiable scripts

| Path | Manifest bytes | Manifest SHA-256 | Source blob at `2b42ec3` |
| --- | ---: | --- | ---: |
| `scripts/kato_final_audit_figures.py` | `8,457` | `aa8dc738…` | `6,050` bytes |
| `scripts/validate_kato_final_audit.py` | `10,048` | `0036fccf…` | `9,225` bytes |

Both paths appear in `git_status_porcelain_at_generation` as modified. Their
exact manifest bytes are not present in any commit or in `git cat-file`; the
later commit `96a8dd3` contains different versions. The manifest hash is the
only surviving byte record.

These are audit/figure utilities, not model code or scientific result files. No
scientific data artifact is affected. The correct status is therefore
**`HISTORICAL_PROVENANCE: INCOMPLETE`**, not PASS and not FAIL: the historical
hash cannot be independently matched to original source bytes for two code
snapshots.

## 3. Interaction manifest

Manifest: `reports/simulator/kato_interactions/manifest.json`  
Recorded head: `4896ab13fb939d4617fd80dfd3f2021b1ff27c90`  
Generated: `2026-11-02T23:01:03Z` — `2026-11-03T01:01:03+02:00`  
Expected two-file commit: `358107a7a55470d48f8092b2a2dc7f704dbd37c1`, `2026-11-03T01:04:34+02:00`

| Path | Manifest bytes | SHA-256 | Source at `4896ab1` | Matching commit |
| --- | ---: | --- | ---: | --- |
| `app/handoff_demo.py` | `43,203` | `f852e719…` | `39,274` bytes | `358107a` |
| `mkdocs.yml` | `8,771` | `5c5d5da0…` | `8,419` bytes | `358107a` |

The manifest was generated from a worktree about three minutes before
`358107a`; those two files were being prepared for that commit. The other 34
entries match `4896ab1`. No scientific data artifact is changed.

**Classification:** `PASS_WITH_RECORDED_WORKTREE_DEVIATIONS`; no unexplained
modification.

## 4. Current Studio manifest

`reports/scientific_studio/manifest.json` has **104 entries**, all matching
current bytes, including 15 docs, 33 code files, 45 screenshots, and 11 final
audit evidence files. The manifest records source head `fafaa40` and explicitly
notes that the final commit adds only the generated manifest.

**CURRENT_ARTIFACT_INTEGRITY: PASS.**

## 5. Command

```powershell
python -X utf8 scripts/studio_provenance_audit.py --pretty
```

Output: `reports/scientific_studio/release_candidate/integrity_report.json`.

Exit codes:

- `0` — no unexplained modification and no unverifiable snapshot;
- `1` — unexplained modification, unverifiable historical snapshot, or current
  manifest mismatch.

For this release the command exits **1** because of the two unverifiable audit
scripts. That is intentional: the integrity problem is classified, not hidden.

## 6. Why the old validators still fail

`validate_kato_final_audit.py` and `validate_kato_interactions.py` compare frozen
manifest hashes to the **current worktree**. Later authorized commits changed
those code/docs files. Their failure is expected at any revision after the
manifest's source revision, and they remain useful only as historical checks.

Use `scripts/studio_provenance_audit.py` for contemporary verification. Do not
rewrite the old manifests to make the old validators green.
