# Scientific Studio reproducibility

## 1. Environment

Verified on 2026-11-03:

| Component | Version |
| --- | --- |
| Python | 3.13.5 |
| Streamlit | 1.66.1 |
| Plotly | 7.1.0 |
| pandas | 2.3.3 |
| NumPy | 2.3.1 |
| scikit-learn | 1.9.1 |
| pytest | project environment |
| ruff | project environment |
| mkdocs | 1.6.1 (strict build) |

The repository requires Python `>=3.10,<3.14`. The Studio uses only the
project's existing dependencies plus Streamlit/Plotly.

## 2. Data identity

```text
path      data/raw/kato2015/WT_NoStim.mat
size      56,390,975 bytes
sha256    7531811c1718e26a0455f59cb2f8188cf25809278cfea85770dfe03315a0098a
OSF       https://osf.io/2395t/
```

Do not regenerate evidence from a different archive. `metadata.py` checks the
audited hash/size flags; the dataset test recomputes the SHA256.

## 3. Build the compact trajectory artifact

```powershell
python -X utf8 scripts/studio_build_trajectories.py
```

The command:

- refits B0/B1m/B3/B5 at every saved H20 landmark using the same windows and
  ridge as `scripts/kato_common_state_main.py`;
- samples 100 innovations with seed `7000 + session*10000 + landmark`;
- rolls 100 steps and stores the first 20;
- writes `kato_trajectories.npz` and `kato_trajectories.json`;
- validates 30 showcase B3/B5 H20 scores against `model_per_landmark.csv`.

The frozen main pipeline uses the same seed formula and H=100 innovation
sampling. A full `scripts/kato_common_state_main.py --skip-surrogates
--skip-synthetic --out-dir ...` rerun (555.6 s) reproduced the frozen B3 values
exactly (maximum difference 0.0), which validated the pipeline independently of
the compact artifact.

Current artifact:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `kato_trajectories.npz` | 5,282,051 | `ba7a3a440b6266e5f2364e40a84709ce64de4243f8289e25b82a43c7ea1fff35` |
| `kato_trajectories.json` | 57,663 | `b98dec015d69514d019f89d0712918f96ce505c91e0b8f2416ceb0f18b32bcf4` |

`score_validation.max_abs_difference = 1.2678296599233363e-06` against a
`2e-5` storage tolerance. This is a storage-path check, not a new inferential
tolerance. The difference is float32/PCA sign/rounding scale.

## 4. Reproduce the evidence tables

The frozen artifacts are committed. To rebuild them from scratch, use the
documented Master Review commands:

```powershell
python -X utf8 scripts/kato_common_state_main.py --skip-surrogates --skip-synthetic --out-dir reports/simulator/kato_common_state
python -X utf8 scripts/master_review/build_audit_ledgers.py
```

Do not write into the frozen directories during Studio development unless you
intend to create a new audited artifact. The Studio loader validates the
primary contrast against `statistical_audit.json`.

## 5. Run tests

```powershell
python -X utf8 -m pytest -q tests/test_studio_artifacts.py tests/test_studio_ui_contract.py tests/test_studio_app_smoke.py
python -X utf8 -m pytest -q
python -X utf8 -m ruff check src/neurobranch/studio app/neurobranch_studio.py
python -X utf8 -m mkdocs build --strict
```

The artifact tests require the raw archive. The app-smoke tests use
`streamlit.testing.v1.AppTest`; they do not launch a browser.

## 6. Run the application

```powershell
python -X utf8 -m streamlit run app/neurobranch_studio.py
```

Optional browser QA uses Playwright Chromium at 1920×1080 and 1366×768. The
screenshots and geometry checks are recorded in
`docs/STUDIO_TESTING_AND_QA.md`; the PNGs live under
`reports/scientific_studio/screenshots/`.

## 7. Determinism

| Source | Rule |
| --- | --- |
| PCA | `random_state=17`, train split only |
| innovations | seed `7000 + session*10000 + landmark` |
| B3/B5 | window 500, ridge 1.0, spectral cap 0.995 |
| B1m | window 200, ridge 1.0 |
| display | no random sampling except shuffled display order? No: order is deterministic |
| perturbation | deterministic given session, landmark, label, magnitude |
| figures | fixed camera/uirevision except user orbit |

The presentation deck and all route content are deterministic from artifacts.
Plotly playback is client-side and deterministic in frame order.

## 8. Known audit inconsistency (do not hide)

`scripts/master_review/independent_model_check.py` reconstructs a reference B3
value that did not match the frozen row at the inspected indices
(e.g. landmark 2750: `1.473899` vs frozen `1.360753`; landmark 2690:
`2.410819` vs frozen `2.498644`). A full common-state rerun reproduced the
frozen values exactly, and the Studio artifact matches `model_per_landmark.csv`
within `1.27e-6`. The independent-check code path therefore has an unresolved
reconstruction discrepancy; it must not be used as a second source of truth
until that discrepancy is explained. The Studio uses the frozen paired rows and
the full rerun, not the independent-check reconstruction.

## 9. Version-control policy

- Branch: `research/neurobranch-scientific-studio`
- No push, amend, reset, rebase, or force-push.
- Additive commits only.
- Frozen science files are not rewritten.
- The Studio documents and artifacts are new files; legacy apps are patched
  only where they contradict the audited wording.
- The working tree should be clean at handoff; the final report records the
  exact HEAD.

## 10. Manifest

`reports/scientific_studio/manifest.json` contains the source Git HEAD, branch,
artifact sizes/hashes, document list, test command, and the source-of-truth
hierarchy. It is generated after the content commit and committed separately;
the final repository HEAD therefore adds only the manifest file. Regenerate it
after final edits:

```powershell
python -X utf8 scripts/studio_manifest.py
```
