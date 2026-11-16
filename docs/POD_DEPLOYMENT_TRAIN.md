# Deploying the Vienna Studio on the `train` pod

**Status:** deployment runbook. This is infrastructure documentation, not a scientific result and not
a claim that any open data gap is closed.

## Target

- Run:ai workload: **`train-01-0-0`**
- Namespace: **`runai-romania-dev`**
- Pod path: **`/work/neurobranch-celegans`**
- App port inside the pod: **`8502`**
- Local port-forward port: **`8504`**
- Runtime: Python 3.13; the image already has NumPy, pandas, SciPy, scikit-learn, h5py, PyTorch and
  PyYAML. Streamlit and Plotly are installed at deployment time.
- Kubeconfig: supplied locally, never committed. The `runai` CLI token may be expired; use
  `kubectl --kubeconfig "$env:KUBECONFIG"` instead.

The pod has no `/workspace`; `/work` is persistent and writable. Do not assume the old
`/workspace/vienna-studio` deployment exists on this pod.

## Verified run

On 2026-11-13 the archive built from app commit `5f05a4c` was deployed to `train-01-0-0` with
Streamlit `1.67.0` and Plotly `7.1.0`. The pod-side health endpoint returned `ok`, the local
forward returned HTTP `200`, and the Streamlit `AppTest` harness reported zero exceptions on all
seven workspace pages. The deployed `app.py` SHA-256 was
`6e921810456e18c99dc6122165eb8768ea4cfd594158224aefdfc7013ffcc7b5`.

On 2026-11-14, repo commit `6c73235` was copied incrementally (no archive rebuild) to the same
pod: `app.py`, the new `scope.py`, `handoff.py`, and the updated Denis documents. The app was
restarted and the checker passed with local and remote `app.py` SHA-256
`f2d6c22eea681b867448d1ccbeb34c4bb6801fd787d5770e7906b34aa5e65a36`. `AppTest` reported zero
exceptions on all **eight** workspace pages. The same app hash was browser-QA'd locally at 1280,
1440 and 1920 px (24 page captures: zero console errors, zero horizontal overflow); the deployed
health endpoint and hash matched the local copy. This is infrastructure evidence, not a scientific
result.

On 2026-11-14, repo commit `3a14432` added the scope claim-readiness advisories and the
`schemas/scope_response.schema.json` contract. The incremental copy was restarted on the same pod;
the checker passed with local and remote `app.py` SHA-256
`c4c93b74a83ed0d5188cdfe976d6a233bf46ae7559f488ea2d0d3672fa9d57a6`. The full suite reported
**648 passed**, `AppTest` covered eight pages, and the same hash passed 24 browser captures at the
three widths with zero console errors and zero horizontal overflow.

On 2026-11-14, repo commit `96494a5` added the SHA-256 scope digest and digest-validated JSON
import. The app was restarted; the checker passed with local and remote `app.py` SHA-256
`98be1ff4ad665123ba3c58578147c4c5bea454daba2eb97f2ffb23df5e0091de`. The full suite reported
**653 passed**. Browser QA repeated 24 page captures at the three widths with zero errors/overflow,
and a downloaded JSON transcription was re-imported into the form with its digest and stop rule
intact.

On 2026-11-14, repo commit `98098c5` added the returned signed-brief audit. The app was restarted;
the checker passed with local and remote `app.py` SHA-256
`3f34de3fb66fe27dbfb5c8c749c26114b2943c0841cc6615e53974587ce192c7`. The full suite reported
**666 passed**. Browser QA repeated 24 page captures with zero errors/overflow, and a synthetic
signed block (reported signer `QA reviewer`, not Denis) reached
`matches_unverified_identity` on the deployed pod, offered the audit download, and kept the
identity caveat visible.

Later on 2026-11-14, repo commit `8bbe377` replaced the scope form with the two-column Decision
Room v3. The checker passed with local and remote `app.py` SHA-256
`79d8b3cf6cc0368ae52169392879641871249f7e9a64be8ab40442e511c8a80b`; the marker records
`app_source_commit=8bbe377` and `deployment_method=incremental_copy`. The
full suite reported **697 passed**. Local and deployed browser QA found the readiness rail, live
contract, and signed-block rail update with zero console errors and zero horizontal overflow at
1280 and 1024 px; the complete import → export → signed-block path was exercised locally.

On 2026-11-15, repo commit `f68a1cc` added the change-aware Decision Room v4. The pod process was
restarted after the incremental copy because Streamlit had cached the previous module. The checker
passed with local and remote `app.py` SHA-256
`430b52a82d4fc0df3db34a22788d9cbbe6e61599b4eb5cfd9b6e917b38b1c016`; the marker records
`app_source_commit=f68a1cc`. The full suite reported **704 passed**. Local and deployed QA exercised
import → contract export → decision edit: the contract showed one field-level change, the digest
stage became `BLOCKED`, and no console error or horizontal overflow appeared at 1440, 1024 or
768 px. The contract card was sticky with internal scrolling at desktop width and static below
900 px.

On 2026-11-16, repo commit `b12692c` added Decision Room v5 (option atlas, copyable digest and
decision packet). The pod was found Running with no listener on `8502`; its Python environment also
lacked `streamlit` and `plotly`, so both were installed at `streamlit==1.66.1` and `plotly==7.1.0`
before the app was restarted. The checker passed with local and remote `app.py` SHA-256
`3f08b9c85d5133356ea16b0c4af2468abf9ae7c3a5a177435039201c03379df3`; the marker records
`app_source_commit=b12692c`. The full suite reported **709 passed**. Local and deployed QA found the
atlas (4, 5 or 6 option cards depending on the inspected decision), the copyable digest, a
10-entry packet with a 9-file hash manifest and `unconfirmed_transcription` status, and the
post-export diff still blocking the digest stage. No console errors or horizontal overflow appeared
at 1440, 1024 or 768 px.

## Quick status check

The read-only checker validates the pod phase, both health endpoints and the deployed `app.py`
hash. It exits `0` only when every check passes:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\check_train_pod.ps1 `
  -Kubeconfig "<path-to-kubeconfig>"
```

Use `-SkipLocal` when checking from a host without the port-forward. On success the command prints a
JSON report with `status: pass`; on failure it prints a `FAIL:` line and exits `1`. It never starts,
stops or modifies the deployment.

## What is transferred

The deployment is deliberately smaller than the repository:

| Content | Purpose |
|---|---|
| tracked files from `git archive HEAD` | source, scripts, schemas, docs, reports |
| `data/processed/` | Tier 1/2/3 processed artifacts |
| `data/metadata/` | premotor neuron names |
| `references/external/Celegans_premotor/data/Atanas_data/` | Tier 3 recordings |

The other external reference checkouts, `.git`, tests, artifacts and figures are not needed at
runtime. The Atanas files are third-party research data; this is an internal, local-only deployment.
Do not expose the forwarded port publicly or redistribute those files without a licence review.

## 1. Build the archive locally

Run from the repository root after the branch is clean and pushed:

```powershell
$stage = Join-Path $env:TEMP "vienna-deploy"
$tar   = Join-Path $env:TEMP "vienna-deploy.tar.gz"
Remove-Item $stage,$tar -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $stage | Out-Null

git archive --format=tar --output="$env:TEMP\vienna-repo.tar" HEAD
tar -xf "$env:TEMP\vienna-repo.tar" -C $stage
Copy-Item data\processed -Destination "$stage\data\processed" -Recurse
Copy-Item data\metadata  -Destination "$stage\data\metadata"  -Recurse
New-Item -ItemType Directory -Path "$stage\references\external\Celegans_premotor\data" -Force | Out-Null
Copy-Item references\external\Celegans_premotor\data\Atanas_data `
  -Destination "$stage\references\external\Celegans_premotor\data\Atanas_data" -Recurse

tar -czf $tar -C $stage .
Get-FileHash $tar -Algorithm SHA256
```

Record the SHA-256. Keep the archive out of Git.

## 2. Copy and verify the archive

`kubectl cp` on Windows rejects a source path with a drive-letter colon, so run it from the archive's
directory with a relative source path:

```powershell
$KUBECONFIG = "<path-to-kubeconfig>"
Push-Location $env:TEMP
kubectl --kubeconfig $KUBECONFIG cp .\vienna-deploy.tar.gz `
  runai-romania-dev/train-01-0-0:/work/vienna-deploy.tar.gz
Pop-Location

kubectl --kubeconfig $KUBECONFIG -n runai-romania-dev exec train-01-0-0 -- `
  sha256sum /work/vienna-deploy.tar.gz
```

The remote hash must equal the local hash before extraction.

## 3. Extract in `/work`

```powershell
kubectl --kubeconfig $KUBECONFIG -n runai-romania-dev exec train-01-0-0 -- bash -lc @'
mkdir -p /work/neurobranch-celegans
tar -xzf /work/vienna-deploy.tar.gz -C /work/neurobranch-celegans
ls -la /work/neurobranch-celegans | head
'@
```

Verify that these paths exist:

```text
/work/neurobranch-celegans/src/neurobranch/bundlenet_sim/app.py
/work/neurobranch-celegans/data/processed/bundlenet_sim/
/work/neurobranch-celegans/data/processed/ncmcm/
/work/neurobranch-celegans/data/processed/premotor/
/work/neurobranch-celegans/references/external/Celegans_premotor/data/Atanas_data/
```

## 4. Install the missing runtime packages

Do not reinstall PyTorch or the scientific stack. Only Streamlit and Plotly are missing:

```powershell
kubectl --kubeconfig $KUBECONFIG -n runai-romania-dev exec train-01-0-0 -- `
  python -m pip install --user --no-cache-dir "streamlit>=1.35" "plotly>=5.20"

kubectl --kubeconfig $KUBECONFIG -n runai-romania-dev exec train-01-0-0 -- `
  python -c "import streamlit, plotly; print(streamlit.__version__, plotly.__version__)"
```

If pip changes NumPy in an incompatible way, pin the existing version or use a virtual environment
with `--system-site-packages`. Record the installed versions in the deployment log.

## 5. Start the app

```powershell
kubectl --kubeconfig $KUBECONFIG -n runai-romania-dev exec train-01-0-0 -- bash -lc @'
cd /work/neurobranch-celegans
nohup env PYTHONPATH="$PWD/src" \
  python -m streamlit run src/neurobranch/bundlenet_sim/app.py \
  --server.port 8502 --server.address 127.0.0.1 --server.headless true \
  > /work/vienna-studio.log 2>&1 &
echo $! > /work/vienna-studio.pid
sleep 5
curl -fsS http://127.0.0.1:8502/_stcore/health && echo
pgrep -af "streamlit run" || true
'@
```

If the process does not survive the `kubectl exec` session, use `setsid -f` or a detached launcher.
Keep `/work/vienna-studio.log` for diagnosis.

## 6. Port-forward and verify

The `kubectl port-forward` command exposes the pod-local port only on this workstation:

```powershell
kubectl --kubeconfig $KUBECONFIG -n runai-romania-dev port-forward `
  pod/train-01-0-0 8504:8502
```

Run the forward as a background job and verify:

```powershell
(Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8504/_stcore/health -TimeoutSec 10).Content
(Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8504 -TimeoutSec 10).StatusCode
```

Acceptance is `ok` and `200`. The forwarded app is local to this workstation; it is not a public URL.

Then open **Evidence & lab** and check:

- Gap closure table and the six gates;
- `Download all gap documents (ZIP)`;
- Incoming data check with the annotation and trial uploaders;
- Tier 3 can see the Atanas recordings.

If the page fails, read the pod log:

```powershell
kubectl --kubeconfig $KUBECONFIG -n runai-romania-dev exec train-01-0-0 -- `
  tail -100 /work/vienna-studio.log
```

## 7. Update, restart and stop

To update, rebuild the archive from the new commit, repeat sections 1–3, then restart Streamlit:

```powershell
kubectl --kubeconfig $KUBECONFIG -n runai-romania-dev exec train-01-0-0 -- bash -lc @'
pkill -f "streamlit run src/neurobranch/bundlenet_sim/app.py" || true
cd /work/neurobranch-celegans
nohup env PYTHONPATH="$PWD/src" \
  python -m streamlit run src/neurobranch/bundlenet_sim/app.py \
  --server.port 8502 --server.address 127.0.0.1 --server.headless true \
  > /work/vienna-studio.log 2>&1 &
'@
```

Stop the local port-forward job with its job id or `Ctrl-C`; stop the app with the `pkill` command.
The deployment directory is removed only on an explicit request:

```text
rm -rf /work/neurobranch-celegans
```

## Known limits

- `/work` persistence is provided by the cluster, not guaranteed by this repository.
- A pod restart can stop the app; the port-forward must be restarted too.
- The kubeconfig token and pod credentials are never committed.
- The deployment does not add data, close a scientific gap, or change any claim in the ledger.
