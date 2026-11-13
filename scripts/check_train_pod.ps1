#Requires -Version 5.1
<#
.SYNOPSIS
Verify the Vienna Studio deployment on the Run:ai train pod.

.DESCRIPTION
Read-only checks:
  1. the pod exists and is Running;
  2. the pod-local Streamlit health endpoint returns "ok";
  3. the local port-forward health endpoint returns "ok" (unless -SkipLocal);
  4. the deployed app.py matches the local working tree by SHA-256.

The script does not start, stop or modify the deployment. It exits 1 if any
check fails, which makes it usable from a shell or a monitoring job.

.PARAMETER Kubeconfig
Path to the kubeconfig. Defaults to $env:KUBECONFIG. Required when the
environment variable is not set.

.PARAMETER Namespace
Run:ai namespace. Default: runai-romania-dev.

.PARAMETER Pod
Pod name. Default: train-01-0-0.

.PARAMETER LocalPort
Local port-forward port. Default: 8504.

.PARAMETER RemotePort
Streamlit port inside the pod. Default: 8502.

.PARAMETER RepoRoot
Local repository root. Default: the parent of this script's directory.

.PARAMETER RemoteRepo
Repository path inside the pod. Default: /work/neurobranch-celegans.

.PARAMETER SkipLocal
Skip the local port-forward check, for example from inside the cluster.

.EXAMPLE
powershell -File scripts/check_train_pod.ps1 -Kubeconfig C:\path\to\kubeconfig.yaml

.EXAMPLE
powershell -File scripts/check_train_pod.ps1 -SkipLocal
#>
[CmdletBinding()]
param(
    [string]$Kubeconfig = $env:KUBECONFIG,
    [string]$Namespace = "runai-romania-dev",
    [string]$Pod = "train-01-0-0",
    [ValidateRange(1, 65535)]
    [int]$LocalPort = 8504,
    [ValidateRange(1, 65535)]
    [int]$RemotePort = 8502,
    [string]$RepoRoot = "",
    [string]$RemoteRepo = "/work/neurobranch-celegans",
    [switch]$SkipLocal
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

if (-not $RepoRoot) {
    $RepoRoot = Split-Path -Parent $PSScriptRoot
}

function Fail([string]$Message) {
    Write-Host "FAIL: $Message" -ForegroundColor Red
    exit 1
}

if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
    Fail "kubectl is not on PATH"
}
if (-not $Kubeconfig) {
    Fail "No kubeconfig. Pass -Kubeconfig or set `$env:KUBECONFIG."
}
if (-not (Test-Path -LiteralPath $Kubeconfig)) {
    Fail "Kubeconfig not found: $Kubeconfig"
}

$localApp = Join-Path $RepoRoot "src/neurobranch/bundlenet_sim/app.py"
if (-not (Test-Path -LiteralPath $localApp)) {
    Fail "Local app.py not found under RepoRoot: $RepoRoot"
}

$report = [ordered]@{
    checked_at = (Get-Date).ToUniversalTime().ToString("o")
    namespace  = $Namespace
    pod        = $Pod
    local_port = $LocalPort
    remote_port = $RemotePort
    remote_repo = $RemoteRepo
    checks     = [ordered]@{}
}

$phase = & kubectl --kubeconfig $Kubeconfig -n $Namespace get pod $Pod `
    -o jsonpath='{.status.phase}' 2>$null
if ($LASTEXITCODE -ne 0) {
    Fail "Cannot read pod $Namespace/$Pod with the supplied kubeconfig"
}
$phase = "$phase".Trim()
$report.checks.pod_phase = $phase
if ($phase -ne "Running") {
    Fail "Pod phase is '$phase', expected Running"
}

$remoteHealth = & kubectl --kubeconfig $Kubeconfig -n $Namespace exec $Pod -- `
    curl -fsS "http://127.0.0.1:$RemotePort/_stcore/health" 2>$null
if ($LASTEXITCODE -ne 0) {
    Fail "Pod-local Streamlit health check failed on port $RemotePort"
}
$remoteHealth = "$remoteHealth".Trim()
$report.checks.remote_health = $remoteHealth
if ($remoteHealth -ne "ok") {
    Fail "Pod-local health returned '$remoteHealth', expected 'ok'"
}

if (-not $SkipLocal) {
    try {
        $localHealth = (
            Invoke-WebRequest -UseBasicParsing `
                -Uri "http://127.0.0.1:$LocalPort/_stcore/health" `
                -TimeoutSec 10
        ).Content.Trim()
    }
    catch {
        Fail "Local port-forward is not healthy on $LocalPort. Start it with: kubectl --kubeconfig <kubeconfig> -n $Namespace port-forward pod/$Pod ${LocalPort}:${RemotePort}"
    }
    $report.checks.local_health = $localHealth
    if ($localHealth -ne "ok") {
        Fail "Local health returned '$localHealth', expected 'ok'"
    }
}

$localHash = (Get-FileHash -LiteralPath $localApp -Algorithm SHA256).Hash.ToLowerInvariant()
$remoteHash = & kubectl --kubeconfig $Kubeconfig -n $Namespace exec $Pod -- `
    sha256sum "$RemoteRepo/src/neurobranch/bundlenet_sim/app.py" 2>$null
if ($LASTEXITCODE -ne 0) {
    Fail "Cannot hash the deployed app.py"
}
$remoteHash = ("$remoteHash" -split "\s+")[0].ToLowerInvariant()
$report.checks.local_app_sha256 = $localHash
$report.checks.remote_app_sha256 = $remoteHash
if ($localHash -ne $remoteHash) {
    Fail "Deployed app.py differs from the local working tree"
}

$report.status = "pass"
$report | ConvertTo-Json -Depth 4
exit 0
