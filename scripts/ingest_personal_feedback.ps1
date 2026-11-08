<#
.SYNOPSIS
    Ingests a sanitized `bird_feedback_bundle.json` on Windows PowerShell with
    zero cloud credentials, updating `bird_live.tex` and `triage_report.md`.

.DESCRIPTION
    Mirrors `scripts/ingest_personal_feedback.sh` natively for PowerShell 5.1+/7+:
      1. Runs a clean-room quarantine scan on the bundle before parsing.
      2. Verifies the canonical SHA-256 digest, promotes quarantine bundles if
         applicable, and regenerates `paper/tables/bird_live.tex` + `triage_report.md`.
      3. Runs full repository clean-room guardrails and hermetic unit tests.

.EXAMPLE
    .\scripts\ingest_personal_feedback.ps1
#>
[CmdletBinding()]
param(
    [string]$BundlePath = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-PythonCommand {
    <#
    .SYNOPSIS
        Resolves workspace `.venv` or global Python 3.11+ while skipping Store stubs.
    #>
    param([Parameter(Mandatory = $true)][string]$RepoRootPath)
    $ErrorActionPreference = "Continue"
    $venvCandidates = @(
        (Join-Path $RepoRootPath ".venv\Scripts\python.exe"),
        (Join-Path $RepoRootPath "venv\Scripts\python.exe")
    )
    foreach ($venvExe in $venvCandidates) {
        if (Test-Path -LiteralPath $venvExe -PathType Leaf) {
            & $venvExe -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >$null 2>$null
            if ($LASTEXITCODE -eq 0) {
                return @{ Exe = $venvExe; Prefix = [string[]]@() }
            }
        }
    }
    $candidates = @(
        @{ Exe = "python"; Prefix = @() },
        @{ Exe = "py"; Prefix = @("-3") },
        @{ Exe = "python3"; Prefix = @() }
    )
    foreach ($cand in $candidates) {
        $cmdInfo = Get-Command $cand.Exe -ErrorAction SilentlyContinue
        if ($null -ne $cmdInfo) {
            $probeArgs = @($cand.Prefix) + @(
                "-c",
                "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
            )
            & $cmdInfo.Source @probeArgs >$null 2>$null
            if ($LASTEXITCODE -eq 0) {
                return @{
                    Exe    = $cmdInfo.Source
                    Prefix = [string[]]$cand.Prefix
                }
            }
        }
    }
    throw "Python 3.11+ interpreter not found on PATH."
}

function Invoke-ArgusPython {
    <#
    .SYNOPSIS
        Executes a Python module directly (no shell interpolation) and checks exit code.
    #>
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$PythonSpec,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )
    $allArgs = @($PythonSpec.Prefix) + $Arguments
    & $PythonSpec.Exe @allArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: python $($Arguments -join ' ')"
    }
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    $PySpec = Resolve-PythonCommand -RepoRootPath $RepoRoot
    $SrcDir = Join-Path $RepoRoot "src"
    $PathSep = [System.IO.Path]::PathSeparator
    if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
        $env:PYTHONPATH = $SrcDir
    } else {
        $env:PYTHONPATH = "${SrcDir}${PathSep}$($env:PYTHONPATH)"
    }

    $DefaultQuarantine = Join-Path $RepoRoot "audit_import/bird_feedback_bundle.json"
    $DefaultTracked = Join-Path $RepoRoot "results/feedback/bird_feedback_bundle.json"

    if (-not [string]::IsNullOrWhiteSpace($BundlePath)) {
        $TargetBundle = if ([System.IO.Path]::IsPathRooted($BundlePath)) {
            $BundlePath
        } else {
            Join-Path $RepoRoot $BundlePath
        }
    } elseif (Test-Path -LiteralPath $DefaultQuarantine -PathType Leaf) {
        $TargetBundle = $DefaultQuarantine
    } else {
        $TargetBundle = $DefaultTracked
    }

    if (-not (Test-Path -LiteralPath $TargetBundle -PathType Leaf)) {
        throw "[ERROR] Feedback bundle not found at '$TargetBundle'. Run 'git pull' or place bird_feedback_bundle.json in audit_import/."
    }

    Write-Host "[1/3] Running quarantine clean-room scan..."
    Invoke-ArgusPython -PythonSpec $PySpec -Arguments @(
        "-m", "argus_sql.guardrails.cli",
        "--repo-root", $RepoRoot,
        "--scan-path", $TargetBundle
    )

    $ResolvedTarget = (Resolve-Path -LiteralPath $TargetBundle).Path
    $TrackedParent = Split-Path -Parent $DefaultTracked
    if (-not (Test-Path -LiteralPath $TrackedParent -PathType Container)) {
        New-Item -ItemType Directory -Force -Path $TrackedParent | Out-Null
    }
    $ResolvedTracked = [System.IO.Path]::GetFullPath($DefaultTracked)

    $IngestArgs = @(
        "-m", "argus_sql.evaluation.feedback_cli",
        "ingest",
        "--repo-root", $RepoRoot,
        "--bundle", $ResolvedTarget,
        "--report-out", (Join-Path $RepoRoot "results/feedback/triage_report.md"),
        "--latex-out-dir", (Join-Path $RepoRoot "paper/tables")
    )
    if ($ResolvedTarget -ne $ResolvedTracked) {
        $IngestArgs += @("--promote-to", $ResolvedTracked)
    }
    if ($null -ne $ExtraArgs -and $ExtraArgs.Count -gt 0) {
        $IngestArgs += $ExtraArgs
    }

    Write-Host "[2/3] Verifying SHA-256 digest, updating LaTeX table & generating triage report..."
    Invoke-ArgusPython -PythonSpec $PySpec -Arguments $IngestArgs

    Write-Host "[3/3] Running full repository clean-room & hermetic unit test verification..."
    Invoke-ArgusPython -PythonSpec $PySpec -Arguments @(
        "-m", "argus_sql.guardrails.cli",
        "--repo-root", $RepoRoot,
        "--verify-all"
    )
    Invoke-ArgusPython -PythonSpec $PySpec -Arguments @(
        "-m", "unittest", "discover",
        "-s", $SrcDir,
        "-p", "*_test.py"
    )

    Write-Host "[DONE] Feedback ingested cleanly! Review results/feedback/triage_report.md."
} finally {
    Pop-Location
}
