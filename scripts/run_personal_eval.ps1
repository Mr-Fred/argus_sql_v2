<#
.SYNOPSIS
    Runs live BIRD-SQL evaluation natively on a personal Windows computer and
    exports a path-scrubbed, clean-room-verified, SHA-256-signed feedback bundle.

.DESCRIPTION
    Executes the 4-stage personal evaluation loop on Windows PowerShell 5.1+ / 7+:
      1. Bootstraps repo-local clean-room Git hooks and audits repository state.
      2. Runs live BIRD-SQL evaluation (`--mode bird`) using `.env` or local
         `gcloud auth application-default login` (`%APPDATA%\gcloud\...`).
      3. Scans `results/feedback/` and `paper/tables/bird_live.tex` for zero
         workstation paths, usernames, or credential leaks.
      4. Verifies the canonical SHA-256 cryptographic digest of the bundle.

.EXAMPLE
    .\scripts\run_personal_eval.ps1 -Limit 20 -OnlyArgus

.EXAMPLE
    .\scripts\run_personal_eval.ps1 -Provider vertex -Model gemini-2.5-flash
#>
[CmdletBinding()]
param(
    [string]$BirdDir = $(if ($env:ARGUS_BIRD_DIR) { $env:ARGUS_BIRD_DIR } else { "data/bird/mini_dev" }),
    [string]$QuestionsFile = $(if ($env:ARGUS_QUESTIONS_FILE) { $env:ARGUS_QUESTIONS_FILE } else { "mini_dev_sqlite.json" }),
    [ValidateSet("auto", "vertex", "openai")]
    [string]$Provider = $(if ($env:ARGUS_PROVIDER) { $env:ARGUS_PROVIDER } else { "auto" }),
    [string]$Model = $(if ($env:ARGUS_MODEL) { $env:ARGUS_MODEL } elseif ($env:ARGUS_LLM_MODEL) { $env:ARGUS_LLM_MODEL } else { "auto" }),
    [ValidateRange(0, 1000000)]
    [int]$Limit = $(if ($env:ARGUS_LIMIT) { [int]$env:ARGUS_LIMIT } else { 0 }),
    [switch]$OnlyArgus,
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
    throw "Python 3.11+ interpreter not found on PATH. Install Python 3.12+ (e.g., 'winget install Python.Python.3.12')."
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

# Resolve repository root strictly relative to this script's location.
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

    Write-Host "[1/4] Bootstrapping local clean-room Git hooks..."
    Invoke-ArgusPython -PythonSpec $PySpec -Arguments @(
        "-m", "argus_sql.guardrails.cli",
        "--repo-root", $RepoRoot,
        "--bootstrap-git",
        "--verify-all"
    )

    $ResolvedBirdCandidate = if ([System.IO.Path]::IsPathRooted($BirdDir)) {
        $BirdDir
    } else {
        Join-Path $RepoRoot $BirdDir
    }
    if (-not (Test-Path -LiteralPath $ResolvedBirdCandidate -PathType Container)) {
        throw "[ERROR] BIRD dataset directory '$BirdDir' not found. Extract MINIDEV.zip into '$BirdDir' or pass -BirdDir."
    }

    $LatexOutDir = Join-Path $RepoRoot "paper/tables"
    $FeedbackOutDir = Join-Path $RepoRoot "results/feedback"
    $EvalArgs = @(
        "-m", "argus_sql.evaluation.cli",
        "--mode", "bird",
        "--bird-dir", $ResolvedBirdCandidate,
        "--questions-filename", $QuestionsFile,
        "--provider", $Provider,
        "--model", $Model,
        "--limit", "$Limit",
        "--latex-out-dir", $LatexOutDir,
        "--feedback-out-dir", $FeedbackOutDir
    )
    if ($OnlyArgus -and ($ExtraArgs -notcontains "--only-argus")) {
        $EvalArgs += "--only-argus"
    }
    if ($null -ne $ExtraArgs -and $ExtraArgs.Count -gt 0) {
        $EvalArgs += $ExtraArgs
    }

    Write-Host "[2/4] Running live BIRD-SQL evaluation (provider=$Provider, model=$Model, limit=$Limit)..."
    Invoke-ArgusPython -PythonSpec $PySpec -Arguments $EvalArgs

    Write-Host "[3/4] Scanning exported feedback bundle & LaTeX table for zero leaks..."
    Invoke-ArgusPython -PythonSpec $PySpec -Arguments @(
        "-m", "argus_sql.guardrails.cli",
        "--repo-root", $RepoRoot,
        "--scan-path", $FeedbackOutDir
    )
    Invoke-ArgusPython -PythonSpec $PySpec -Arguments @(
        "-m", "argus_sql.guardrails.cli",
        "--repo-root", $RepoRoot,
        "--scan-path", (Join-Path $LatexOutDir "bird_live.tex")
    )

    Write-Host "[4/4] Verifying SHA-256 cryptographic integrity of feedback bundle..."
    Invoke-ArgusPython -PythonSpec $PySpec -Arguments @(
        "-m", "argus_sql.evaluation.feedback_cli",
        "verify",
        "--repo-root", $RepoRoot,
        "--feedback-dir", $FeedbackOutDir
    )

    Write-Host ""
    Write-Host "[DONE] Sanitized feedback bundle verified and ready to push!"
    Write-Host "  - Bundle JSON : results/feedback/bird_feedback_bundle.json"
    Write-Host "  - Triage MD   : results/feedback/triage_report.md"
    Write-Host "  - LaTeX Table : paper/tables/bird_live.tex"
    Write-Host ""
    Write-Host "Next step (PowerShell Git sync):"
    Write-Host "  git add results/feedback/bird_feedback_bundle.json results/feedback/triage_report.md paper/tables/bird_live.tex"
    Write-Host '  git commit -m "chore(eval): publish sanitized BIRD feedback bundle"'
    Write-Host "  git push origin main"
} finally {
    Pop-Location
}
