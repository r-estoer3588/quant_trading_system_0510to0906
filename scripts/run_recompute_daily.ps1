<#
.SYNOPSIS
  Safe wrapper to run recompute_rolling_bulk.py in dry-run or execute mode.

.DESCRIPTION
  This script calls the Python recompute script with safe defaults. It creates a lock file
  to avoid duplicate recompute runs within the same day and writes the JSON report to
  results_csv_test/recompute_rolling_bulk_report_YYYYMMDD_HHMMSS.json.

.PARAMETER DryRun
  If supplied, runs in dry-run mode (default: DryRun).

.PARAMETER Execute
  If supplied, runs with --execute (writes changes). Use with caution.

.PARAMETER Backup
  If supplied together with Execute, passes --backup to the Python script.

.PARAMETER Workers
  Number of workers to pass to the recompute script. Default: 2
#>
param(
    [switch]$DryRun,
    [switch]$Execute,
    [switch]$Backup,
    [int]$Workers = 2
)

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$ReportsDir = Join-Path $ProjectRoot 'results_csv_test'
if (-not (Test-Path $ReportsDir)) { New-Item -ItemType Directory -Path $ReportsDir | Out-Null }

$now = Get-Date -Format 'yyyyMMdd_HHmmss'
$reportPath = Join-Path $ReportsDir ("recompute_rolling_bulk_report_$now.json")
$lockFile = Join-Path $ProjectRoot 'tmp_recompute_lock'

function Write-Log { param($m) Write-Host "[recompute] $m" }

# Prevent multiple recomputes within 6 hours by default
$lockTTLHours = 6
if (Test-Path $lockFile) {
    try {
        $lockAgeHours = ((Get-Date) - (Get-Item $lockFile).LastWriteTime).TotalHours
        if ($lockAgeHours -lt $lockTTLHours) {
            Write-Log "Lock file exists and is recent ($([math]::Round($lockAgeHours,1))h). Skipping recompute."
            exit 0
        }
        else {
            Write-Log "Lock file exists but older than $lockTTLHours h; continuing and rotating lock."
        }
    }
    catch { Write-Log "Could not inspect lock file: $_" }
}

# Create/refresh lock file
try { Set-Content -Path $lockFile -Value (Get-Date).ToString('o') -Force } catch { Write-Log "Failed to create lock file: $_" }

$py = Join-Path $ProjectRoot 'venv\Scripts\python.exe'
$script = Join-Path $ProjectRoot 'scripts\recompute_rolling_bulk.py'
if (-not (Test-Path $py)) { $py = 'python' }
if (-not (Test-Path $script)) { Write-Log "recompute_rolling_bulk.py not found: $script"; exit 1 }

if (-not $DryRun.IsPresent -and -not $Execute.IsPresent) { $DryRun = $true }

$cmd = @($py, $script, '--workers', $Workers)
if ($DryRun.IsPresent) { $cmd += '--dry-run' }
if ($Execute) { $cmd += '--execute' }
if ($Backup) { $cmd += '--backup' }
$cmd += '--output'
$cmd += $reportPath

Write-Log "Running: $($cmd -join ' ')"
& $cmd 2>&1 | Tee-Object -FilePath (Join-Path $ProjectRoot "logs\recompute_rolling_${now}.log")
$rc = $LASTEXITCODE
if ($rc -eq 0) { Write-Log ("Recompute finished: report={0}" -f $reportPath) } else { Write-Log ("Recompute finished with code {0}: report={1}" -f $rc, $reportPath) }

exit $rc
