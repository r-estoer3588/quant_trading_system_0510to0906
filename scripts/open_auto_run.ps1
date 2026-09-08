<#
.SYNOPSIS
    Task Scheduler wrapper for the open (US market-open) auto-submit runner.

.DESCRIPTION
    Thin launcher for scripts/open_auto_run.py. Runs from THIS worktree
    (pinned to origin/main so equity-linked sizing is in effect) whose
    data_cache/results_csv are junctions to the primary repo (shared live data).

    Responsibilities kept in PowerShell (everything else is in the Python
    orchestrator so Japanese text never touches the cp932 console codepage):
      - Load the PRIMARY repo .env (Alpaca creds + NTFY_TOPIC) because this
        worktree has no .env of its own.
      - Force UTF-8 for the child python (PYTHONUTF8 / PYTHONIOENCODING) to
        avoid the cp932 UnicodeDecodeError the one-off runner hit.
      - Invoke python with pass-through flags and tee a launch log.

    PAPER ONLY. Never touches live money. The Python runner asserts paper env,
    gates on market-open + signal count, and enforces exit->entry ordering.

.NOTES
    Exit codes propagate from open_auto_run.py
    (0 ok / 3 aborted-before-trades / 4 trades-complete, observability degraded).
    Keep this file ASCII-only; the Python side owns all Japanese output.

    -AllowClosed only bypasses a KNOWN market-closed clock. A clock that cannot be
    read at all (DNS/API outage) still aborts unless -AllowClockUnknown is given.
    OPEN_RUN_ALLOW_CLOCK_UNKNOWN from .env is explicitly scrubbed by this wrapper:
    scheduled runs can arm that emergency hatch only through -AllowClockUnknown.

    Non-dry scheduled/manual wrapper runs also acquire an exclusive INFLIGHT.lock.
    It is released only for exit 0/4 (trades completed) or exit 3 (pre-trade abort).
    An unexpected crash/kill leaves the lock behind so a retry fails closed instead
    of re-entering the order stages before Python has written DONE.lock.
#>
param(
    [string]$Date = "",
    [switch]$DryRun = $false,
    [switch]$AllowClosed = $false,
    [switch]$AllowClockUnknown = $false,
    [switch]$SkipSignals = $false,
    [switch]$Force = $false,
    [switch]$FlattenAll = $false,
    [switch]$NoPublish = $false,
    [int]$MinSignals = 10,
    [double]$PollTimeout = 300,
    [string]$PrimaryRoot = "C:\Repos\quant_trading_system_0510to0906"
)

$ErrorActionPreference = "Continue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorktreeRoot = Split-Path -Parent $ScriptDir
$LogDir = Join-Path $WorktreeRoot "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LaunchLog = Join-Path $LogDir "open_auto_run_launch_$Stamp.log"

function Write-Launch {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -Path $LaunchLog -Value $line -Encoding UTF8
}

Write-Launch "=== open_auto_run.ps1 launch ==="
Write-Launch "WorktreeRoot: $WorktreeRoot"
Write-Launch "PrimaryRoot : $PrimaryRoot"

# --- load PRIMARY .env (creds + NTFY) into this process env --------------
$EnvFile = Join-Path $PrimaryRoot ".env"
if (Test-Path $EnvFile) {
    Write-Launch "loading .env: $EnvFile"
    Get-Content $EnvFile | ForEach-Object {
        $line = $_
        if ($line -match '^\s*#') { return }
        if ($line -match '^\s*([^#=\s]+)\s*=\s*(.*)$') {
            $k = $matches[1].Trim()
            $v = $matches[2].Trim()
            if ($v.Length -ge 2) {
                if (($v.StartsWith('"') -and $v.EndsWith('"')) -or
                    ($v.StartsWith("'") -and $v.EndsWith("'"))) {
                    $v = $v.Substring(1, $v.Length - 2)
                }
            }
            try { Set-Item -Path "Env:$k" -Value $v -ErrorAction Stop } catch {}
        }
    }
}
else {
    Write-Launch "WARN: primary .env not found at $EnvFile (creds/NTFY may be missing)"
}

# OPEN_RUN_ALLOW_CLOCK_UNKNOWN must never be armed implicitly by PRIMARY .env.
# The wrapper's explicit -AllowClockUnknown switch is the only supported path for
# human-supervised use. Scrub both inherited and .env-provided values before Python.
if (Test-Path Env:OPEN_RUN_ALLOW_CLOCK_UNKNOWN) {
    Write-Launch "safety: ignoring OPEN_RUN_ALLOW_CLOCK_UNKNOWN from environment; use -AllowClockUnknown explicitly"
    Remove-Item Env:OPEN_RUN_ALLOW_CLOCK_UNKNOWN -ErrorAction SilentlyContinue
}

# hard paper guard at the wrapper level too (belt and suspenders)
if ($env:ALPACA_PAPER -and ($env:ALPACA_PAPER.ToLower() -notin @("1", "true", "yes", "y", "on"))) {
    Write-Launch "SAFETY ABORT: ALPACA_PAPER is not truthy ($($env:ALPACA_PAPER)); refusing to run."
    exit 2
}

# --- force UTF-8 for the child python ------------------------------------
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

# --- one-time flatten-all reset marker -----------------------------------
# logs\RESET_ONCE.flag が在れば「次に成功したオープン run」で一度だけ全ポジションを
# flatten してからクリーン再エントリーする (PHASE 2 リセット)。
#   - dry-run では消費も強制もしない (テストが marker に干渉しない)。
#   - 注文完了 (exit 0 / 4) で marker を削除 = 観測劣化だけなら再 flatten しない。
#   - pre-trade abort (market closed 等 = exit 3) は marker 保持 -> 次の open で再試行。
$ResetMarker = Join-Path $LogDir "RESET_ONCE.flag"
$ConsumeResetMarker = $false
if ((Test-Path $ResetMarker) -and (-not $DryRun)) {
    Write-Launch "RESET_ONCE.flag 検出 -> この run は --flatten-all (一回限りリセット)"
    $FlattenAll = $true
    $ConsumeResetMarker = $true
}

# --- build python args ---------------------------------------------------
$py = Join-Path $WorktreeRoot "scripts\open_auto_run.py"
$pyArgs = @($py)
if ($Date) { $pyArgs += @("--date", $Date) }
$pyArgs += @("--min-signals", "$MinSignals", "--poll-timeout", "$PollTimeout")
$pyArgs += @("--primary-root", $PrimaryRoot)
if ($DryRun) { $pyArgs += "--dry-run" }
if ($AllowClosed) { $pyArgs += "--allow-closed" }
if ($AllowClockUnknown) { $pyArgs += "--allow-clock-unknown" }
if ($SkipSignals) { $pyArgs += "--skip-signals" }
if ($Force) { $pyArgs += "--force" }
if ($FlattenAll) { $pyArgs += "--flatten-all" }
if ($NoPublish) { $pyArgs += "--no-publish" }

# --- scheduler retry fence ------------------------------------------------
# Python writes DONE.lock only after notify/publish. A hard kill in that window can
# otherwise leave no DONE.lock and let the second nightly trigger re-enter orders.
# CreateNew is atomic: concurrent wrappers cannot both acquire this fence.
$TargetCompact = if ($Date -match '^\d{4}-\d{2}-\d{2}$') { $Date.Replace("-", "") } else { Get-Date -Format "yyyyMMdd" }
$RunLogDir = Join-Path $LogDir "open_run_$TargetCompact"
$InflightLock = Join-Path $RunLogDir "INFLIGHT.lock"
$DoneLock = Join-Path $RunLogDir "DONE.lock"

if (-not $DryRun) {
    if (-not (Test-Path $RunLogDir)) {
        New-Item -ItemType Directory -Path $RunLogDir -Force | Out-Null
    }

    # If Python completed far enough to write DONE but the wrapper was killed before
    # cleanup, the INFLIGHT marker is stale and safe to remove. -Force is already the
    # explicit operator override for rerunning a completed day, so it may clear stale
    # INFLIGHT as well.
    if ((Test-Path $InflightLock) -and ((Test-Path $DoneLock) -or $Force)) {
        try {
            Remove-Item $InflightLock -Force -ErrorAction Stop
            Write-Launch "cleared stale INFLIGHT.lock (DONE.lock present or -Force)"
        }
        catch {
            Write-Launch "SAFETY ABORT: cannot clear stale INFLIGHT.lock: $_"
            exit 5
        }
    }

    try {
        $lockStream = [System.IO.File]::Open(
            $InflightLock,
            [System.IO.FileMode]::CreateNew,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::None
        )
        $lockWriter = New-Object System.IO.StreamWriter($lockStream)
        $lockWriter.WriteLine("pid=$PID")
        $lockWriter.WriteLine("started_at=$([DateTimeOffset]::UtcNow.ToString('o'))")
        $lockWriter.Flush()
        $lockWriter.Dispose()
        Write-Launch "acquired INFLIGHT.lock: $InflightLock"
    }
    catch {
        Write-Launch "SAFETY ABORT: INFLIGHT.lock already exists or cannot be created: $InflightLock"
        Write-Launch "Inspect the previous run before retrying; use -Force only after confirming order state."
        exit 5
    }
}

Set-Location $WorktreeRoot
Write-Launch ("python " + ($pyArgs -join " "))

& python @pyArgs 2>&1 | ForEach-Object { Write-Launch $_ }
$code = $LASTEXITCODE
Write-Launch "=== open_auto_run.py exit=$code ==="

# Release the retry fence only when Python has given us a state that is safe to
# retry/continue automatically. exit 3 is explicitly pre-trade; 0/4 mean trade
# stages completed and DONE.lock should be durable. Unknown failures keep the fence.
if (-not $DryRun) {
    if (($code -eq 0) -or ($code -eq 3) -or ($code -eq 4)) {
        try {
            Remove-Item $InflightLock -Force -ErrorAction Stop
            Write-Launch "released INFLIGHT.lock after safe exit=$code"
        }
        catch {
            Write-Launch "WARN: INFLIGHT.lock cleanup failed after safe exit=$code: $_"
        }
    }
    else {
        Write-Launch "SAFETY: preserving INFLIGHT.lock after unexpected exit=$code; manual order-state check required"
    }
}

# 注文が完了した flatten-all run の後に marker を消費 (再発防止 = 一回限り)。
# exit 4 は notify/publish の劣化だけで trades + DONE は完了済みなので消費対象。
if ($ConsumeResetMarker) {
    if (($code -eq 0) -or ($code -eq 4)) {
        try {
            Remove-Item $ResetMarker -Force -ErrorAction Stop
            Write-Launch "RESET_ONCE.flag を消費 (リセット完了、以降は通常 open run)"
        }
        catch { Write-Launch "WARN: RESET_ONCE.flag 削除失敗 (次回も flatten の恐れ): $_" }
    }
    else {
        Write-Launch "リセット run が exit=$code (非成功) -> marker 保持 (次の open で再試行)"
    }
}

exit $code
