<#
.SYNOPSIS
    Safety sweep for positions created by the current US trading session.
.DESCRIPTION
    Runs paper_exit_check in the non-destructive post-entry protection scope.
    It never executes time/target/breakout exits and never performs cancel+replace.
    Designed for repeated Task Scheduler runs so entries filled after open_auto_run
    reconciliation still receive resident protection.
#>
param(
    [switch]$DryRun = $false,
    [string]$PrimaryRoot = "C:\Repos\quant_trading_system_0510to0906"
)
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorktreeRoot = Split-Path -Parent $ScriptDir
$EnvFile = Join-Path $PrimaryRoot ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^\s*#') { return }
        if ($_ -match '^\s*([^#=\s]+)\s*=\s*(.*)$') {
            $k = $matches[1].Trim(); $v = $matches[2].Trim()
            if ($v.Length -ge 2 -and (($v.StartsWith('"') -and $v.EndsWith('"')) -or ($v.StartsWith("'") -and $v.EndsWith("'")))) {
                $v = $v.Substring(1, $v.Length - 2)
            }
            Set-Item -Path "Env:$k" -Value $v
        }
    }
}
$paper = [string]$env:ALPACA_PAPER
if ($paper -and ($paper.ToLower() -notin @("1","true","yes","y","on"))) {
    throw "SAFETY STOP: ALPACA_PAPER is not enabled"
}
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

# The US session crosses JST midnight. 00:00-11:59 JST still belongs to the
# run date that started on the previous JST evening.
$now = Get-Date
$tradeDate = if ($now.Hour -lt 12) { $now.AddDays(-1) } else { $now }
$dateText = $tradeDate.ToString("yyyy-MM-dd")
$compact = $tradeDate.ToString("yyyyMMdd")
$py = Join-Path $WorktreeRoot "scripts\paper_exit_check.py"
$out = Join-Path $WorktreeRoot "results_csv\exit_orders_${compact}_post_entry_sweep.json"
$args = @($py, "--date", $dateText, "--output-json", $out, "--today-entry-protection-only")
if (-not $DryRun) { $args += @("--confirm", "--yes") }

Push-Location $WorktreeRoot
try {
    & python @args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
