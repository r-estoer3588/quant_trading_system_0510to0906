<#
.SYNOPSIS
  Register a Windows Scheduled Task to run the recompute wrapper at 10:30 JST weekdays.

.DESCRIPTION
  Creates or updates a Scheduled Task named QuantTradingRecomputeRolling that executes
  scripts\run_recompute_daily.ps1 in dry-run mode at 10:30 JST on weekdays. If the task
  already exists, it will be replaced.

.PARAMETER Time
  Time string in HH:MM (JST). Default: 10:30

.PARAMETER TaskName
  Task name. Default: QuantTradingRecomputeRolling

.PARAMETER Unregister
  If supplied, unregisters the task instead of creating it.
#>
param(
    [string]$Time = '10:30',
    [string]$TaskName = 'QuantTradingRecomputeRolling',
    [switch]$Unregister,
    [int]$Workers = 2
)

$ErrorActionPreference = 'Stop'
$ScriptPath = if ($PSCommandPath) { $PSCommandPath } else { $MyInvocation.MyCommand.Path }
$ScriptDir = Split-Path -Parent $ScriptPath
$ProjectRoot = Split-Path -Parent $ScriptDir
$RecomputeScript = Join-Path $ProjectRoot 'scripts\run_recompute_daily.ps1'

if ($Unregister) {
    $ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($ExistingTask) { Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false; Write-Host "✅ Unregistered task: $TaskName" } else { Write-Host "ℹ️ Task not found: $TaskName" }
    exit 0
}

if (-not (Test-Path $RecomputeScript)) { Write-Host "Recompute script not found: $RecomputeScript"; exit 1 }

# Remove existing task if present
$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($ExistingTask) {
    Write-Host "⚠️ Existing task found; replacing: $TaskName"
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$actionArgs = '-NoProfile -ExecutionPolicy Bypass -File "' + $RecomputeScript + '" -DryRun -Workers ' + $Workers
$Action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $actionArgs
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $Time
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "Daily recompute (dry-run) for rolling cache diagnostics" -User $env:USERNAME -RunLevel Limited

Write-Host "✅ Registered recompute task: $TaskName at $Time JST (weekdays)"
Write-Host "Use: Start-ScheduledTask -TaskName '$TaskName' to run immediately."
