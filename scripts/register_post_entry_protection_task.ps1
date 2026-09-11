<#
.SYNOPSIS
    Register the recurring Paper post-entry protection safety sweep.
.DESCRIPTION
    Creates QuantTrading_PostEntryProtection at 22:35 JST daily, repeating every
    15 minutes for 8 hours. The sweep is deliberately non-destructive: it can arm
    missing resident protection for same-session entries but cannot submit exits
    or cancel/replace existing protection.
#>
param(
    [string]$WorktreeRoot = "C:\tmp\qts-main-clean",
    [string]$PrimaryRoot = "C:\Repos\quant_trading_system_0510to0906",
    [switch]$Unregister = $false
)
$ErrorActionPreference = "Stop"
$TaskName = "QuantTrading_PostEntryProtection"
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "removed: $TaskName"
}
if ($Unregister) { return }
$launcher = Join-Path $WorktreeRoot "scripts\post_entry_protection_sweep.ps1"
if (-not (Test-Path $launcher)) { throw "launcher not found: $launcher" }
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$launcher`" -PrimaryRoot `"$PrimaryRoot`"" `
    -WorkingDirectory $WorktreeRoot
$trigger = New-ScheduledTaskTrigger -Daily -At "22:35"
$trigger.Repetition.Interval = "PT15M"
$trigger.Repetition.Duration = "PT8H"
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings `
    -Description "Paper post-entry protection sweep; non-destructive scope." | Out-Null
Write-Host "registered: $TaskName at 22:35 JST, every 15m for 8h"
Write-Host "dry-run smoke test: powershell -File `"$launcher`" -DryRun -PrimaryRoot `"$PrimaryRoot`""
