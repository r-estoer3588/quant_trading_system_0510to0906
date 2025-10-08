# ===================================================================
# Task Scheduler Setup - Run as Administrator
# ===================================================================
#
# This script registers a daily task to automatically update cache
# and perform health checks every morning at 06:00.
#
# How to run:
#   1. Open PowerShell as Administrator
#   2. Run: cd C:\Repos\quant_trading_system
#   3. Run: .\scripts\setup_daily_scheduler_admin.ps1
#
# ===================================================================

$TaskName = "QuantTradingSystem_DailyUpdate"
$Description = "Quant Trading System - Daily cache update with health check"
$PythonExe = "C:\Repos\quant_trading_system\venv\Scripts\python.exe"
$ScriptPath = "C:\Repos\quant_trading_system\scripts\scheduler_update_with_healthcheck.py"
$WorkingDir = "C:\Repos\quant_trading_system"

Write-Host "=== Quant Trading System - Daily Scheduler Setup ===" -ForegroundColor Cyan
Write-Host ""

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "ERROR: This script requires administrator privileges." -ForegroundColor Red
    Write-Host "Please run PowerShell as Administrator and try again." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Steps:" -ForegroundColor Yellow
    Write-Host "  1. Right-click on PowerShell" -ForegroundColor White
    Write-Host "  2. Select 'Run as Administrator'" -ForegroundColor White
    Write-Host "  3. Navigate to: cd C:\Repos\quant_trading_system" -ForegroundColor White
    Write-Host "  4. Run: .\scripts\setup_daily_scheduler_admin.ps1" -ForegroundColor White
    exit 1
}

# Verify Python executable exists
if (-not (Test-Path $PythonExe)) {
    Write-Host "ERROR: Python executable not found at: $PythonExe" -ForegroundColor Red
    Write-Host "Please verify the virtual environment is set up correctly." -ForegroundColor Yellow
    exit 1
}

# Verify script exists
$FullScriptPath = Join-Path $WorkingDir "scripts\scheduler_update_with_healthcheck.py"
if (-not (Test-Path $FullScriptPath)) {
    Write-Host "ERROR: Script not found at: $FullScriptPath" -ForegroundColor Red
    Write-Host "Please verify the file exists." -ForegroundColor Yellow
    exit 1
}

# Remove existing task if present
$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($ExistingTask) {
    Write-Host "Removing existing task: $TaskName" -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Existing task removed." -ForegroundColor Green
}

# Create action
$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "scripts\scheduler_update_with_healthcheck.py" `
    -WorkingDirectory $WorkingDir

# Create trigger (daily at 06:00)
$Trigger = New-ScheduledTaskTrigger -Daily -At "06:00"

# Create settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 15)

# Create principal (run whether user is logged on or not, with highest privileges)
$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType S4U `
    -RunLevel Highest

# Register task
Write-Host "Creating scheduled task: $TaskName" -ForegroundColor Cyan
try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Description $Description `
        -ErrorAction Stop | Out-Null

    Write-Host "SUCCESS: Task created successfully!" -ForegroundColor Green
}
catch {
    Write-Host "ERROR: Failed to create task: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== Task Configuration ===" -ForegroundColor Cyan
Write-Host "  Task Name       : $TaskName" -ForegroundColor White
Write-Host "  Schedule        : Daily at 06:00" -ForegroundColor White
Write-Host "  Python Exe      : $PythonExe" -ForegroundColor White
Write-Host "  Script          : $ScriptPath" -ForegroundColor White
Write-Host "  Working Dir     : $WorkingDir" -ForegroundColor White
Write-Host "  Run Level       : Highest (Administrator)" -ForegroundColor White
Write-Host "  Retry on Failure: Yes (2 times, 15-minute intervals)" -ForegroundColor White
Write-Host ""

Write-Host "=== Next Steps ===" -ForegroundColor Cyan
Write-Host "1. Verify task in Task Scheduler:" -ForegroundColor Yellow
Write-Host "   Get-ScheduledTask -TaskName '$TaskName'" -ForegroundColor White
Write-Host ""
Write-Host "2. Test manual execution:" -ForegroundColor Yellow
Write-Host "   Start-ScheduledTask -TaskName '$TaskName'" -ForegroundColor White
Write-Host ""
Write-Host "3. Check health log:" -ForegroundColor Yellow
Write-Host "   Get-Content C:\Repos\quant_trading_system\logs\scheduler_update_health_`$(Get-Date -Format yyyyMMdd).log -Tail 5" -ForegroundColor White
Write-Host ""
Write-Host "4. Review documentation:" -ForegroundColor Yellow
Write-Host "   docs\operations\daily_scheduler_setup.md" -ForegroundColor White
Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
