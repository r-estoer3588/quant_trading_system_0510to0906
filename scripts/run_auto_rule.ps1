# PowerShell wrapper to run the Python auto-rule script in the repository venv
# Usage (Task Scheduler): set action to run PowerShell with this script path as argument.
# Example Task Scheduler action: Program/script: powershell.exe
# Arguments: -ExecutionPolicy Bypass -File "C:\Repos\quant_trading_system\scripts\run_auto_rule.ps1"

param(
    [switch]$Paper,
    [switch]$DryRun
)

$Repo = Split-Path -Parent $MyInvocation.MyCommand.Definition
# Activate venv if exists (assumes venv in .venv)
$Venv = Join-Path $Repo ".venv\Scripts\Activate.ps1"
if (Test-Path $Venv) {
    Write-Host "Activating virtualenv $Venv"
    & $Venv
}
else {
    Write-Host "No virtualenv activate script found at $Venv - running system Python"
}

$python = Get-Command python -ErrorAction Stop | Select-Object -ExpandProperty Source
$script = Join-Path $Repo "scripts\run_auto_rule.py"
$pythonArgsList = [System.Collections.ArrayList]@()
if ($Paper) { $null = $pythonArgsList.Add('--paper') }
if ($DryRun) { $null = $pythonArgsList.Add('--dry-run') }

$argStr = $pythonArgsList -join ' '
Write-Host "Running: $python $script $argStr"
& $python $script $pythonArgsList
$exitCode = $LASTEXITCODE
Write-Host "Exit code: $exitCode"
exit $exitCode
