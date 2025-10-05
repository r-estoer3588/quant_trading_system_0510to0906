Param(
    [string]$Pattern = "not slow and not integration"
)

$ErrorActionPreference = "Stop"

& "$PSScriptRoot/../venv/Scripts/python.exe" -m pytest -q --tb=short -k "$Pattern"
exit $LASTEXITCODE
