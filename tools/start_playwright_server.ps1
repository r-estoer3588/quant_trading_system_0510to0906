# Start Playwright server wrapper
# This script runs the full data pipeline first, then starts Streamlit.

$RepoRoot = "c:\Repos\quant_trading_system"
$Python = Join-Path $RepoRoot "venv\Scripts\python.exe"

Write-Host "[$(Get-Date -Format o)] Running full pipeline: run_all_systems_today.py (this may take several minutes)..."
# Ensure a unique run namespace is set for CI/test runs to avoid output collisions.
if (-not $env:RUN_NAMESPACE -or [string]::IsNullOrWhiteSpace($env:RUN_NAMESPACE)) {
    $guid = [guid]::NewGuid().ToString('N').Substring(0, 8)
    $env:RUN_NAMESPACE = "ci_$guid"
    Write-Host "[INFO] Generated RUN_NAMESPACE=$env:RUN_NAMESPACE"
}
else {
    Write-Host "[INFO] Using RUN_NAMESPACE=$env:RUN_NAMESPACE"
}

# By default use per-run subdirectory for outputs and enable run-lock to avoid
# concurrent writes when CI or local tests may start multiple runs.
$env:PIPELINE_USE_RUN_SUBDIR = $env:PIPELINE_USE_RUN_SUBDIR -or "1"
$env:PIPELINE_USE_RUN_LOCK = $env:PIPELINE_USE_RUN_LOCK -or "1"

& $Python (Join-Path $RepoRoot "scripts\run_all_systems_today.py") --parallel --save-csv
$rc = $LASTEXITCODE
if ($rc -ne 0) {
    Write-Error "run_all_systems_today.py failed with exit code $rc. Aborting web server startup."
    exit $rc
}

Write-Host "[$(Get-Date -Format o)] Pipeline completed successfully. Starting Streamlit..."
# Start Streamlit in the foreground so Playwright can detect the server
& $Python -m streamlit run (Join-Path $RepoRoot "apps\app_integrated.py") --server.headless true

# When Streamlit exits, script ends
exit $LASTEXITCODE
