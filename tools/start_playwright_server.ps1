# Start Playwright server wrapper
# This script runs the full data pipeline first, then starts Streamlit.

$RepoRoot = "c:\Repos\quant_trading_system"
$Python = Join-Path $RepoRoot "venv\Scripts\python.exe"

Write-Host "[$(Get-Date -Format o)] Running full pipeline: run_all_systems_today.py (this may take several minutes)..."
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
