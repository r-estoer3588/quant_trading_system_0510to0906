param(
    [int]$MaxIter = 8
)

$py = Join-Path $PSScriptRoot "..\venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    $py = "venv\Scripts\python.exe"
}

for ($i = 1; $i -le $MaxIter; $i++) {
    Write-Host "[Converge] Iteration $i / $MaxIter"

    & $py -m isort .
    if ($LASTEXITCODE -ne 0) { Write-Host "isort exited with $LASTEXITCODE" }

    & $py -m ruff check --fix .
    if ($LASTEXITCODE -ne 0) { Write-Host "ruff --fix exited with $LASTEXITCODE" }

    & $py -m black .
    if ($LASTEXITCODE -ne 0) { Write-Host "black exited with $LASTEXITCODE" }

    & $py -m ruff check --fix .
    if ($LASTEXITCODE -ne 0) { Write-Host "ruff --fix (2) exited with $LASTEXITCODE" }

    git add -A

    Write-Host "Attempting commit (will run pre-commit hooks)..."
    git commit -m "style: converge formatting (isort/ruff/black) - iteration $i"
    $commitCode = $LASTEXITCODE
    if ($commitCode -eq 0) {
        Write-Host "Commit succeeded on iteration $i"
        exit 0
    }

    Write-Host "Commit failed (exit $commitCode). Showing status and diff summary..."
    git status --porcelain
    git --no-pager diff --name-only | Select-Object -First 200 | ForEach-Object { Write-Host "  $_" }

    Write-Host "Waiting 1s before next iteration..."
    Start-Sleep -Seconds 1
}

Write-Host "Reached max iterations ($MaxIter) without a successful commit. Exiting with code 1."
git status --porcelain
exit 1
