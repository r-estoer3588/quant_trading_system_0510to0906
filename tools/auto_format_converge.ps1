param(
    [int]$MaxIter = 6
)

$py = "C:\Repos\quant_trading_system\venv\Scripts\python.exe"
$i = 0

while ($i -lt $MaxIter) {
    Write-Host "Iteration $($i + 1) / $MaxIter"
    & $py -m black .
    if ($LASTEXITCODE -ne 0) { Write-Host "black exited with code $LASTEXITCODE" }

    & $py -m isort .
    if ($LASTEXITCODE -ne 0) { Write-Host "isort exited with code $LASTEXITCODE" }

    & $py -m ruff check --fix .
    if ($LASTEXITCODE -ne 0) { Write-Host "ruff --fix exited with code $LASTEXITCODE" }

    git add -A
    $status = git status --porcelain
    if (-not $status) {
        Write-Host "No unstaged changes detected — formatting converged. Creating commit..."
        git commit -m "style: apply auto-fixes (black/isort/ruff)"
        exit 0
    }

    Write-Host "Unstaged changes remain after iteration $($i + 1). Will run another iteration."
    $i++
}

Write-Host "Reached max iterations ($MaxIter). Formatting did not fully converge."
git status --porcelain
exit 1
