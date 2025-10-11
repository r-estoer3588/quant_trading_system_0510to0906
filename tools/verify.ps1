param(
    [string[]]$SnapshotSources = @("results_csv", "results_csv_test", "logs", "results_images"),
    [string]$SrcDir = "results_images",
    [string]$Python = ""
)

if (-not $Python) {
    $Python = [System.Environment]::GetEnvironmentVariable("PYTHON_EXE")
    if (-not $Python) {
        $Python = "python"
    }
}

Write-Host "==> pytest" -ForegroundColor Cyan
& $Python -m pytest -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "Tests failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "==> snapshot" -ForegroundColor Cyan
$arguments = @("tools/snapshot.py")
foreach ($src in $SnapshotSources) {
    $arguments += @("--source", "$src")
}
& $Python $arguments
if ($LASTEXITCODE -ne 0) {
    Write-Host "Snapshot step failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "==> imgdiff" -ForegroundColor Cyan
& $Python tools/imgdiff.py --src-dir "$SrcDir"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Image diff step reported issues." -ForegroundColor Yellow
}
else {
    Write-Host "Validation loop finished." -ForegroundColor Green
}
