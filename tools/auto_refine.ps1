param(
    [int]$MaxIterations = 10,
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

Write-Host "== Semi-automatic pixel diff loop ==" -ForegroundColor Cyan
Write-Host "  Python: $Python" -ForegroundColor Gray
Write-Host "  Max iterations: $MaxIterations" -ForegroundColor Gray
Write-Host "  Snapshot sources: $($SnapshotSources -join ', ')" -ForegroundColor Gray
Write-Host "  Image dir: $SrcDir" -ForegroundColor Gray

$arguments = @("tools/auto_refine_loop.py", "--max-iterations", "$MaxIterations", "--src-dir", "$SrcDir", "--python", "$Python")
foreach ($src in $SnapshotSources) {
    $arguments += @("--snapshot-source", "$src")
}

& $Python $arguments

if ($LASTEXITCODE -eq 0) {
    Write-Host "Loop finished." -ForegroundColor Green
}
else {
    Write-Host "Loop exited with code $LASTEXITCODE." -ForegroundColor Yellow
}
