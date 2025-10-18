param(
    [int]$maxAttempts = 4
)
$i = 0
while ($i -lt $maxAttempts) {
    Write-Host "pre-commit attempt $($i + 1)"
    & .\venv\Scripts\python.exe -X utf8 -m pre_commit run --all-files
    $code = $LASTEXITCODE
    if ($code -eq 0) {
        Write-Host "pre-commit stable"
        break
    }
    else {
        Write-Host "pre-commit modified files; staging and retrying..."
        git add -A
        $i = $i + 1
        Start-Sleep -Seconds 1
    }
}
if ($LASTEXITCODE -ne 0) {
    Write-Host "pre-commit did not stabilize after $maxAttempts attempts"
    exit 1
}
else {
    Write-Host "staging final changes and committing"
    git add -A
    git commit -m "style: sync formatting with updated pre-commit hooks"
    git push origin branch0906
}
