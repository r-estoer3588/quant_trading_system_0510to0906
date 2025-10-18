param(
    [int]$max = 3
)
$i = 0
while ($i -lt $max) {
    git add -A
    git commit -m 'chore: apply pre-commit automatic fixes' 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "commit succeeded"
        exit 0
    } else {
        Write-Host "commit failed, retrying..."
        Start-Sleep -Seconds 1
        $i = $i + 1
    }
}
Write-Host "Reached retry limit"
exit 1
