Param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Targets
)

# UTF-8 強制環境で mypy を実行する PowerShell ラッパー
# 使用例:
#   pwsh ./scripts/run_mypy_utf8.ps1 core/system2.py core/system3.py

if (-not $Targets -or $Targets.Length -eq 0) {
    Write-Host "Usage: pwsh ./scripts/run_mypy_utf8.ps1 <files or options>" -ForegroundColor Yellow
    exit 1
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

# Python 実行ファイル解決
$python = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $python) {
    Write-Host "python executable not found in PATH" -ForegroundColor Red
    exit 2
}

Write-Host "[run_mypy_utf8] python=$($python.Source)" -ForegroundColor Cyan
Write-Host "[run_mypy_utf8] Targets: $($Targets -join ' ')" -ForegroundColor Cyan

& $python.Source tools/mypy_utf8_runner.py @Targets
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Host "[run_mypy_utf8] Success" -ForegroundColor Green
}
else {
    Write-Host "[run_mypy_utf8] ExitCode=$exitCode" -ForegroundColor Red
}

exit $exitCode
