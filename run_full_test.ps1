# フル実行スクリプト（PowerShell版）

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "フル実行（6000+銘柄）開始" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 環境変数設定
$env:ENABLE_PROGRESS_EVENTS = "1"
$env:EXPORT_DIAGNOSTICS_SNAPSHOT_ALWAYS = "1"

Write-Host "環境変数設定:" -ForegroundColor Yellow
Write-Host "  ENABLE_PROGRESS_EVENTS=$env:ENABLE_PROGRESS_EVENTS"
Write-Host "  EXPORT_DIAGNOSTICS_SNAPSHOT_ALWAYS=$env:EXPORT_DIAGNOSTICS_SNAPSHOT_ALWAYS"
Write-Host ""

# 仮想環境のPythonパス
$pythonPath = "C:\Repos\quant_trading_system\venv\Scripts\python.exe"

Write-Host "Pythonスクリプト実行中..." -ForegroundColor Green
Write-Host ""

# スクリプト実行
& $pythonPath scripts/run_all_systems_today.py --parallel --save-csv

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "実行完了（終了コード: $LASTEXITCODE）" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
