# UTF-8エンコーディングを保持してテスト実行するヘルパースクリプト
# 使い方: .\tools\run_test_utf8.ps1 [フィルタパターン]
# 例: .\tools\run_test_utf8.ps1 "TEST-MODE|鮮度許容|stale_over"

param(
    [string]$Pattern = "TEST-MODE|鮮度許容|stale_over",
    [string]$OutputFile = "temp_test_output_utf8.txt"
)

# コンソールとファイル出力をUTF-8に設定
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "🔧 UTF-8モードでテスト実行中..." -ForegroundColor Cyan
Write-Host "   パターン: $Pattern" -ForegroundColor Gray

# テスト実行して出力をキャプチャ
& venv\Scripts\python.exe scripts\run_all_systems_today.py --test-mode mini --skip-external --benchmark 2>&1 | Out-File -FilePath $OutputFile -Encoding utf8

# UTF-8でファイルを読み込んでフィルタリング（PowerShellパイプを避ける）
$content = Get-Content $OutputFile -Encoding utf8
$filteredLines = $content | Where-Object { $_ -match $Pattern }

Write-Host "`n✅ マッチした行 ($($filteredLines.Count)件):" -ForegroundColor Green
$filteredLines | ForEach-Object { Write-Host $_ }

Write-Host "`n📄 完全な出力: $OutputFile" -ForegroundColor Yellow
