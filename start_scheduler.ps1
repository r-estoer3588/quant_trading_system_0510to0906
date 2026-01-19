# ============================================================================
# ❌ DEPRECATED: スケジューラー起動スクリプト（2026-01-19 無効化）
#
# 説明:
#   ⚠️  EODHD API契約終了に伴い、本スクリプトは無効化されました
#   詳細は CHANGELOG.md を参照してください
#
# 現在のステータス:
#   - Windows タスクスケジューラーのタスク: すべて Disabled
#   - 日次更新・定期シグナル生成: 停止状態
#
# 復旧方法（新たなデータ供給元導入時）:
#   1. config/ で新API設定を構成
#   2. scripts/cache_daily_data.py を新API対応に修正
#   3. register_task_scheduler.ps1 を再実行
# ============================================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "❌ エラー: このスクリプトは使用できません" -ForegroundColor Red
Write-Host ""
Write-Host "EODHD API契約終了（2026-01-19）に伴い、スケジューラーは無効化されました。" -ForegroundColor Yellow
Write-Host ""
Write-Host "詳細:"
Write-Host "  - CHANGELOG.md で操作履歴を確認" -ForegroundColor Gray
Write-Host "  - Windows タスク状態確認:" -ForegroundColor Gray
Write-Host "    Get-ScheduledTask -TaskName 'QuantTrading*' | Select-Object TaskName, State" -ForegroundColor Cyan
Write-Host ""
Write-Host "新たなデータ供給元導入後に復旧してください。" -ForegroundColor Yellow
Write-Host ""
exit 1

# プロジェクトルートディレクトリに移動
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "📅 スケジューラーを起動します..." -ForegroundColor Cyan
Write-Host "プロジェクトルート: $ProjectRoot" -ForegroundColor Gray
Write-Host ""

# .env ファイルの存在確認
if (-not (Test-Path ".env")) {
    Write-Host "⚠️  警告: .env ファイルが見つかりません" -ForegroundColor Yellow
    Write-Host "   .env.example をコピーして .env を作成し、必要な環境変数を設定してください" -ForegroundColor Yellow
    Write-Host ""
}

# Python仮想環境のアクティベート（存在する場合）
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "🐍 Python仮想環境をアクティベートします..." -ForegroundColor Green
    & ".\venv\Scripts\Activate.ps1"
} elseif (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "🐍 Python仮想環境をアクティベートします..." -ForegroundColor Green
    & ".\.venv\Scripts\Activate.ps1"
}

Write-Host "📋 設定されているスケジュール:" -ForegroundColor Cyan
Write-Host "  • update_tickers        : 平日 06:00 - ティッカーリスト更新" -ForegroundColor Gray
Write-Host "  • bulk_last_day         : 平日 06:45 - 前営業日データ一括更新" -ForegroundColor Gray
Write-Host "  • warm_cache            : 平日 07:00 - キャッシュウォームアップ" -ForegroundColor Gray
Write-Host "  • precompute_indicators : 平日 07:30 - 共有指標の事前計算" -ForegroundColor Gray
Write-Host "  • send_signals          : 平日 08:30 - シグナル通知送信" -ForegroundColor Gray
Write-Host "  • update_trailing_stops : 平日 08:45 - トレーリングストップ更新" -ForegroundColor Gray
Write-Host "  • notify_metrics        : 平日 08:50 - メトリクス通知" -ForegroundColor Gray
Write-Host "  • build_metrics_report  : 平日 08:55 - レポート生成" -ForegroundColor Gray
Write-Host "  • run_today_signals     : 平日 11:00 - 当日シグナル生成 ⭐" -ForegroundColor Yellow
Write-Host "  • daily_run             : 火-土 06:15 - 日次バッチ処理" -ForegroundColor Gray
Write-Host ""
Write-Host "⏰ スケジューラーをバックグラウンドで起動しています..." -ForegroundColor Green
Write-Host ""

# スケジューラーをバックグラウンドプロセスで起動（PowerShellウィンドウは自動閉じ）
$SchedulerProcess = Start-Process `
    -FilePath "python" `
    -ArgumentList "-m", "schedulers.runner" `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -PassThru

Write-Host "✅ スケジューラーが起動しました" -ForegroundColor Green
Write-Host "   プロセスID: $($SchedulerProcess.Id)" -ForegroundColor Gray
Write-Host "   ログファイル: .\logs\app.log" -ForegroundColor Gray
Write-Host ""
Write-Host "ℹ️  スケジューラーは以下の場所でバックグラウンド稼働します:" -ForegroundColor Cyan
Write-Host "   - Windows タスクスケジューラ (起動時自動化)" -ForegroundColor Gray
Write-Host "   - ローカルプロセス (現在のセッション)" -ForegroundColor Gray
Write-Host ""
Write-Host "停止方法:" -ForegroundColor Yellow
Write-Host "   PowerShell: Get-Process python | Where-Object {$_.CommandLine -like '*schedulers.runner*'} | Stop-Process" -ForegroundColor Gray
Write-Host "   または、タスクマネージャーで該当の python.exe を終了" -ForegroundColor Gray

# スクリプト終了（PowerShellウィンドウは3秒後に自動閉じ）
Start-Sleep -Seconds 3
exit 0
