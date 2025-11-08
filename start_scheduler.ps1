# ============================================================================
# スケジューラー起動スクリプト
#
# 説明:
#   このスクリプトは当日シグナル生成を含む全ての定期タスクを実行するスケジューラーを起動します
#
# 使い方:
#   .\start_scheduler.ps1
#
# スケジュール設定:
#   - config/config.yaml の scheduler セクションで設定されたタスクを自動実行
#   - run_today_signals: 平日 08:15 (JST)
#   - 他のタスク: ティッカー更新、キャッシュ更新、通知など
#
# 停止方法:
#   - Ctrl+C で停止
#   - または、タスクマネージャーでPythonプロセスを終了
# ============================================================================

$ErrorActionPreference = "Stop"

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
Write-Host "⏰ スケジューラー実行中... (Ctrl+C で停止)" -ForegroundColor Green
Write-Host ""

# スケジューラーを起動
python -m schedulers.runner
