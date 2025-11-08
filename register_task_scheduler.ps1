# ============================================================================
# Windowsタスクスケジューラー登録スクリプト
#
# 説明:
#   このスクリプトはWindowsタスクスケジューラーにスケジューラーを自動起動するタスクを登録します
#
# 使い方:
#   .\register_task_scheduler.ps1
#
# 動作:
#   - タスク名: "QuantTradingScheduler"
#   - トリガー: ログイン時に自動起動
#   - 実行: start_scheduler.ps1 を実行
#
# 注意:
#   - 管理者権限で実行する必要があります
#   - 登録後は再起動時に自動的にスケジューラーが起動します
# ============================================================================

param(
    [switch]$Unregister = $false
)

$ErrorActionPreference = "Stop"

# 管理者権限チェック
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "❌ このスクリプトは管理者権限で実行する必要があります" -ForegroundColor Red
    Write-Host ""
    Write-Host "以下のいずれかの方法で実行してください:" -ForegroundColor Yellow
    Write-Host "  1. PowerShellを管理者として実行してから、このスクリプトを実行" -ForegroundColor Gray
    Write-Host "  2. または、以下のコマンドを実行:" -ForegroundColor Gray
    Write-Host "     Start-Process powershell -Verb RunAs -ArgumentList '-File', '$($MyInvocation.MyCommand.Path)'" -ForegroundColor Cyan
    exit 1
}

# プロジェクトルート
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ScriptPath = Join-Path $ProjectRoot "start_scheduler.ps1"

# タスク名
$TaskName = "QuantTradingScheduler"

# タスクの削除
if ($Unregister) {
    Write-Host "🗑️  タスク '$TaskName' を削除します..." -ForegroundColor Yellow
    try {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
        Write-Host "✅ タスクを削除しました" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  タスクが見つかりませんでした" -ForegroundColor Yellow
    }
    exit 0
}

# 既存タスクの確認と削除
$existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Write-Host "⚠️  既存のタスク '$TaskName' が見つかりました" -ForegroundColor Yellow
    Write-Host "   既存タスクを削除して再登録します..." -ForegroundColor Gray
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Write-Host "📝 タスクスケジューラーにタスクを登録します..." -ForegroundColor Cyan
Write-Host ""
Write-Host "タスク名: $TaskName" -ForegroundColor Gray
Write-Host "スクリプト: $ScriptPath" -ForegroundColor Gray
Write-Host ""

# タスクアクション: PowerShellスクリプトを実行
$Action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$ScriptPath`"" `
    -WorkingDirectory $ProjectRoot

# トリガー: ユーザーログイン時
$Trigger = New-ScheduledTaskTrigger -AtLogOn

# 設定: バックグラウンドで実行、バッテリーでも動作
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Days 365) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5)

# タスク登録（現在のユーザーで実行）
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

# タスクを登録
Register-ScheduledTask -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "量的トレーディングシステムのスケジューラー（当日シグナル生成など）" | Out-Null

Write-Host "✅ タスクを登録しました！" -ForegroundColor Green
Write-Host ""
Write-Host "📋 次回のログイン時から自動的にスケジューラーが起動します" -ForegroundColor Cyan
Write-Host ""
Write-Host "今すぐ起動する場合:" -ForegroundColor Yellow
Write-Host "  .\start_scheduler.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "タスクの確認:" -ForegroundColor Yellow
Write-Host "  1. タスクスケジューラーを開く" -ForegroundColor Gray
Write-Host "  2. タスクスケジューラライブラリで '$TaskName' を検索" -ForegroundColor Gray
Write-Host ""
Write-Host "タスクの削除:" -ForegroundColor Yellow
Write-Host "  .\register_task_scheduler.ps1 -Unregister" -ForegroundColor Cyan
Write-Host ""
