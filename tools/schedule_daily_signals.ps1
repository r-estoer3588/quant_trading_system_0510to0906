<#
.SYNOPSIS
    Windows タスクスケジューラに日次シグナル実行タスクを登録

.DESCRIPTION
    PowerShell を使用して、Windows タスクスケジューラに自動実行タスクを登録します。
    デフォルトでは 16:30（US マーケット終了後）に実行されます。

.PARAMETER Time
    実行時刻（24時間形式: HH:MM）

.PARAMETER TaskName
    タスク名（デフォルト: "QuantTradingDailySignals"）

.PARAMETER Unregister
    タスクを削除

.EXAMPLE
    .\tools\schedule_daily_signals.ps1
    .\tools\schedule_daily_signals.ps1 -Time "17:00"
    .\tools\schedule_daily_signals.ps1 -Unregister
#>

param(
    [string]$Time = "10:00",  # 日本時間 10:00 (米国時間 19:00 ET 頃、データ更新後)
    [string]$TaskName = "QuantTradingDailySignals",
    [switch]$Unregister = $false
)

$ErrorActionPreference = "Stop"

# プロジェクトルートを取得（スクリプトレベルで実行）
$ScriptPath = if ($PSCommandPath) { $PSCommandPath } else { $MyInvocation.MyCommand.Path }
$ScriptDir = Split-Path -Parent $ScriptPath
$ProjectRoot = Split-Path -Parent $ScriptDir

function Register-DailyTask {
    param(
        [string]$Name,
        [string]$ExecutionTime
    )

    $DailyScript = Join-Path $ProjectRoot "scripts\daily_auto_run.ps1"

    if (-not (Test-Path $DailyScript)) {
        throw "実行スクリプトが見つかりません: $DailyScript"
    }

    Write-Host "========================================="
    Write-Host "タスクスケジューラ登録"
    Write-Host "========================================="
    Write-Host "タスク名: $Name"
    Write-Host "実行時刻: $ExecutionTime (日本時間)"
    Write-Host "スクリプト: $DailyScript"
    Write-Host ""
    Write-Host "⚠️  重要: EODHD データは米国市場終了後 2-3時間で更新されます"
    Write-Host "    推奨時刻: 09:00-10:00 JST (18:00-19:00 ET)"
    Write-Host ""

    # 既存タスクを削除
    $ExistingTask = Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue

    if ($ExistingTask) {
        Write-Host "⚠️  既存のタスクを削除します..."
        Unregister-ScheduledTask -TaskName $Name -Confirm:$false
    }

    # タスクアクション
    $Action = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$DailyScript`""

    # トリガー（平日のみ）
    $Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $ExecutionTime

    # 設定
    $Settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RunOnlyIfNetworkAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

    # タスク登録
    Register-ScheduledTask `
        -TaskName $Name `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Description "Quant Trading System の日次シグナル生成を自動実行" `
        -User $env:USERNAME `
        -RunLevel Limited

    Write-Host ""
    Write-Host "✅ タスクを登録しました"
    Write-Host ""
    Write-Host "確認方法:"
    Write-Host "  Get-ScheduledTask -TaskName '$Name'"
    Write-Host ""
    Write-Host "手動実行:"
    Write-Host "  Start-ScheduledTask -TaskName '$Name'"
    Write-Host ""
    Write-Host "タスク削除:"
    Write-Host "  .\tools\schedule_daily_signals.ps1 -Unregister"
    Write-Host "========================================="
}

function Unregister-DailyTask {
    param([string]$Name)

    Write-Host "========================================="
    Write-Host "タスクスケジューラ削除"
    Write-Host "========================================="
    Write-Host "タスク名: $Name"
    Write-Host ""

    $ExistingTask = Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue

    if ($ExistingTask) {
        Unregister-ScheduledTask -TaskName $Name -Confirm:$false
        Write-Host "✅ タスクを削除しました"
    }
    else {
        Write-Host "ℹ️  タスクが見つかりません（既に削除済み）"
    }

    Write-Host "========================================="
}

# メイン処理
try {
    if ($Unregister) {
        Unregister-DailyTask -Name $TaskName
    }
    else {
        Register-DailyTask -Name $TaskName -ExecutionTime $Time
    }

    exit 0

}
catch {
    Write-Host ""
    Write-Host "ERROR occurred:"
    Write-Host $_.Exception.Message
    Write-Host ""
    exit 1
}
