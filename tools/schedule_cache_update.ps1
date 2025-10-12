<#
.SYNOPSIS
    Windows タスクスケジューラにキャッシュ更新タスクを登録

.DESCRIPTION
    毎朝 6:00（プレマーケット前）にデータキャッシュを更新するタスクを登録。

.PARAMETER Time
    実行時刻（24時間形式: HH:MM）

.PARAMETER TaskName
    タスク名（デフォルト: "QuantTradingCacheUpdate"）

.PARAMETER Unregister
    タスクを削除

.EXAMPLE
    .\tools\schedule_cache_update.ps1
    .\tools\schedule_cache_update.ps1 -Time "06:30"
    .\tools\schedule_cache_update.ps1 -Unregister
#>

param(
    [string]$Time = "09:00",  # 日本時間 09:00 (米国時間 18:00 ET 頃、データ更新完了後)
    [string]$TaskName = "QuantTradingCacheUpdate",
    [switch]$Unregister = $false
)

$ErrorActionPreference = "Stop"

# プロジェクトルートを取得（スクリプトレベルで実行）
$ScriptPath = if ($PSCommandPath) { $PSCommandPath } else { $MyInvocation.MyCommand.Path }
$ScriptDir = Split-Path -Parent $ScriptPath
$ProjectRoot = Split-Path -Parent $ScriptDir

function Register-CacheUpdateTask {
    param(
        [string]$Name,
        [string]$ExecutionTime
    )

    # リトライ付きのガードラッパーを呼ぶ
    $UpdateScript = Join-Path $ProjectRoot "scripts\guarded_cache_update.ps1"

    if (-not (Test-Path $UpdateScript)) {
        throw "実行スクリプトが見つかりません: $UpdateScript"
    }

    Write-Host "========================================="
    Write-Host "キャッシュ更新タスク登録"
    Write-Host "========================================="
    Write-Host "タスク名: $Name"
    Write-Host "実行時刻: $ExecutionTime (日本時間)"
    Write-Host "スクリプト: $UpdateScript"
    Write-Host ""
    Write-Host "⚠️  重要: EODHD データは米国市場終了後 2-3時間で更新されます"
    Write-Host "    推奨時刻: 09:00 JST (18:00 ET) 以降"
    Write-Host "    シグナル生成は 10:00 JST に設定してください"
    Write-Host ""

    # 既存タスクを削除
    $ExistingTask = Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue

    if ($ExistingTask) {
        Write-Host "⚠️  既存のタスクを削除します..."
        Unregister-ScheduledTask -TaskName $Name -Confirm:$false
    }

    # タスクアクション（並列実行版）
    $Action = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$UpdateScript`" -Parallel -Workers 4"

    # トリガー（平日毎日 指定時刻）。さらに 1 時間、15 分間隔で繰り返し
    $Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $ExecutionTime

    # Repetitionを作成（CIMインスタンスとして）
    $RepetitionInterval = New-TimeSpan -Minutes 15
    $RepetitionDuration = New-TimeSpan -Hours 1
    $Trigger.Repetition = (New-ScheduledTaskTrigger -Once -At $ExecutionTime -RepetitionInterval $RepetitionInterval -RepetitionDuration $RepetitionDuration).Repetition

    # 設定（キャッシュ更新は時間がかかるため 60分制限、ネットワーク必須）
    $Settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RunOnlyIfNetworkAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 60)

    # タスク登録
    Register-ScheduledTask `
        -TaskName $Name `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Description "Quant Trading System のデータキャッシュを自動更新" `
        -User $env:USERNAME `
        -RunLevel Limited

    Write-Host ""
    Write-Host "✅ タスクを登録しました"
    Write-Host ""
    Write-Host "確認方法:"
    $Trigger.Repetition = New-ScheduledTaskTrigger -Once -At $ExecutionTime
    Write-Host ""
    Write-Host "手動実行:"
    Write-Host "  Start-ScheduledTask -TaskName '$Name'"
    Write-Host ""
    Write-Host "タスク削除:"
    Write-Host "  .\tools\schedule_cache_update.ps1 -Unregister"
    Write-Host "========================================="
}

function Unregister-CacheUpdateTask {
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
        Unregister-CacheUpdateTask -Name $TaskName
    }
    else {
        Register-CacheUpdateTask -Name $TaskName -ExecutionTime $Time
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
