<#
.SYNOPSIS
    日次シグナル生成の自動実行スクリプト

.DESCRIPTION
    毎日決まった時間にシグナル生成を自動実行し、結果を Slack/メールで通知。
    Windows タスクスケジューラから実行されることを想定。

.PARAMETER DryRun
    ドライランモード（通知なし）

.PARAMETER SkipNotification
    通知をスキップ

.EXAMPLE
    .\scripts\daily_auto_run.ps1
    .\scripts\daily_auto_run.ps1 -DryRun
#>

param(
    [switch]$DryRun = $false,
    [switch]$SkipNotification = $false
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$LogDir = Join-Path $ProjectRoot "logs"
$LogFile = Join-Path $LogDir "auto_run_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

# ログディレクトリ作成
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

function Write-Log {
    param([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogMessage = "[$Timestamp] $Message"
    Write-Host $LogMessage
    Add-Content -Path $LogFile -Value $LogMessage -Encoding UTF8
}

try {
    Write-Log "========================================="
    Write-Log "日次シグナル自動実行開始"
    Write-Log "========================================="

    # Python 仮想環境をアクティベート
    $VenvPath = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"

    if (Test-Path $VenvPath) {
        Write-Log "Python 仮想環境をアクティベート: $VenvPath"
        & $VenvPath
    }
    else {
        Write-Log "⚠️  仮想環境が見つかりません。システムの Python を使用します。"
    }

    # 休場日チェック（週末/US 休日時はスキップして Slack に通知）
    try {
        $holidayCheck = & python (Join-Path $ProjectRoot "scripts\market_holiday_check.py")
        if ($LASTEXITCODE -eq 2) {
            Write-Log "ℹ️  休場日のためシグナル生成をスキップします"
            if (-not $DryRun -and -not $SkipNotification) {
                & python (Join-Path $ProjectRoot "scripts\notify_info.py") --title "市場休場" --message "本日は US 市場が休場のため、シグナル生成をスキップしました（前日据え置き）。"
            }
            exit 0
        }
    }
    catch {
        Write-Log "⚠️  休場日チェックに失敗（フォールバック: 実行継続）"
    }

    # シグナル生成実行
    Write-Log "シグナル生成を開始..."

    $PythonScript = Join-Path $ProjectRoot "scripts\run_all_systems_today.py"

    $Env:COMPACT_TODAY_LOGS = "1"
    $Env:ENABLE_PROGRESS_EVENTS = "1"

    $Result = & python $PythonScript --parallel --save-csv 2>&1
    $ExitCode = $LASTEXITCODE

    # 出力をログに保存
    $Result | ForEach-Object { Write-Log $_ }

    if ($ExitCode -ne 0) {
        throw "シグナル生成が失敗しました (Exit Code: $ExitCode)"
    }

    Write-Log "✅ シグナル生成完了"

    # 結果サマリーを生成
    $SignalsDir = Join-Path $ProjectRoot "data_cache\signals"
    $LatestCSV = Get-ChildItem "$SignalsDir\signals_final_*.csv" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

    if ($LatestCSV) {
        $SignalData = Import-Csv $LatestCSV.FullName
        $SignalCount = $SignalData.Count

        $SystemCounts = $SignalData | Group-Object system | ForEach-Object {
            "$($_.Name): $($_.Count)件"
        }

        Write-Log "📊 生成されたシグナル:"
        Write-Log "   総数: $SignalCount 件"
        $SystemCounts | ForEach-Object { Write-Log "   $_" }
    }
    else {
        $SignalCount = 0
        Write-Log "⚠️  シグナルCSVが見つかりません"
    }

    # Slack/メール通知
    if (-not $DryRun -and -not $SkipNotification) {
        Write-Log "通知を送信中..."

        $NotifyScript = Join-Path $ProjectRoot "scripts\notify_results.py"

        if (Test-Path $NotifyScript) {
            & python $NotifyScript --signals $SignalCount --log $LogFile

            if ($LASTEXITCODE -eq 0) {
                Write-Log "✅ 通知送信完了"
            }
            else {
                Write-Log "⚠️  通知送信に失敗しました"
            }
        }
        else {
            Write-Log "⚠️  通知スクリプトが見つかりません: $NotifyScript"
        }
    }
    elseif ($DryRun) {
        Write-Log "ℹ️  [DryRun] 通知スキップ"
    }

    Write-Log "========================================="
    Write-Log "日次シグナル自動実行完了"
    Write-Log "========================================="

    exit 0

}
catch {
    Write-Log "========================================="
    Write-Log "❌ エラーが発生しました"
    Write-Log "========================================="
    Write-Log "エラー: $_"
    Write-Log "Stack Trace:"
    Write-Log $_.ScriptStackTrace

    # エラー通知
    if (-not $DryRun -and -not $SkipNotification) {
        $ErrorNotifyScript = Join-Path $ProjectRoot "scripts\notify_error.py"

        if (Test-Path $ErrorNotifyScript) {
            & python $ErrorNotifyScript --error "$_" --log $LogFile 2>&1 |
            ForEach-Object { Write-Log $_ }
        }
    }

    exit 1
}
