<#
.SYNOPSIS
  祝日/週末スキップ + リトライ付きのキャッシュ更新ラッパー

.DESCRIPTION
  - 市場休場日は Slack に「前日据え置き」メッセージを送り、更新をスキップ
  - 失敗時は 15 分間隔で最大 4 回（合計 1 時間）リトライ
  - 成功時のみ 0 終了コード、スキップは 0、最終失敗は 1
#>

param(
    [int]$MaxRetries = 4,
    [int]$DelayMinutes = 15,
    # Opt-in: when specified the update script will invoke a recompute dry-run after a successful cache update
    [switch]$RunRecomputeOnSuccess = $false,
    # Number of workers to pass to the recompute wrapper when RunRecomputeOnSuccess is enabled
    [int]$RecomputeWorkers = 2
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

function Invoke-SlackInfo {
    param([string]$Title, [string]$Message)
    try {
        & python "$ProjectRoot\scripts\notify_info.py" --title $Title --message $Message | Out-Host
    }
    catch {}
}

# 1) 休場日チェック（週末含む）
& python "$ProjectRoot\scripts\market_holiday_check.py"
if ($LASTEXITCODE -eq 2) {
    Invoke-SlackInfo -Title "市場休場/週末" -Message "本日は US 市場が休場のため、キャッシュ更新をスキップします（前日据え置き）。"
    exit 0
}

# 2) 更新本体 + リトライ
$UpdateScript = Join-Path $ProjectRoot "scripts\update_cache_all.ps1"
if (-not (Test-Path $UpdateScript)) {
    Write-Host "❌ update_cache_all.ps1 が見つかりません: $UpdateScript"
    exit 1
}

for ($i = 0; $i -le $MaxRetries; $i++) {
    try {
        Write-Host "🚀 キャッシュ更新を実行 (試行 $($i+1)/$($MaxRetries+1))..."
        powershell -NoProfile -ExecutionPolicy Bypass -File $UpdateScript -Parallel -Workers 4
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ キャッシュ更新 成功"

            if ($RunRecomputeOnSuccess) {
                try {
                    $RecomputeWrapper = Join-Path $ProjectRoot "scripts\run_recompute_daily.ps1"
                    if (-not (Test-Path $RecomputeWrapper)) {
                        Write-Host "⚠️  Recompute wrapper not found, skipping recompute: $RecomputeWrapper"
                    }
                    else {
                        Write-Host "ℹ️  Running recompute dry-run (invoked by cache update success)..."
                        # Call the recompute wrapper synchronously so logs are available in the same run
                        powershell -NoProfile -ExecutionPolicy Bypass -File $RecomputeWrapper -DryRun -Workers $RecomputeWorkers
                        $rc_recomp = $LASTEXITCODE
                        if ($rc_recomp -eq 0) {
                            Write-Host "✅ Recompute dry-run completed successfully (exit code 0)."
                        }
                        else {
                            Write-Host "⚠️  Recompute dry-run exited with code: $rc_recomp"
                        }
                    }
                }
                catch {
                    Write-Host "⚠️  Recompute invocation failed: $_"
                }
            }

            exit 0
        }
        else {
            throw "update_cache_all.ps1 exit code: $LASTEXITCODE"
        }
    }
    catch {
        Write-Host "⚠️  キャッシュ更新 失敗: $_"
        if ($i -lt $MaxRetries) {
            Write-Host "⏳ $DelayMinutes 分後にリトライします..."
            Start-Sleep -Seconds ($DelayMinutes * 60)
        }
        else {
            Write-Host "❌ 規定回数リトライしても成功しませんでした"
            exit 1
        }
    }
}
