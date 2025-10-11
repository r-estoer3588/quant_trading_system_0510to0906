#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Streamlit UIで本日のシグナルを実行し、完了画面をスナップショット

.DESCRIPTION
    1. Playwright で「▶ 本日のシグナル実行」ボタンをクリック
    2. 実行完了を待機（デフォルト30秒）
    3. フルページスクリーンショットを撮影
    4. results_csv, logs, results_images をスナップショット

.PARAMETER Url
    Streamlit アプリのURL（デフォルト: http://localhost:8501）

.PARAMETER WaitAfterClick
    ボタンクリック後の待機時間（秒）（デフォルト: 30）

.PARAMETER SkipSnapshot
    スナップショット作成をスキップ

.EXAMPLE
    .\tools\run_and_snapshot.ps1
    デフォルト設定で実行

.EXAMPLE
    .\tools\run_and_snapshot.ps1 -WaitAfterClick 60
    実行完了まで60秒待機
#>

param(
    [string]$Url = "http://localhost:8501",
    [int]$WaitAfterClick = 30,
    [switch]$SkipSnapshot
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"

Push-Location $ProjectRoot
try {
    Write-Host "=== Step 1: Clicking '▶ 本日のシグナル実行' and capturing screenshot ===" -ForegroundColor Cyan

    & $VenvPython tools/capture_ui_screenshot.py `
        --url $Url `
        --output results_images/today_signals_complete.png `
        --click-button "▶ 本日のシグナル実行" `
        --wait-after-click $WaitAfterClick

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Screenshot failed"
        exit 1
    }

    if (-not $SkipSnapshot) {
        Write-Host "`n=== Step 2: Creating snapshot ===" -ForegroundColor Cyan

        & $VenvPython tools/snapshot.py `
            --source results_csv `
            --source logs `
            --source results_images

        if ($LASTEXITCODE -ne 0) {
            Write-Error "Snapshot failed"
            exit 1
        }
    }

    Write-Host "`n✅ Complete!" -ForegroundColor Green
    Write-Host "Screenshot: results_images/today_signals_complete.png"

    if (-not $SkipSnapshot) {
        $SnapshotDir = Get-ChildItem snapshots | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        Write-Host "Snapshot:   snapshots/$($SnapshotDir.Name)"
    }

}
finally {
    Pop-Location
}
