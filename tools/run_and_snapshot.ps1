#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Streamlit UIで本日のシグナルを実行し、完了画面をスナップショット

.DESCRIPTION
    1. Playwright で「▶ 本日のシグナル実行」ボタンをクリック
    2. 実行完了を待機（デフォルト60秒）
    3. フルページスクリーンショットを撮影
    4. results_csv, logs, results_images をスナップショット

.PARAMETER Url
    Streamlit アプリのURL（デフォルト: http://localhost:8501）

.PARAMETER WaitAfterClick
    ボタンクリック後の待機時間（秒）（デフォルト: 60）

.PARAMETER ShowBrowser
    ブラウザウィンドウを表示（デバッグ用）

.PARAMETER SkipSnapshot
    スナップショット作成をスキップ

.EXAMPLE
    .\tools\run_and_snapshot.ps1
    デフォルト設定で実行

.EXAMPLE
    .\tools\run_and_snapshot.ps1 -WaitAfterClick 120
    実行完了まで120秒待機

.EXAMPLE
    .\tools\run_and_snapshot.ps1 -ShowBrowser
    ブラウザを表示して実行過程を確認
#>

param(
    [string]$Url = "http://localhost:8501",
    [int]$WaitAfterClick = 60,
    [switch]$ShowBrowser,
    [switch]$SkipSnapshot
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"

Push-Location $ProjectRoot
try {
    Write-Host "=== Step 1: Clicking '▶ 本日のシグナル実行' and capturing screenshot ===" -ForegroundColor Cyan

    $CaptureArgs = @(
        "tools/capture_ui_screenshot.py",
        "--url", $Url,
        "--output", "results_images/today_signals_complete.png",
        "--click-button", "▶ 本日のシグナル実行",
        "--wait-after-click", $WaitAfterClick
    )

    if ($ShowBrowser) {
        $CaptureArgs += "--show-browser"
        Write-Host "ブラウザウィンドウを表示します（デバッグモード）" -ForegroundColor Yellow
    }

    & $VenvPython @CaptureArgs

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
