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

.PARAMETER WaitForUser
    処理完了後、ユーザーが Enter を押すまで待機（10分以上かかる場合に推奨）

.PARAMETER WithInspector
    Playwright Inspector を同時起動（UI 要素の調査に便利）

.PARAMETER SkipSnapshot
    スナップショット作成をスキップ

.EXAMPLE
    .\tools\run_and_snapshot.ps1
    デフォルト設定で実行

.EXAMPLE
    .\tools\run_and_snapshot.ps1 -WaitAfterClick 120
    実行完了まで120秒待機

.EXAMPLE
    .\tools\run_and_snapshot.ps1 -ShowBrowser -WaitForUser
    ブラウザを表示し、ユーザーが Enter を押すまで待機（全銘柄実行時に推奨）

.EXAMPLE
    .\tools\run_and_snapshot.ps1 -ShowBrowser -WaitForUser -WithInspector
    Inspector も同時起動して UI 要素を調査しながら実行
#>

param(
    [string]$Url = "http://localhost:8501",
    [int]$WaitAfterClick = 60,
    [switch]$ShowBrowser,
    [switch]$WaitForUser,
    [switch]$WithInspector,
    [switch]$SkipSnapshot,
    [string]$WaitText,
    [string]$WaitSelector,
    [switch]$NoWaitResults
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"

Push-Location $ProjectRoot
try {
    # Inspector を起動（オプション）
    $InspectorJob = $null
    if ($WithInspector) {
        Write-Host "=== Step 0: Playwright Inspector を起動 ===" -ForegroundColor Cyan
        $InspectorJob = Start-Job -ScriptBlock {
            param($VenvPython, $Url)
            $env:PWDEBUG = "1"
            & $VenvPython -c @"
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context(viewport={'width': 2560, 'height': 1440}, color_scheme='dark')
    page = context.new_page()
    page.goto('$Url')
    page.pause()
    browser.close()
"@
        } -ArgumentList $VenvPython, $Url

        Write-Host "Inspector を起動しました（別ウィンドウ）" -ForegroundColor Green
        Start-Sleep -Seconds 3  # Inspector の起動を待つ
    }

    Write-Host "=== Step 1: Clicking 'Generate Signals' and capturing screenshot ===" -ForegroundColor Cyan

    $CaptureArgs = @(
        "tools/capture_ui_screenshot.py",
        "--url", $Url,
        "--output", "results_images/today_signals_complete.png",
        "--click-button", "Generate Signals",
        "--wait-after-click", $WaitAfterClick
    )

    if ($WaitText) {
        $CaptureArgs += @("--wait-text", $WaitText)
    }
    if ($WaitSelector) {
        $CaptureArgs += @("--wait-selector", $WaitSelector)
    }
    if (-not $NoWaitResults.IsPresent) {
        $CaptureArgs += "--wait-results"
        # Wait until UI progress reaches >=87% to avoid capturing intermediate loading screens
        $CaptureArgs += "--wait-progress-pct"
        $CaptureArgs += "87"
        # Also explicitly wait for the app's completion marker text to appear
        $CaptureArgs += "--wait-text"
        $CaptureArgs += "Signals generation complete"
    }

    if ($ShowBrowser) {
        $CaptureArgs += "--show-browser"
        Write-Host "ブラウザウィンドウを表示します（ダークモード）" -ForegroundColor Yellow
    }

    if ($WaitForUser) {
        $CaptureArgs += "--wait-for-user"
        Write-Host "処理完了後、Enter キーを押すまで待機します" -ForegroundColor Yellow
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

    # Inspector のクリーンアップ
    if ($InspectorJob) {
        Write-Host "`nInspector を終了するには、Inspector ウィンドウを閉じてください" -ForegroundColor Yellow
        # Inspector は手動で閉じる（自動終了させない）
    }

}
finally {
    Pop-Location

    # Inspector ジョブのクリーンアップ（既に終了している場合のみ）
    if ($InspectorJob -and $InspectorJob.State -eq 'Completed') {
        Remove-Job -Job $InspectorJob -Force
    }
}
