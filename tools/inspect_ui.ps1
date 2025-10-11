#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Playwright Inspector で UI 要素を調査

.DESCRIPTION
    Streamlit アプリを開いて、UI 要素のセレクターを取得します。
    ブラウザウィンドウとインスペクターが表示されます。

.PARAMETER Url
    Streamlit アプリのURL（デフォルト: http://localhost:8501）

.EXAMPLE
    .\tools\inspect_ui.ps1
    Inspector を起動してセレクターを調査
#>

param(
    [string]$Url = "http://localhost:8501"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"

Push-Location $ProjectRoot
try {
    Write-Host "=== Playwright Inspector を起動します ===" -ForegroundColor Cyan
    Write-Host "1. ブラウザウィンドウが開きます" -ForegroundColor Yellow
    Write-Host "2. Inspector ウィンドウの 'Record' ボタンをクリック" -ForegroundColor Yellow
    Write-Host "3. ブラウザで調査したい要素の上にマウスを置く" -ForegroundColor Yellow
    Write-Host "4. セレクターが自動的に Inspector に表示されます" -ForegroundColor Yellow
    Write-Host ""

    # Playwright Inspector を起動
    $env:PWDEBUG = "1"
    & $VenvPython -c @"
from playwright.sync_api import sync_playwright
import time

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page(viewport={'width': 1920, 'height': 1080})
    page.goto('$Url')

    print('Inspector が開きました。調査が終わったらブラウザを閉じてください。')
    page.pause()  # Inspector を開く

    browser.close()
"@

}
finally {
    Pop-Location
    Remove-Item Env:PWDEBUG -ErrorAction SilentlyContinue
}
