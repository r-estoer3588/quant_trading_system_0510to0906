# Start-Dashboard.ps1
# 統合ダッシュボード起動スクリプト

Write-Host "🚀 統合ダッシュボードを起動します..." -ForegroundColor Cyan

# FastAPI バックエンド起動
Write-Host "📡 FastAPI (port 8000) を起動中..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
cd c:\Repos\quant_trading_system
Write-Host '🔧 FastAPI Backend Starting...' -ForegroundColor Green
python -m uvicorn apps.api.main:app --reload --port 8000
"@

# 少し待機
Start-Sleep -Seconds 2

# Next.js フロントエンド起動
Write-Host "🌐 Next.js (port 3000) を起動中..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
cd c:\Repos\quant_trading_system\apps\dashboards\alpaca-next
Write-Host '⚛️ Next.js Frontend Starting...' -ForegroundColor Green
npm run dev -- --port 3000
"@

# ブラウザを開く
Start-Sleep -Seconds 3
Write-Host "🌍 ブラウザを開きます..." -ForegroundColor Cyan
Start-Process "http://localhost:3000/integrated"

Write-Host ""
Write-Host "✅ 起動完了！" -ForegroundColor Green
Write-Host "  - FastAPI: http://localhost:8000" -ForegroundColor White
Write-Host "  - Next.js: http://localhost:3000" -ForegroundColor White
Write-Host "  - ダッシュボード: http://localhost:3000/integrated" -ForegroundColor White
