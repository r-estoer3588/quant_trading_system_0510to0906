#!/usr/bin/env powershell
# Daily Cache Update Pipeline - PowerShell版

param(
    [switch]$Parallel,
    [int]$Workers = 0
)

Write-Host "🚀 Daily Cache Update Pipeline 開始" -ForegroundColor Green

try {
    # Step 1: cache_daily_data.py
    Write-Host "`n📥 Step 1: Daily data caching (cache_daily_data.py)" -ForegroundColor Cyan
    $startTime1 = Get-Date
    python scripts/cache_daily_data.py
    if ($LASTEXITCODE -ne 0) {
        throw "cache_daily_data.py が失敗しました"
    }
    $duration1 = (Get-Date) - $startTime1
    Write-Host "   ✅ cache_daily_data.py 完了" -ForegroundColor Green

    # Step 2: build_rolling_with_indicators.py
    Write-Host "`n🔁 Step 2: Rolling cache rebuild" -ForegroundColor Cyan
    $startTime2 = Get-Date
    
    if ($Parallel -and $Workers -gt 0) {
        python scripts/build_rolling_with_indicators.py --workers $Workers
        Write-Host "   🔧 並列処理: $Workers ワーカー" -ForegroundColor Yellow
    }
    elseif ($Parallel) {
        python scripts/build_rolling_with_indicators.py
        Write-Host "   🔧 並列処理: デフォルトワーカー数" -ForegroundColor Yellow
    }
    else {
        python scripts/build_rolling_with_indicators.py --workers 1
        Write-Host "   🔧 シリアル実行" -ForegroundColor Yellow
    }
    
    if ($LASTEXITCODE -ne 0) {
        throw "build_rolling_with_indicators.py が失敗しました"
    }
    $duration2 = (Get-Date) - $startTime2
    Write-Host "   ✅ build_rolling_with_indicators.py 完了" -ForegroundColor Green

    # サマリー
    Write-Host "`n🎉 Daily Cache Update Pipeline 完了!" -ForegroundColor Green
    Write-Host "   📋 cache_daily_data: $($duration1.TotalMinutes.ToString('F1')) 分" -ForegroundColor Gray
    Write-Host "   📋 build_rolling: $($duration2.TotalMinutes.ToString('F1')) 分" -ForegroundColor Gray

}
catch {
    Write-Host "`n❌ エラー: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}