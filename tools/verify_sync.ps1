# JSONL・ログ・CSVの三点同期検証スクリプト
# 使い方: .\tools\verify_sync.ps1

$ErrorActionPreference = "Continue"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  同期検証: JSONL vs ログ vs CSV" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# 1. JSONL進捗イベントの確認
Write-Host "【1】JSONL進捗イベント (logs/progress_today.jsonl)" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────────" -ForegroundColor Gray

try {
    $jsonl = Get-Content logs\progress_today.jsonl -ErrorAction Stop | ConvertFrom-Json

    # 主要イベントを抽出
    $session_start = $jsonl | Where-Object { $_.event_type -eq 'session_start' } | Select-Object -First 1
    $pipeline_complete = $jsonl | Where-Object { $_.event_type -eq 'pipeline_complete' } | Select-Object -Last 1
    $phase1 = $jsonl | Where-Object { $_.event_type -eq 'phase1_symbol_universe_complete' } | Select-Object -Last 1
    $phase2 = $jsonl | Where-Object { $_.event_type -eq 'phase2_data_loading_complete' } | Select-Object -Last 1
    $phase5 = $jsonl | Where-Object { $_.event_type -eq 'phase5_allocation_complete' } | Select-Object -Last 1
    $systems = $jsonl | Where-Object { $_.event_type -eq 'system_complete' }

    Write-Host "セッション開始: $($session_start.timestamp)" -ForegroundColor Green
    if ($phase1) {
        Write-Host "Phase1 (シンボル): $($phase1.data.symbols) 銘柄" -ForegroundColor Green
    }
    if ($phase2) {
        Write-Host "Phase2 (データ読込): $($phase2.data.loaded_assets) 資産" -ForegroundColor Green
    }

    Write-Host "`nシステム別候補数:" -ForegroundColor Cyan
    foreach ($sys in $systems) {
        $name = $sys.data.system
        $cand = $sys.data.candidates
        Write-Host "  $name : $cand 件" -ForegroundColor White
    }

    if ($phase5) {
        $final_df = $phase5.data.final_df_rows
        Write-Host "`nPhase5 (配分後): $final_df 行" -ForegroundColor Green
    }

    if ($pipeline_complete) {
        $final_rows = $pipeline_complete.data.final_rows
        Write-Host "パイプライン完了: $final_rows 行 (最終)" -ForegroundColor Green
        Write-Host "完了時刻: $($pipeline_complete.timestamp)" -ForegroundColor Green
    }
    else {
        Write-Host "⚠️  pipeline_complete イベントが見つかりません（実行中?）" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "❌ JSONL読み込みエラー: $_" -ForegroundColor Red
}

# 2. コンソールログの確認
Write-Host "`n`n【2】コンソールログ (logs/today_signals*.log)" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────────" -ForegroundColor Gray

try {
    $log_latest = Get-ChildItem logs\today_signals*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($log_latest) {
        Write-Host "最新ログ: $($log_latest.Name)" -ForegroundColor White

        # Phase 0
        $phase0 = Get-Content $log_latest.FullName | Select-String "Phase 0 完了.*銘柄が処理対象" | Select-Object -Last 1
        if ($phase0) {
            Write-Host $phase0.Line -ForegroundColor White
        }

        # システム完了
        Write-Host "`nシステム完了:" -ForegroundColor Cyan
        Get-Content $log_latest.FullName | Select-String "system\d+ 完了: \d+件" | Select-Object -Last 7 | ForEach-Object {
            Write-Host "  $($_.Line)" -ForegroundColor White
        }

        # 最終候補
        $final_cand = Get-Content $log_latest.FullName | Select-String "最終候補件数: \d+" | Select-Object -Last 1
        if ($final_cand) {
            Write-Host "`n$($final_cand.Line)" -ForegroundColor Green
        }

        # AllocationSummary
        $alloc = Get-Content $log_latest.FullName | Select-String "AllocationSummary final_counts=" | Select-Object -Last 1
        if ($alloc) {
            Write-Host $alloc.Line -ForegroundColor Green
        }

        # 処理終了
        $end = Get-Content $log_latest.FullName | Select-String "シグナル検出処理終了.*最終候補" | Select-Object -Last 1
        if ($end) {
            Write-Host $end.Line -ForegroundColor Green
        }
    }
    else {
        Write-Host "⚠️  ログファイルが見つかりません" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "❌ ログ読み込みエラー: $_" -ForegroundColor Red
}

# 3. CSV結果の確認
Write-Host "`n`n【3】CSV結果 (data_cache/signals/)" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────────" -ForegroundColor Gray

try {
    $csv_latest = Get-ChildItem data_cache\signals\signals_final_*.csv | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($csv_latest) {
        Write-Host "最新CSV: $($csv_latest.Name)" -ForegroundColor White
        Write-Host "更新時刻: $($csv_latest.LastWriteTime)" -ForegroundColor White

        $csv_data = Import-Csv $csv_latest.FullName
        $total_rows = $csv_data.Count
        Write-Host "総行数: $total_rows" -ForegroundColor Green

        Write-Host "`nシステム別集計:" -ForegroundColor Cyan
        $csv_data | Group-Object system | Sort-Object Name | ForEach-Object {
            Write-Host "  $($_.Name): $($_.Count) 件" -ForegroundColor White
        }

        Write-Host "`nサイド別集計:" -ForegroundColor Cyan
        $csv_data | Group-Object side | ForEach-Object {
            Write-Host "  $($_.Name): $($_.Count) 件" -ForegroundColor White
        }
    }
    else {
        Write-Host "⚠️  CSVファイルが見つかりません" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "❌ CSV読み込みエラー: $_" -ForegroundColor Red
}

# 4. 同期チェック
Write-Host "`n`n【4】同期チェック結果" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────────" -ForegroundColor Gray

try {
    # JSONL final_rows
    $jsonl_final = ($jsonl | Where-Object { $_.event_type -eq 'pipeline_complete' } | Select-Object -Last 1).data.final_rows

    # CSV総行数
    $csv_total = 0
    if ($csv_latest) {
        $csv_total = (Import-Csv $csv_latest.FullName).Count
    }

    # ログ最終候補
    $log_final = 0
    if ($log_latest) {
        $match = Get-Content $log_latest.FullName | Select-String "シグナル検出処理終了.*最終候補 (\d+) 件" | Select-Object -Last 1
        if ($match -and $match.Matches.Groups.Count -gt 1) {
            $log_final = [int]$match.Matches.Groups[1].Value
        }
    }

    Write-Host "JSONL final_rows  : $jsonl_final" -ForegroundColor White
    Write-Host "ログ 最終候補    : $log_final" -ForegroundColor White
    Write-Host "CSV 総行数        : $csv_total" -ForegroundColor White

    $all_match = ($jsonl_final -eq $log_final) -and ($log_final -eq $csv_total) -and ($jsonl_final -gt 0)

    if ($all_match) {
        Write-Host "`n✅ 完全同期確認！すべて一致しています。" -ForegroundColor Green
    }
    elseif ($jsonl_final -eq 0 -or $null -eq $jsonl_final) {
        Write-Host "`n⚠️  実行がまだ完了していない可能性があります。" -ForegroundColor Yellow
    }
    else {
        Write-Host "`n❌ 不一致を検出しました。詳細を確認してください。" -ForegroundColor Red
    }

}
catch {
    Write-Host "❌ 同期チェックエラー: $_" -ForegroundColor Red
}

Write-Host "`n========================================`n" -ForegroundColor Cyan
