# 3 点同期クイックスタート

**前提**: プロジェクトルートで実行

---

## PowerShell（Windows）

```powershell
# 1. 環境変数設定
$env:ENABLE_PROGRESS_EVENTS = "1"
$env:EXPORT_DIAGNOSTICS_SNAPSHOT_ALWAYS = "1"

# 2a. UI経由でフル実行（別ターミナルでStreamlit起動）
streamlit run apps/app_today_signals.py
# ↓別ターミナル
python scripts/capture_ui_progress.py --headful --dark

# 2b. CLI経由でフル実行（スクショ不要）
python scripts/run_all_systems_today.py --parallel --save-csv

# 3. 出力確認
Get-Content logs/progress_today.jsonl | Measure-Object -Line
Get-ChildItem results_csv/diagnostics_test/diagnostics_snapshot_*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1

# 4. 同期分析
python tools/sync_analysis.py

# 5. 結果確認
code screenshots/progress_tracking/tri_sync_report.json
code screenshots/progress_tracking/ANALYSIS_REPORT.md
```

---

## Bash（Linux/Mac）

```bash
# 1. 環境変数設定
export ENABLE_PROGRESS_EVENTS=1
export EXPORT_DIAGNOSTICS_SNAPSHOT_ALWAYS=1

# 2a. UI経由でフル実行（別ターミナルでStreamlit起動）
streamlit run apps/app_today_signals.py &
python scripts/capture_ui_progress.py --headful --dark

# 2b. CLI経由でフル実行（スクショ不要）
python scripts/run_all_systems_today.py --parallel --save-csv

# 3. 出力確認
wc -l logs/progress_today.jsonl
ls -lt results_csv/diagnostics_test/diagnostics_snapshot_*.json | head -1

# 4. 同期分析
python tools/sync_analysis.py

# 5. 結果確認
cat screenshots/progress_tracking/tri_sync_report.json | jq .
cat screenshots/progress_tracking/ANALYSIS_REPORT.md
```

---

## 期待される結果

- ✅ `tri_sync_report.json` の `sum_jsonl_vs_alloc_match: true`
- ✅ 各システム `status: "match"` が多数
- ✅ `diagnostics_mode: null` または空文字（テストモードでない）

**詳細手順**: [docs/operations/tri_sync_full_run_guide.md](tri_sync_full_run_guide.md)
