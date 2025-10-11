# System7 Archived Test Files

このディレクトリには、System7 のカバレッジ改善過程で作成された実験的テストファイルが保管されています。

## アーカイブ日

2025 年 10 月 11 日

## アーカイブ理由

System7 のテストカバレッジが 66%に到達し、目標の 65%を達成したため、公式 4 ファイルに統合されました。

## 公式テストファイル (tests/ 直下に残存)

### 1. test_system7_branches.py (16 テスト、89%カバレッジ)

- **目的**: 分岐条件の網羅的テスト
- **カバー範囲**: 各種条件分岐、エッジケース
- **主要テスト**:
  - Empty data handling
  - Missing columns
  - Invalid date formats
  - ATR50 fallback logic

### 2. test_system7_latest_only.py (10 テスト、87%カバレッジ)

- **目的**: 当日シグナル生成の高速パステスト
- **カバー範囲**: Lines 219-262 (latest_only=True path)
- **主要テスト**:
  - Fast-path execution
  - Setup predicate validation
  - Entry date resolution
  - Diagnostics collection

### 3. test_system7_error_cases.py (9 テスト、96%カバレッジ)

- **目的**: エラーハンドリングと欠損データ処理
- **カバー範囲**: Lines 33-47, 131-147, 197, 201, 214-215, 230
- **主要テスト**:
  - Missing SPY data
  - None/empty DataFrame handling
  - Missing atr50 indicator
  - skip_callback behavior
  - Predicate exception handling
  - ATR50 lowercase fallback

### 4. test_system7_full_scan.py (6 テスト、98%カバレッジ)

- **目的**: フル履歴スキャンモードのテスト
- **カバー範囲**: Lines 275-401 (latest_only=False path)
- **主要テスト**:
  - Multiple candidates generation
  - top_n limit per date (not total dates)
  - Callback invocation (log_callback, progress_callback)
  - Date-based grouping
  - Empty results handling
  - Diagnostics without include_diagnostics flag

## 統合結果

- **総テスト数**: 41 テスト (全てパス)
- **達成カバレッジ**: 66% (162/247 行)
- **開始カバレッジ**: 53% (132/247 行)
- **改善**: +13 ポイント
- **目標**: 65% → **超過達成** ✅

## アーカイブファイル一覧

以下のファイルは実験的/重複/古いバージョンとして保管:

1. `test_system7.py` - 初期テスト
2. `test_system7_cache.py` - キャッシュ関連テスト
3. `test_system7_cache_incremental.py` - インクリメンタルキャッシュテスト
4. `test_system7_coverage_boost.py` - カバレッジ改善試行 1
5. `test_system7_coverage_final.py` - カバレッジ改善試行 2
6. `test_system7_direct.py` - 直接呼び出しテスト
7. `test_system7_edge_cases.py` - エッジケーステスト (branches.py に統合)
8. `test_system7_enhanced.py` - 拡張テスト
9. `test_system7_final_65.py` - 65%目標試行 1
10. `test_system7_final_coverage.py` - 最終カバレッジ試行
11. `test_system7_final_push.py` - 最終プッシュ試行
12. `test_system7_latest_only_parity.py` - Parity 検証 (latest_only.py に統合)
13. `test_system7_latest_only_path.py` - **壊れたファイル** (重複コード、インポートエラー)
14. `test_system7_max70_optimization.py` - max_70 最適化テスト
15. `test_system7_partial.py` - 部分的テスト
16. `test_system7_push_to_65.py` - 65%プッシュ試行 2

## 注意事項

これらのアーカイブファイルは参考用として保管されていますが、**実行は推奨されません**。
以下の理由により:

1. 一部のファイルが壊れている (test_system7_latest_only_path.py)
2. 公式 4 ファイルに機能が統合済み
3. 重複したテストケースが含まれる
4. 実験的な実装が含まれる

## 復元方法

万が一、特定のテストロジックを参照したい場合:

```powershell
# 特定ファイルを tests/ に戻す
Copy-Item tests/experimental/system7_archive/test_system7_*.py tests/
```

ただし、公式 4 ファイルで 66%カバレッジ達成済みのため、通常は不要です。
