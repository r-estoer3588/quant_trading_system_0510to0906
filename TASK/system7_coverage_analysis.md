# System7 カバレッジ分析レポート

**作成日**: 2025 年 10 月 11 日  
**目的**: core/system7.py の 65%カバレッジ達成戦略

## 現状サマリー

| 項目               | 値                 |
| ------------------ | ------------------ |
| 総ステートメント数 | 243 行             |
| カバー済み         | 128 行 (52.7%)     |
| 未カバー           | 115 行 (47.3%)     |
| **目標カバレッジ** | **157 行 (65.0%)** |
| **必要追加カバー** | **29 行 (11.9%)**  |

## 既存テスト状況

### 合格テスト (43 件)

- `test_core_system7_focused.py`: prepare_data 基本動作
- `test_system7_branches.py`: 分岐カバレッジ
- `test_system7_cache.py`: キャッシュロジック
- `test_system7_coverage_boost.py`: カバレッジ補強

### 失敗テスト (修正済み)

- 以前の 11 件失敗は別チャットで解決済み
- test interference 問題が修正された

### 新規テスト (未動作)

- `test_system7_latest_only_path.py`: Lines 219-262 ターゲット
  - 5 件失敗 (test interference 残存の可能性)
  - 単体実行では成功、複数実行で失敗

## 未カバー領域詳細 (115 行)

### 高価値領域 (ROI 高い順)

#### 1. Lines 214-257 (44 行) - latest_only fast-path ⭐⭐⭐

**カバレッジ貢献**: 18.1% (44/243)  
**テスト容易性**: 高  
**ビジネス価値**: 高 (日次シグナル生成の最速パス)

```python
# fast-path実行条件:
# - latest_only=True
# - setup predicate合格
# - entry_date有効

# 主要ロジック:
# - Lines 219-221: setup_ok判定
# - Lines 228-229: Close→entry_price変換
# - Lines 233-244: DataFrame生成
# - Lines 244-248: normalized dict生成
```

**現状**: `test_system7_latest_only_path.py` 作成済みだが未動作

#### 2. Lines 317-336 (20 行) - full_scan ranking ⭐⭐

**カバレッジ貢献**: 8.2% (20/243)  
**テスト容易性**: 中  
**ビジネス価値**: 中 (複数候補のランキング処理)

```python
# 実行条件:
# - latest_only=False
# - 複数セットアップ日あり
# - limit_n指定

# 主要ロジック:
# - Lines 322-326: entry_date計算
# - Lines 327-332: レコード生成
# - Lines 333-336: バケツ管理
```

**アクション**: 新規テスト作成が必要

#### 3. Lines 99-116 (18 行) - cache incremental ⭐

**カバレッジ貢献**: 7.4% (18/243)  
**テスト容易性**: 低 (内部実装依存)  
**ビジネス価値**: 中 (パフォーマンス最適化)

```python
# 実行条件:
# - use_cache=True
# - cached.featherが存在
# - new_rows非空

# 主要ロジック:
# - Lines 99-102: 新規行チェック
# - Lines 105-110: 70日コンテキスト再計算
# - Lines 109-111: max_70優先保持
```

**アクション**: prepare_data_vectorized_system7 API の制約により後回し

### 中価値領域

#### 4. Lines 34-35, 64, 90 (4 行) - エラーパス

**カバレッジ貢献**: 1.6%  
**テスト容易性**: 高  
**ビジネス価値**: 低 (異常系)

#### 5. Lines 122-123, 127-128, etc. - その他分岐

**カバレッジ貢献**: 合計 ~10 行 (4.1%)  
**テスト容易性**: 中  
**ビジネス価値**: 低

## 65%達成戦略

### プラン A: latest_only 修正 (推奨) ✅

1. `test_system7_latest_only_path.py` の test interference 解消
2. 5 件の失敗テスト修正
3. **期待カバレッジ**: 53% + 18% = **71%** (目標超過!)

**アクション**:

- 単体実行で成功しているため、テストデータ修正で対応可能
- 優先度: 最高

### プラン B: full_scan ranking 追加

1. 新規テストファイル作成
2. Lines 317-336 をターゲット
3. **期待カバレッジ**: 53% + 8% = 61%

**アクション**:

- プラン A で不足の場合に実施
- 優先度: 中

### プラン C: 細かい分岐追加

1. エラーパスなど小さな未カバー領域
2. **期待カバレッジ**: +4%程度

**アクション**:

- プラン A+B で不足の場合のみ
- 優先度: 低

## 次のステップ

### 即時実行 (このチャット)

1. ✅ `test_system7_latest_only_path.py` の問題診断
2. ⏳ テストデータ修正 (setup 条件を確実に満たす)
3. ⏳ 5 件失敗テストの修正
4. ⏳ カバレッジ再測定

### 待機中 (別チャット)

- なし (test interference 修正は完了済み)

### 予備計画

- プラン A で 65%未達の場合、プラン B 実施

## 見積もり

| プラン               | 作業時間 | 成功確率 | 期待カバレッジ |
| -------------------- | -------- | -------- | -------------- |
| A (latest_only 修正) | 30 分    | 80%      | 71%            |
| B (full_scan 追加)   | 1 時間   | 70%      | 61%            |
| C (細かい分岐)       | 2 時間   | 50%      | 57%            |

**推奨**: プラン A 実施 → 71%到達で完了

## 参考情報

### ファイル一覧

- `core/system7.py`: ターゲットファイル (243 行)
- `tests/test_system7_latest_only_path.py`: latest_only テスト (作成済み、未動作)
- `tests/test_core_system7_focused.py`: 既存テスト
- `htmlcov/core_system7_py.html`: カバレッジビジュアル

### 関連ドキュメント

- `TASK/parallel_test_fix_prompt.md`: test interference 修正ガイド (完了済み)
- `docs/today_signal_scan/`: システム仕様
