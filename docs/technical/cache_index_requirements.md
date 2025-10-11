# Cache Index Requirements (キャッシュインデックス要件)

最終更新: 2025 年 10 月 11 日

## 概要

このドキュメントは、データキャッシュ(rolling/base/full_backup)における**インデックス型の要件と制約**をまとめたものです。特に Feather 形式ファイルの制約と、日付ベース操作を行う際の必須対応について説明します。

## 重要な技術的知見

### 1. Feather 形式の制約

**問題**: Feather 形式(`.feather`ファイル)は、DataFrame のインデックス型情報を保存しません。

**影響**:

- 保存時に`DatetimeIndex`だった DataFrame も、読み込み時は`RangeIndex`(0, 1, 2, ...)になる
- 日付情報は`date`列として保存されるが、**インデックスには格納されない**
- `df.index[-1]`で日付を取得しようとすると、整数(行番号)が返される

**具体例**:

```python
# 保存前 (DatetimeIndex)
df.index = DatetimeIndex(['2025-10-06', '2025-10-07', ..., '2025-10-10'])
df.to_feather('rolling/AAPL.feather')

# 読み込み後 (RangeIndex)
df = pd.read_feather('rolling/AAPL.feather')
df.index  # RangeIndex(start=0, stop=330, step=1) ← 日付情報なし!
df['date']  # DatetimeIndex(['2025-10-06', ...]) ← 列には残っている
```

### 2. 発生したエラーケース (System1)

**エラー**: `ValueError: year 10312 is out of range`

**原因連鎖**:

1. Rolling cache(`.feather`)が`RangeIndex`で読み込まれる
2. `prepare_data_vectorized_system1()`が DataFrame をコピーするだけでインデックス変換しない
3. `generate_candidates_system1()`が`df.index[-1]`で整数(例: 10312)を取得
4. コードが`pd.Timestamp(str(10312))`を実行 → "10312 年"として解釈
5. Pandas の日付範囲外(1677-2262 年)エラー

**発生日**: 2025 年 10 月 11 日  
**該当ファイル**: `core/system1.py`  
**影響範囲**: AA, AAPL, ABT など複数銘柄でエラー

### 3. 解決策: DatetimeIndex 変換の必須化

**修正内容**: `prepare_data_vectorized_system1()` (行 411-418)に以下のコードを追加:

```python
# Ensure date index (if not already set)
if 'date' in x.columns and not isinstance(x.index, pd.DatetimeIndex):
    try:
        x['date'] = pd.to_datetime(x['date'])
        x = x.set_index('date', drop=False)
    except Exception:
        pass  # Keep original index if conversion fails
```

**修正ポイント**:

- インデックス型を明示的にチェック(`isinstance(x.index, pd.DatetimeIndex)`)
- `date`列が存在する場合のみ変換を試行
- `drop=False`で`date`列も残す(後続処理で使用するため)
- 例外発生時は変換をスキップ(既存動作を壊さない)

### 4. Fast-path 最適化における注意点

**背景**:

- System1 は指標キャッシュを再利用する"Fast-path"を実装している
- `prepare_data_vectorized_system1()`で事前計算済み指標をロード
- パフォーマンス向上のため、指標再計算をスキップ

**落とし穴**:

- 指標データだけでなく、**メタデータ(インデックス型)**も保証する必要がある
- `x = df.copy()`だけでは不十分 → インデックス変換が必須
- メモリ効率とデータ整合性のバランスが重要

**教訓**:

> Fast-path 最適化では、データそのものだけでなく、データ構造の前提条件(インデックス型、列順、dtype 等)も明示的に保証すること。

### 5. Silent Exception (静かな例外)の診断手法

**問題**:

- `try-except`ブロックで例外が握りつぶされ、エラーが見えない
- 空のリストや None が返るだけで、原因が不明

**診断手法**:

```python
except Exception as e_latest:
    import traceback
    print(f"[ERROR] System1 latest_only exception: {e_latest}")
    traceback.print_exc()  # ← フルスタックトレースを出力
    if log_callback:
        log_callback(f"System1 latest_only error: {e_latest}")
```

**ベストプラクティス**:

1. まず`traceback.print_exc()`で完全なスタックトレースを表示
2. エラーメッセージに文脈情報(symbol 名、index 値、型情報)を追加
3. 診断完了後、本番用エラーハンドリングに整理

## 実装ガイドライン

### 必須対応: 日付ベース操作を行う場合

日付インデックスを前提とするコード(例: `df.index[-1]`で日付取得)を書く場合:

1. **インデックス型を明示的にチェック**:

   ```python
   if not isinstance(df.index, pd.DatetimeIndex):
       # 変換処理
   ```

2. **変換は冪等的に**:

   ```python
   # 複数回呼ばれても安全
   if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
       df['date'] = pd.to_datetime(df['date'])
       df = df.set_index('date', drop=False)
   ```

3. **エラーハンドリングを追加**:
   ```python
   try:
       # 変換処理
   except Exception:
       pass  # または適切なフォールバック
   ```

### 推奨対応: キャッシュ読み込み時の統一処理

`CacheManager`などキャッシュ読み込みレイヤーで統一的に変換することも検討可能:

```python
def load_rolling(self, symbol: str) -> pd.DataFrame:
    df = pd.read_feather(f'rolling/{symbol}.feather')
    # 統一的にDatetimeIndexに変換
    if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date', drop=False)
    return df
```

**注意**: 既存コードへの影響が大きいため、段階的導入を推奨。

## 関連ファイル

- **キャッシュ管理**: `common/cache_manager.py`
- **System1 Fast-path**: `core/system1.py::prepare_data_vectorized_system1()`
- **候補生成**: `core/system1.py::generate_candidates_system1()`
- **キャッシュ階層**: `data_cache/rolling/`, `data_cache/base/`, `data_cache/full_backup/`

## トラブルシューティング

### 症状: "year XXXXX is out of range" エラー

**原因**: インデックスが整数(RangeIndex)で、日付として解釈されている

**確認方法**:

```python
df = pd.read_feather('data_cache/rolling/AAPL.feather')
print(f"Index type: {type(df.index)}")
print(f"Index dtype: {df.index.dtype}")
print(f"Last 5 indices: {df.index[-5:]}")
```

**修正**: 上記「DatetimeIndex 変換の必須化」を参照

### 症状: 候補数が 0 だがエラーログもない

**原因**: Silent exception で例外が握りつぶされている

**診断**: 該当の`except`ブロックに`traceback.print_exc()`を追加

## 参考情報

### Pandas の日付範囲制限

- **サポート範囲**: 1677 年 9 月 21 日 ～ 2262 年 4 月 11 日
- **超過時エラー**: `OutOfBoundsDatetime` または `ValueError`
- **原因**: Unix timestamp(ns 精度)の限界

### Feather vs Parquet

| 形式    | インデックス保存  | 読み込み速度 | ファイルサイズ |
| ------- | ----------------- | ------------ | -------------- |
| Feather | ❌ (列として保存) | 超高速       | 74%削減        |
| Parquet | ⭕ (保存可能)     | 高速         | 高圧縮         |

**現状**: Feather を使用(速度優先)  
**将来検討**: Parquet 移行でインデックス型も保存可能

## 変更履歴

| 日付       | 変更内容                                                                 | 担当           |
| ---------- | ------------------------------------------------------------------------ | -------------- |
| 2025-10-11 | 初版作成。System1 日付インデックスエラー修正に基づく知見をドキュメント化 | GitHub Copilot |

## 関連ドキュメント

- [キャッシュ設計](../README.md#キャッシュ階層) - キャッシュ階層(rolling/base/full_backup)の説明
- [環境変数](environment_variables.md) - `ROLLING_ISSUES_VERBOSE_HEAD`等の診断用変数
- [エラーハンドリング](error_handling_guide.md) - 例外処理のベストプラクティス
