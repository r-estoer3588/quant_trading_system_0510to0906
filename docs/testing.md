# テスト実行ガイド

## 概要

本プロジェクトでは、単体テストと統合テスト（パイプラインテスト）を提供しています。

## 単体テスト

### 基本実行

```bash
# 全テスト実行
pytest

# 高速実行（詳細出力なし）
pytest -q

# カバレッジ付き実行
pytest --cov --cov-report=term --cov-report=html:htmlcov
```

### 特定テスト実行

```bash
# 特定ファイルのみ
pytest tests/test_cache_manager_working.py -v

# 特定テストクラス
pytest tests/test_core_system6_focused.py::TestSystem6IndicatorComputation -v

# 特定テストメソッド
pytest tests/test_core_system6_focused.py::TestSystem6IndicatorComputation::test_compute_indicators_from_frame_basic -v
```

## パイプライン高速テスト

### テスト高速化オプション

`run_all_systems_today.py` スクリプトに以下のオプションを追加して、テスト実行を大幅に高速化できます。

| オプション           | 説明                        | 効果                   |
| -------------------- | --------------------------- | ---------------------- |
| `--test-mode mini`   | 10 銘柄に制限               | 2 秒で完了             |
| `--test-mode quick`  | 50 銘柄に制限               | 約 10 秒で完了         |
| `--test-mode sample` | 100 銘柄に制限              | 約 30 秒で完了         |
| `--skip-external`    | 外部 API 呼び出しをスキップ | ネットワーク依存を排除 |
| `--benchmark`        | パフォーマンス計測          | 実行時間レポート生成   |

### 4 つのテストシナリオ

#### 1. 基本パイプラインテスト

```bash
python scripts/run_all_systems_today.py --test-mode mini --skip-external --benchmark
```

**実行内容:**

- 8 フェーズパイプライン処理
- System1-7 のシグナル抽出
- 配分・最終リスト生成
- 実行時間: **約 2 秒**

**検証項目:**

- ✅ 銘柄ユニバース構築
- ✅ データ読み込み（rolling キャッシュ）
- ✅ 指標事前計算
- ✅ フィルター実行（二段階処理）
- ✅ セットアップ評価
- ✅ シグナル抽出
- ✅ 配分・最終リスト生成
- ✅ 保存・通知フェーズ

#### 2. 並列処理テスト

```bash
python scripts/run_all_systems_today.py --test-mode mini --skip-external --parallel --benchmark
```

**実行内容:**

- システム別並列実行
- ThreadPoolExecutor による処理最適化
- 実行時間: **約 2 秒**

**検証項目:**

- ✅ 並列実行制御
- ✅ スレッド安全性
- ✅ パフォーマンス向上確認

#### 3. CSV 保存機能テスト

```bash
python scripts/run_all_systems_today.py --test-mode mini --skip-external --save-csv --benchmark
```

**実行内容:**

- CSV ファイル生成機能
- `data_cache/signals/` への出力
- 実行時間: **約 2 秒**

**検証項目:**

- ✅ CSV 出力機能
- ✅ ファイル命名規則
- ✅ データ形式の正確性

#### 4. 全機能統合テスト

```bash
python scripts/run_all_systems_today.py --test-mode quick --skip-external --parallel --save-csv --benchmark
```

**実行内容:**

- 50 銘柄での統合テスト
- 並列処理 + CSV 保存
- 実行時間: **約 10 秒**

**検証項目:**

- ✅ 全機能統合動作
- ✅ スケーラビリティ確認
- ✅ 出力品質検証

### 実行時間比較

| モード | 銘柄数 | 通常実行 | 高速テスト | 短縮率 |
| ------ | ------ | -------- | ---------- | ------ |
| mini   | 10     | 数分     | **2 秒**   | 99%↓   |
| quick  | 50     | 10 分+   | **10 秒**  | 95%↓   |
| sample | 100    | 30 分+   | **30 秒**  | 98%↓   |

### CI/CD での使用

```yaml
# GitHub Actions 例
- name: 高速パイプラインテスト
  run: |
    python scripts/run_all_systems_today.py --test-mode mini --skip-external --benchmark
    python scripts/run_all_systems_today.py --test-mode mini --skip-external --parallel --benchmark
```

### トラブルシューティング

#### 外部依存エラー

```bash
# 問題: NASDAQ API 接続エラー
# 解決: --skip-external オプション使用
python scripts/run_all_systems_today.py --test-mode mini --skip-external
```

#### データ不足エラー

```bash
# 問題: SPY データが見つからない
# 解決: 必要最小限のキャッシュ確保
python scripts/cache_daily_data.py --symbols SPY
```

#### パフォーマンス測定

```bash
# ベンチマークレポート生成
python scripts/run_all_systems_today.py --test-mode sample --benchmark

# 結果確認
ls logs/perf/
```

## テストデータ管理

### 最小構成

高速テストは以下の最小データで動作します：

- SPY: 1 銘柄（System7 アンカー用）
- rolling キャッシュ: 直近 300 営業日分
- 外部 API: 完全スキップ可能

### データ更新

```bash
# Rolling キャッシュ更新
python scripts/build_rolling_with_indicators.py --workers 4

# 特定銘柄のみ
python scripts/cache_daily_data.py --symbols SPY AAPL MSFT
```

## 品質保証

### pre-commit フック

```bash
# 全ファイルチェック
pre-commit run --all-files

# 変更ファイルのみ
pre-commit run
```

### 継続的テスト

```bash
# 開発中の継続実行
pytest --looponfail
```

## まとめ

- **単体テスト**: `pytest -q` で基本検証
- **統合テスト**: `--test-mode mini --skip-external` で高速検証
- **本格テスト**: データ完備後の `--parallel --save-csv` で本番同等検証
- **CI/CD**: 高速オプションで短時間での品質保証

テスト実行時間の大幅短縮により、開発効率と品質保証の両立が可能になりました。
