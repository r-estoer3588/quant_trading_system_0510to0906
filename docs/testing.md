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

`run_all_systems_today.py` には複数のテストモードが用意されており、開発フェーズや CI/CD の用途に応じて実行コストを調整できます。

### テスト高速化オプション

| オプション                 | 説明                                    | 効果                             |
| -------------------------- | --------------------------------------- | -------------------------------- |
| `--test-mode mini`         | 銘柄数を 10 件に制限                    | 実行時間はおおむね 2 秒          |
| `--test-mode quick`        | 銘柄数を 50 件に制限                    | 実行時間はおおむね 10 秒         |
| `--test-mode sample`       | 銘柄数を 100 件に制限                   | 実行時間はおおむね 30 秒         |
| `--test-mode test_symbols` | 架空データ 113 件を使用 (詳細は後述)    | 実行時間は 1 分前後で再現性 100% |
| `--skip-external`          | 外部 API を呼び出さずキャッシュのみ利用 | ネットワーク待ちを排除           |
| `--benchmark`              | パフォーマンス情報を JSON へ出力        | 実行時間の変化を定量確認         |

### 5 つのテストシナリオ

#### 1. 制御されたテスト環境 (推奨)

```bash
# テスト銘柄生成 (初回のみ)
python tools/generate_test_symbols.py

# パイプライン実行
python scripts/run_all_systems_today.py --test-mode test_symbols --skip-external --benchmark

# まとめて実行する場合
python run_controlled_tests.py
```

**検証内容**

- フィルター、セットアップ、エントリーの全段階を通過する銘柄が揃う
- TRDlist と Entry の双方で 10 銘柄を必ず生成
- 診断 JSON、ベンチマーク JSON、TRDlist CSV を自動検証
- 外部 API を一切利用せず再現性が完全に担保される

#### 2. 基本パイプラインテスト

```bash
python scripts/run_all_systems_today.py --test-mode mini --skip-external --benchmark
```

フェーズ 1 から 8 までの動作を最小コストで確認します。

#### 3. 並列処理テスト

```bash
python scripts/run_all_systems_today.py --test-mode mini --skip-external --parallel --benchmark
```

ThreadPoolExecutor を使った並列実行の安定性を確認します。

#### 4. CSV 保存機能テスト

```bash
python scripts/run_all_systems_today.py --test-mode mini --skip-external --save-csv --benchmark
```

シグナル CSV の命名規則、出力先、内容を検証します。

#### 5. 全機能統合テスト

```bash
python scripts/run_all_systems_today.py --test-mode quick --skip-external --parallel --save-csv --benchmark
```

50 銘柄規模で並列処理と CSV 保存を組み合わせ、本番運用に近い負荷を掛けます。

### 実行時間比較

| モード       | 銘柄数 | 実行時間 | 再現性 | 主な用途                     |
| ------------ | ------ | -------- | ------ | ---------------------------- |
| test_symbols | 113    | 約 1 分  | 100%   | 開発、PR、CI/CD の基準テスト |
| mini         | 10     | 約 2 秒  | 中     | 最小限の動作確認             |
| quick        | 50     | 約 10 秒 | 中     | 中規模の統合検証             |
| sample       | 100    | 約 30 秒 | 中     | データ量を増やした検証       |

### CI/CD 例

```yaml
# GitHub Actions サンプル
- name: Generate controlled test symbols
  run: python tools/generate_test_symbols.py

- name: Run controlled pipeline test
  run: python scripts/run_all_systems_today.py --test-mode test_symbols --skip-external --benchmark

- name: Run smoke pipeline tests
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

### 制御されたテスト銘柄

`python tools/generate_test_symbols.py` を実行すると、以下のテストデータが生成されます。

- 出力先: `data_cache/test_symbols/`, `data_cache/rolling/`
- 銘柄構成:
  - `FAIL_ALL_00`〜`04`: 全システムでフィルターを通過しない 5 件
  - `FILTER_ONLY_S{1..6}_00`〜`02`: フィルターのみ通過する 18 件
  - `SETUP_PASS_S{1..6}_00`〜`14`: セットアップを通過しランキング対象になる 90 件
  - `SPY`: ヘッジシステム用のアンカー銘柄 (rolling のみ)
- すべて日次インジケーター付きで保存され、Rolling キャッシュも同時生成

このセットを利用すると、TRDlist と Entry がそれぞれ 10 件に確定し、診断 JSON で各システムの件数が容易に検証できます。

### 実データ運用時の最小構成

- SPY など最低限の銘柄を `data_cache/rolling/` に揃える
- 直近 300 営業日分の Rolling キャッシュ
- `--skip-external` オプションで外部 API を抑制

### キャッシュ更新コマンド

```bash
# Rolling キャッシュをまとめて再構築
python scripts/build_rolling_with_indicators.py --workers 4

# 個別銘柄のみ更新
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

- 単体テストは `pytest -q`
- 制御された統合テストは `--test-mode test_symbols`
- 超高速スモークは `--test-mode mini`
- 本番同等テストは `--test-mode quick` に `--parallel --save-csv`
- CI/CD では制御データ生成と mini モードの組み合わせが推奨

テストモードを切り替えるだけで、再現性と実行時間のバランスを柔軟に調整できます。
