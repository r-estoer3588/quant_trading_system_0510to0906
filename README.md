# Quant Trading System (Streamlit)

Streamlit ベースのアプリで 7 つの売買システムを可視化・バックテストします。

## セットアップ

1. 仮想環境を作成し依存関係をインストール:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. `.env` を用意し `EODHD_API_KEY` に加え、Alpaca 連携を行う場合は
   `ALPACA_API_KEY` と `ALPACA_SECRET_KEY` を設定します。

### 主要な環境変数

- `EODHD_API_KEY`: EOD Historical Data API キー（必須）
- `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`: Alpaca ブローカー連携用
- `SLACK_WEBHOOK_URL` または `SLACK_BOT_TOKEN`+`SLACK_CHANNEL`: Slack 通知設定
- `DISCORD_WEBHOOK_URL`: Discord 通知設定

### ログ・進捗関連の環境変数

- `COMPACT_TODAY_LOGS`: 当日パイプラインの詳細ログを抑制（デフォルト: false）
- `ENABLE_PROGRESS_EVENTS`: 進捗イベント出力を有効化（デフォルト: false）
- `TODAY_SIGNALS_LOG_MODE`: ログファイル名形式（`single` または `dated`）
- `ROLLING_ISSUES_VERBOSE_HEAD`: Rolling キャッシュ問題の詳細表示件数
- `RUN_PLANNED_EXITS`: 自動手仕舞い実行モード（`off`/`open`/`close`/`auto`）

## 実行例

### 基本実行

- UI: `streamlit run app_integrated.py`
- Alpaca ダッシュボード: `streamlit run app_alpaca_dashboard.py`
- 当日パイプライン: `python scripts/run_all_systems_today.py --parallel --save-csv`

### ログ最適化機能

当日パイプラインのログ出力を制御する環境変数：

```bash
# コンパクトログモード（詳細ログをDEBUGレベルに変更）
COMPACT_TODAY_LOGS=1 python scripts/run_all_systems_today.py

# 進捗イベント出力有効化（UIでリアルタイム監視可能）
ENABLE_PROGRESS_EVENTS=1 python scripts/run_all_systems_today.py

# Rolling キャッシュ問題の詳細表示件数制限
ROLLING_ISSUES_VERBOSE_HEAD=5 python scripts/run_all_systems_today.py

# 全て同時使用
COMPACT_TODAY_LOGS=1 ENABLE_PROGRESS_EVENTS=1 ROLLING_ISSUES_VERBOSE_HEAD=3 \
    python scripts/run_all_systems_today.py --parallel --save-csv
```

### その他の実行例

- 日次キャッシュ: `python scripts/cache_daily_data.py`
  - 並列度調整: `--max-workers 20` (デフォルト: 20)
  - API 取得並列度: `--fetch-workers 1` (デフォルト: 1、順次実行でレート制限遵守)
  - 保存並列度: `--save-workers` (デフォルト: max_workers)
  - スロットリング: `--throttle-seconds 0.0667` (デフォルト: 0.0667 秒、約 15req/sec、公式制限 1000req/min 以内に収まるよう調整)
  - 進捗表示間隔: `--progress-interval 300` (デフォルト: 300 件、指定件数ごとに進捗を表示)
  - 注意: 引数を指定しない `python scripts/cache_daily_data.py` の既定実行は
    全銘柄の全ヒストリカルデータを取得する（フル取得）。当日の一括（Bulk）更新を
    実行したい場合は `--bulk-today` を指定するか、`scripts/update_from_bulk_last_day.py`
    を直接実行してください。
- 簡易スケジューラ: `python -m schedulers.runner`

### リアルタイム進捗監視

`ENABLE_PROGRESS_EVENTS=1` で当日パイプラインを実行中、同時に Streamlit UI で進捗をリアルタイム監視できます：

1. ターミナル 1: 当日パイプライン実行

   ```bash
   ENABLE_PROGRESS_EVENTS=1 python scripts/run_all_systems_today.py --parallel
   ```

2. ターミナル 2: UI 起動

   ```bash
   streamlit run app_integrated.py
   ```

3. UI の「当日シグナル」タブで「進捗ログを表示」をチェック
   - `logs/progress_today.jsonl` を 1 秒間隔でポーリング
   - システム実行開始、配分処理、通知完了などの主要イベントをリアルタイム表示
   - 各システムの候補数、エントリ数、現在ポジション数などの詳細情報も含む

### ログ最適化の効果

- **通常モード**: 全詳細ログが INFO レベルで出力され、大量のメッセージが表示される
- **コンパクトモード (`COMPACT_TODAY_LOGS=1`)**:
  - システム内訳や進捗詳細が DEBUG レベルに変更
  - レート制限により同種メッセージが適切な間隔（2-10 秒）で出力
  - 重要な情報（エラー、最終結果など）は引き続き INFO レベルで表示
  - ログファイルサイズが約 60-80%削減（実測値）

## テスト

```bash
pytest -q
```

## 設定

優先順位は **JSON > YAML > .env**（`config/settings.py` 実装に準拠）。
推奨: `config.yaml` をベースに、秘匿値は `.env`、上書きは JSON で。

## ログ運用

`logging_utils` にてローテーション設定。容量上限と日次ローテの使い分けを明記し、
古いログのクリーンアップ方針を追加。

## ディレクトリ構成

- `app_integrated.py` – 統合 UI
- `strategies/` – 戦略ラッパ
- `core/` – 各システム純ロジック
- `common/` – 共通ユーティリティ
- `config/` – 設定
- `docs/` – ドキュメント
- `tests/` – テスト

## データストレージ最適化

### デュアルフォーマット対応

- `data_cache/rolling/`：CSV + Feather 同時保存
- パフォーマンス：Feather 優先読み取り、CSV 自動フォールバック
- ファイルサイズ：Feather は CSV の約 74%（重複列削除効果）
- 対応範囲：6,200+銘柄すべて

### 重複列クリーンアップ

- 削除対象：`open/Open/OPEN` 等の冗長列（約 7 列）
- 優先順位：PascalCase > ALL_CAPS > lowercase
- データ削減：列数を約 40%削減（58→35 列）

## キャッシュ階層（base / rolling / full_backup）

- base: 指標付与済みの長期データ（バックテスト・分析の既定）。
  - 読み込みが最速。欠損時は内部で再構築されます（full_backup/rolling から）。
- rolling: 直近 N 営業日（既定 300）の軽量データ（当日シグナル抽出用）。
  - 無ければ base から必要分を生成して保存します。
- full_backup: 取得元そのままの長期バックアップ（原本）。
  - 通常は読みません。復旧や base 再構築のソースとしてのみ使用します。

解決ポリシー（SPY 含む）:

- backtest: base → full_backup（rolling は参照しない）。
- today: rolling → base → full_backup（rolling が無ければ base から生成して保存）。

補足:

- 旧来の `data_cache/` 直下ファイルは参照しません（移行済みを前提）。
- SPY フル履歴の復旧は `recover_spy_cache.py` が `data_cache/full_backup/` のみへ保存します。

## 貢献ガイド

- コミットメッセージは命令形・現在形で 72 文字以内。
- 変更後は `pytest -q` を実行してテストを確認してください。
