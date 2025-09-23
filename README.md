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

## 実行例

- UI: `streamlit run app_integrated.py`
- Alpaca ダッシュボード: `streamlit run app_alpaca_dashboard.py`
- 日次キャッシュ: `python scripts/cache_daily_data.py`
  - 並列度調整: `--max-workers 20` (デフォルト: 20)
  - API 取得並列度: `--fetch-workers 1` (デフォルト: 1、順次実行でレート制限遵守)
  - 保存並列度: `--save-workers` (デフォルト: max_workers)
  - スロットリング: `--throttle-seconds 0.0667` (デフォルト: 0.0667 秒、約 15req/sec、公式制限 1000req/min 以内に収まるよう調整)
  - 進捗表示間隔: `--progress-interval 600` (デフォルト: 600 件、指定件数ごとに進捗を表示)
  - 注意: 引数を指定しない `python scripts/cache_daily_data.py` の既定実行は
    全銘柄の全ヒストリカルデータを取得する（フル取得）。当日の一括（Bulk）更新を
    実行したい場合は `--bulk-today` を指定するか、`scripts/update_from_bulk_last_day.py`
    を直接実行してください。
- 簡易スケジューラ: `python -m schedulers.runner`

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
