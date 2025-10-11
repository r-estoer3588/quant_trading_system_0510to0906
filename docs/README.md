# Quant Trading System - ドキュメント総覧

このドキュメントは、7 つの売買システムを統合したクオンツトレーディングシステムの包括的なガイドです。

## 📚 ドキュメント構成

### 🚀 [クイックスタート](#quick-start)

- [セットアップ](../README.md#セットアップ) - 初回環境構築
- [基本実行](../README.md#実行例) - UI 起動と基本操作
- [テスト実行](./testing.md) - システム動作確認

### 📊 [システム概要](#trading-systems)

- [システム構成と資産配分](#システム構成と資産配分)
- [各システム詳細](./systems/) - System1-7 の個別仕様
- [パフォーマンス指標](#kpi)

### 🔧 [技術文書](#technical-docs)

- [キャッシュインデックス要件](./technical/cache_index_requirements.md) - Feather 形式の制約と日付インデックス変換
- [候補数ゼロガイド](./technical/zero_candidates_guide.md) - System6 等で候補が出ない理由(正常動作)
- [環境変数一覧](./technical/environment_variables.md) - 既定値と用途
- [SPY/取引日ユーティリティ](./technical/spy_utils.md) - 営業日ヘルパの仕様

### 🏃 [運用ガイド](#operations)

- [自動実行設定](./schedule_quick_start.md) - Windows Task Scheduler
- [通知設定](./NOTIFICATIONS.md) - Slack/Discord 連携
- [UI メトリクス](./today_signals_ui_metrics.md) - ダッシュボード

### 🔗 関連リンク

- [メイン README](../README.md) - プロジェクト全体概要
- [変更履歴](../CHANGELOG.md) - リリースノート
- [GitHub Instructions](../.github/copilot-instructions.md) - AI 開発ガイド

---

## システム構成と資産配分

4 つの買いシステム

- システム 1 ーロング・トレンド・ハイ・モメンタム（トレード資産の 25%を配分）
- システム 4 ーロング・トレンド・ロー・ボラティリティ（トレード資産の 25%を配分）
- システム 3 ーロング・ミーン・リバージョン・セルオフ（トレード資産の 25%を配分）
- システム 5 ーロング・ミーン・リバージョン・ハイ ADX・リバーサル（トレード資産の 25%を配分）

3 つの売りシステム

- システム 2 ーショート RSI スラスト（トレード資産の 40%を配分）
- システム 6 ーショート・ミーン・リバージョン・ハイ・シックスデイサージ（トレード資産の 40%を配分）
- システム 7 ーカタストロフィーヘッジ（トレード資産の 20%を配分）

### モニタリング（daily_metrics.csv）

- 出力先: `results_csv/daily_metrics.csv`
- 生成タイミング: `scripts/run_all_systems_today.py` 実行時（当日シグナル抽出の終盤）
- カラム:
  - `date`: NYSE 最新営業日
  - `system`: `system1`〜`system7`
  - `prefilter_pass`: 事前フィルター通過銘柄数（system7 は SPY の有無で 1/0）
  - `candidates`: 当日候補数（最終スコアリング前のシステム別集計）
- 用途: 事前フィルターの通過数と候補数の推移を日次で可視化し、データ品質やシグナル強度の変動を監視する。

#### UI: Metrics タブ

- `app_integrated.py` のタブに `Metrics` を追加。`results_csv/daily_metrics.csv` を読み込み、システム別に `prefilter_pass` と `candidates` の推移をライン／バーで表示できる。

#### 検証レポート（任意）

- `tools/build_metrics_report.py` が最新日のメトリクスと各システムのシグナル CSV（`signals_systemX_YYYY-MM-DD.csv`）を突き合わせ、`results_csv/daily_metrics_report.csv` を生成する。件数の齟齬チェックやサンプル銘柄の目視確認に使う。
