## ドキュメント総覧（System1〜7）

### 共通前提

### 設定の優先順位

### バックテスト前提

### KPI

### システム構成と資産配分

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
