# Copilot instructions — 速習版（このリポジトリ専用）

このリポジトリで AI エージェントが迷わないようにするための短い案内書です。まず `docs/README.md` で文脈をつかんでからここを読んでください。

## 全体像

7 つの売買システムで構成し、System1/3/4/5 が買い、System2/6 が売り、System7 が SPY の固定ヘッジです。日次パイプラインは `scripts/run_all_systems_today.py` が担い、銘柄取得 → キャッシュ読込 → 指標計算 → フィルタ → ランキングとシグナル → `core/final_allocation.py::finalize_allocation()` までをひとつに束ねます。戻り値 `(df, summary)` の契約は変えないでください。

## レイヤーと主担当

純ロジックは `core/systemX.py`、戦略ラッパは `strategies/systemX_strategy.py`、UI は `apps/app_integrated.py` と `apps/systems/app_systemX.py` が担います。横断ユーティリティは `common/` にあり、並列実行や計測は `common/performance_optimization.py`、診断と通知は `common/alert_framework.py` が受け持ちます。各ファイル冒頭の Context Note を先に読み、禁止事項や前提を踏み外さないようにしましょう。

## データとキャッシュ

キャッシュ層は `rolling` → `base` → `full_backup` の順で解決し、I/O は `common/cache_manager.py::CacheManager` を経由します。候補データを日付単位に正規化する三段構えの仕組み（Option-B）は `common/system_candidates_utils.py` に集約されており、`normalize_dataframe_to_by_date()` や `set_diagnostics_after_ranking()` を使うと診断フォーマットを保てます。

## フラグと環境設定

設定値は `config/settings.py::get_settings()`、環境別の挙動は `config/environment.py::get_env_config()` から取得します。System3/5/6 の Option-B 切り替えは `ENABLE_OPTION_B_SYSTEM3` などのフラグで制御し、進捗表示と配分デバッグは `ENABLE_PROGRESS_EVENTS` と `allocation_debug` で有効化します。環境変数を直接読む実装は避けてください。

## 検証ルーチン

当日フローの再現は `python scripts/run_all_systems_today.py --parallel --save-csv`、高速確認は `--test-mode mini --skip-external --benchmark` を追加します。システム別の整合性は `python scripts/run_controlled_tests.py`（内部で `tests/test_systems_controlled_all.py` を呼びます）で確かめられます。時間依存の揺らぎが出たら `common/testing.py::set_test_determinism()` をテスト冒頭で呼んでください。

## 監視と診断

診断キーは `setup_predicate_count`、`ranked_top_n_count`、`ranking_source` が共通に並びます。日次メトリクスは `results_csv/daily_metrics.csv` に蓄積され、UI の Metrics タブがここを参照します。遅延や欠損の監視は `make_freshness_alert()` や `make_latency_alert()` で設定し、イベントは `AlertManager` が返します。

## 注意点

- キャッシュパスを直接触らず `CacheManager` と `common/cache_io.py` の API を使います。
- `core/system7.py` は SPY の固定ヘッジなので銘柄追加や条件変更を行わないでください。
- `finalize_allocation()` で株数を出すときは `include_trade_management=True` を渡し、`strategies` 情報を空にしないでください。

## 参考ファイル

UI の挙動は `apps/app_integrated.py`、配分の仕組みは `core/final_allocation.py`、指標取得は `common/indicator_access.py` が入口になります。詳細仕様は `docs/TECHNICAL_SPECS.md`、AI 運用ルールは `docs/internal/AGENTS.md` にまとまっています。
