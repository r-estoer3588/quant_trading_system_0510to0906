# Copilot Instructions for This Repo

AI エージェントが初手から迷わないための要点を 1 枚に凝縮しました。詳細は `AGENTS.md` と `README.md` を併読してください。

## 全体像（System1–7 の流れ）

- UI: `app_integrated.py` が Streamlit で System1–7 を統合表示。各タブは `common/ui_tabs.py`・部品は `common/ui_components.py`。
- 戦略分割: 純ロジックは `core/system{1..7}.py`、戦略ラッパは `strategies/system{1..7}_strategy.py`。準備/候補抽出は `StrategyProtocol`（`common/integrated_backtest.py`）。
- データ層: EODHD 日足を三層キャッシュ（`data_cache/full_backup` 原本・`base` 指標付与済み・`rolling` 直近 N 営業日）。読み取りは必ず `common/cache_manager.py::CacheManager` と `load_base_cache()` 経由。
- 今日のシグナル: `common/today_signals.py`（long: system1/3/4/5、short: system2/6/7）。UI は `scripts/run_all_systems_today.py` を経由して実行（`common/ui_tabs.py` 参照）。
- 連携: 通知は `common/notifier.py`、ブローカーは Alpaca（`common/broker_alpaca.py`, `common/alpaca_order.py`）。

## プロジェクト固有の作法

- 命名/スタイル: PEP 8。関数/ファイルは snake_case、クラスは PascalCase。日本語テキストは UTF-8 のまま保持。
- キャッシュ解決順（厳守）:
  - backtest: `base → full_backup`（`rolling` は参照しない）。
  - today: `rolling → base → full_backup`（`rolling` 無ければ `base` から生成・保存）。
- 直読禁止: `data_cache/` 直下の CSV を直接読まない。常に `CacheManager.read()` を使う。
- 特例: System1 は ROC200 ランキングを一部特経路（`common/ui_components.py::generate_roc200_ranking_system1`）。System7 は SPY 固定（`integrated_backtest.prepare_all_systems`）。

## 重要ポイント（実装の勘所）

- `compute_entry/compute_exit`: 戦略で上書き可能。未実装時は ATR を使うフォールバックが使われ、long はエントリ −5×ATR、short はエントリ＋ 5×ATR 付近を基準（`common/integrated_backtest.py`）。
- 今日のシグナル並び順: スコアキーはシステム別に優先順位あり（例: s1=ROC200 降順, s4=RSI4 昇順）。`_score_from_candidate` と `_asc_by_score_key` を参照（`common/today_signals.py`）。
- 指標付与: `compute_base_indicators()` は列名を正規化し、Close は adjusted を優先。必須列 High/Low/Close が無い場合は計算をスキップ（`common/cache_manager.py`）。
- Rolling 保守: 長さは `base_lookback_days + buffer_days`。古い行の剪定は `prune_rolling_if_needed()`（メタは `rolling/_meta.json`、アンカーは既定 SPY）。

## よく使うコマンド

- 依存関係: `pip install -r requirements.txt`（開発ツールは `requirements-dev.txt`）。
- UI 起動: `streamlit run app_integrated.py`
- 日次キャッシュ更新: `python scripts/cache_daily_data.py`（`EODHD_API_KEY` 必須）。
- テスト: `pytest -q`（ネットワーク禁止・決定性重視）。例: `pytest tests/test_headless_app.py tests/test_utils.py -q`
- 品質: `pre-commit install`; 以降 `ruff/black/isort/mypy` がコミット時に走る。

## 設定/環境変数（`config/settings.py::get_settings()`）

- 優先度: JSON > YAML > .env。ディレクトリは `get_settings(create_dirs=True)` で自動作成。
- 必須/推奨: `EODHD_API_KEY`、Alpaca: `ALPACA_API_KEY`/`ALPACA_SECRET_KEY`、通知: `SLACK_WEBHOOK_URL` または `SLACK_BOT_TOKEN`+`SLACK_CHANNEL(_ID)`、`DISCORD_WEBHOOK_URL`。
- 実行ログ: `TODAY_SIGNALS_LOG_MODE=dated` で `today_signals_YYYYMMDD_HHMM.log` を使用（`scripts/run_all_systems_today.py`）。
- タイムゾーン: ログ/通知は JST。スケジューラ既定は `America/New_York`。

## 変更時に見る場所（ファイルの勘所）

- データ系: `common/cache_manager.py`（IO/rolling/prune）、`common/data_loader.py`、`common/utils.py::get_cached_data`。
- 新戦略: `core/systemX.py` → `strategies/systemX_strategy.py`（`prepare_data`/`generate_candidates`/サイズ計算）。UI の分岐は `common/ui_components.py`。
- バックテスト統合: `common/integrated_backtest.py::run_integrated_backtest` と `DEFAULT_ALLOCATIONS`（long 4 系列=各 25%、short 系列=2/6=40%、7=20%）。
- 今日の実行: `common/today_signals.py` と `scripts/run_all_systems_today.py`（UI は `common/ui_tabs.py` 経由）。

## レビュー観点（壊しやすい箇所）

- 書き込み先は `get_settings()` 配下（`results_csv/`・`logs/`・`data_cache/`）に限定。
- 長短バケット配分・System7 の SPY 固定・long/short 集合を崩さない。
- 日本語の文言（コメント/ログ/通知）は変形しない。
- テストはオフライン・決定性（`common/testing.py::set_test_determinism`）。

不足や不明点は遠慮なく聞いてください。必要に応じてこの文書を追補します。
