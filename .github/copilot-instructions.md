# Copilot Instructions for This Repo

このリポジトリで AI コーディングエージェントがすぐに生産的に動けるよう、最小限で実務的な指針をまとめます。詳細は `AGENTS.md` と `README.md` も参照してください。

## 全体像（何がどこで動くか）

- UI: `app_integrated.py` が Streamlit で System1–7 を統合表示。各タブ/UI は `common/ui_tabs.py`・`common/ui_components.py` に実装。
- 戦略境界: 純ロジックは `core/system{1..7}.py`、戦略ラッパは `strategies/system{1..7}_strategy.py`。バックテスト/準備フローは `common/integrated_backtest.py`（`StrategyProtocol` に準拠）。
- データ: EODHD 日足をローカルキャッシュ。キャッシュは三層 `data_cache/full_backup`（原本）・`data_cache/base`（指標付与済み長期）・`data_cache/rolling`（直近 N 営業日）。読み取りは原則 `common/cache_manager.py` の `CacheManager` と `load_base_cache()` 経由。
- 今日のシグナル: `common/today_signals.py`（long: system1/3/4/5、short: system2/6/7）。UI からは `scripts/run_all_systems_today.py::compute_today_signals` を呼ぶ経路あり（`common/ui_tabs.py` 参照）。
- 通知/連携: Slack/Discord は `common/notifier.py`。ブローカー連携は Alpaca（`common/broker_alpaca.py`, `common/alpaca_order.py`）。

## 重要な規約とパターン

- 命名/スタイル: PEP8、関数・ファイルは snake_case、クラスは PascalCase。型ヒントは公開 API で推奨。日本語コメント/文字列は UTF-8 のまま保持（置換や削除をしない）。
- キャッシュの原則: 直接 `data_cache/` 直下の CSV を読む実装は禁止。`CacheManager.read()` と `load_base_cache(symbol, rebuild_if_missing=True)` を利用。再構築時は `full_backup→rolling` を参照して保存。
- リゾルバ方針（既定）:
  - backtest: base → full_backup（rolling は見ない）
  - today: rolling → base → full_backup（rolling 無ければ base から生成）
- 戦略フック: `StrategyProtocol` のほか、必要に応じ `compute_entry/compute_exit` を実装するとエントリ/イグジットの挙動を上書き可能（`common/integrated_backtest.py` 参照）。
- UI 統合: System1 は ROC200 ランキングの都合で一部特別経路（`common/ui_components.py` で `generate_roc200_ranking_system1` を直参照）。System7 は SPY 固定（`integrated_backtest.prepare_all_systems` 内の分岐）。

## 開発フロー（よく使うコマンド）

- 依存関係: `pip install -r requirements.txt`（開発ツールは `requirements-dev.txt`）。
- UI 起動: `streamlit run app_integrated.py`
- データ更新: `python scripts/cache_daily_data.py`（`EODHD_API_KEY` 必須）。
- テスト: `pytest -q`（決定性重視・ネットワーク呼び出し禁止）。重点テスト例:
  - `pre-commit run --files tests/test_headless_app.py tests/test_utils.py tests/app_smoke.py`
  - `pytest tests/test_headless_app.py tests/test_utils.py -q`
- 型/品質: `mypy .` / `ruff` / `black` / `isort` は pre-commit 連携。`pre-commit install` 推奨。

## 設定と環境変数

- 入口: `config/settings.py::get_settings()` が `.env`・YAML・JSON を統合（優先度: JSON > YAML > .env）。主要ディレクトリは `get_settings(create_dirs=True)` で作成。
- 必須・推奨環境変数:
  - `EODHD_API_KEY`（EODHD 取得）
  - Alpaca 連携: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`
  - 通知（任意）: `SLACK_WEBHOOK_URL` もしくは `SLACK_BOT_TOKEN` + `SLACK_CHANNEL(_ID)`、`DISCORD_WEBHOOK_URL`
  - テストや CI で通知抑止: `DISABLE_NOTIFICATIONS=1`
- タイムゾーン: ログ/通知は JST（`notifier.py`）。スケジューラ既定は `America/New_York`（`settings.scheduler.timezone`）。

## 具体例（変更時に見るべき場所）

- データ系を触る: `common/cache_manager.py`（読み書き・prune・rolling 窓の長さ）、`common/data_loader.py`、`common/utils.py::get_cached_data`。
- 新戦略/ロジックの追加: `core/systemX.py` に純ロジック追加 → `strategies/systemX_strategy.py` でラップ（`prepare_data`/`generate_candidates`/サイズ計算など）。UI は `common/ui_components.py` で System 分岐に追随。
- 統合バックテスト: `common/integrated_backtest.py::run_integrated_backtest` と `prepare_all_systems`。配分は `DEFAULT_ALLOCATIONS` と UI 設定（`config/settings.py::UIConfig`）。
- 今日のシグナル生成: `common/today_signals.py` と `scripts/run_all_systems_today.py::compute_today_signals`。UI 経由呼び出しは `common/ui_tabs.py`。
- 通知: `common/notifier.py::Notifier.send(_with_mention)`。Slack API を使う場合は `slack_sdk` が必要（未導入なら Webhook のみ）。

## レビューポイント（このプロジェクト特有）

- I/O パスは `get_settings()` 由来のディレクトリ配下に限定。`results_csv/`・`logs/`・`data_cache/` 以外へ書かない。
- 大域の動作条件（長短の配分や SPY 依存、System7 の SPY 固定）を壊さない。`today_signals` の long/short 集合や `SYSTEM_POSITION` を要確認。
- 日本語の文言は壊さない（コメント/ログ/通知テキスト含む）。
- テストではネットワークアクセス禁止。キャッシュ済みデータを利用し、乱数/日付は固定（`common/testing.py::set_test_determinism`）。

---

不足や曖昧な点があれば教えてください。作業内容に合わせて追補します。
