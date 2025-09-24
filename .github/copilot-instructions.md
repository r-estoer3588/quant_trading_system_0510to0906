# Copilot Instructions for This Repo

このドキュメントは、このリポジトリで AI コーディングエージェントが正しく・安全に・効率よく作業するための詳細ガイドです。日々の運用（当日シグナル算出）、バックテスト、データキャッシュ、通知/発注連携、品質チェックの要点を体系化しています。開発者は `AGENTS.md` / `README.md` も併読してください。

## 目的と対象

- 目的: このプロジェクト固有の作法・構成・禁止事項を一元化し、AI 編集の安全性と速度を高める。
- 対象: AI コーディングエージェント、レビュー担当、運用担当（当日パイプライン/通知/発注）。

## 全体アーキテクチャ

- UI/入口: `app_integrated.py`（Streamlit）。タブは `common/ui_tabs.py`、部品は `common/ui_components.py`。
- 戦略分割: 純ロジックは `core/system{1..7}.py`、戦略ラッパは `strategies/system{1..7}_strategy.py`。共通実行は `common/integrated_backtest.py`。
- 当日パイプライン: `scripts/run_all_systems_today.py` が候補抽出 → 配分 → 通知 →CSV 出力 →（任意）発注までを統合。UI からもここを呼び出し。
- 配分/連携:
  - 配分確定: `core/final_allocation.py::finalize_allocation()`
  - 通知: `common/notifier.py` / `tools/notify_metrics.py` / `tools/notify_signals.py`
  - ブローカー: Alpaca（`common/broker_alpaca.py`, `common/alpaca_order.py`）
- データ層（3 層キャッシュ）: `data_cache/`
  - `full_backup/`: 原本（日次株価など）
  - `base/`: 指標付与済み（列名正規化、Adjusted Close 優先）
  - `rolling/`: 直近 N 営業日（`base_lookback_days + buffer_days` を維持）
  - I/O は必ず `common/cache_manager.py::CacheManager` / `load_base_cache()` を経由。CSV の直読禁止。

## システム別の要点

- ロング: System1/3/4/5、ショート: System2/6/7。System7 は SPY 固定（アンカー用途）。
- スコア並び: システムごとに方向が異なる（例: s1=ROC200 降順、s4=RSI4 昇順）。`common/today_signals.py::_score_from_candidate` / `_asc_by_score_key` を参照。
- Entry/Exit:
  - 各戦略は `compute_entry` / `compute_exit` を実装可能。
  - 未実装時は ATR ベースのフォールバック（long はエントリ −5×ATR、short は +5×ATR 近辺）。実装は `common/integrated_backtest.py`。

## キャッシュ運用（厳守）

- 解決順:
  - backtest: `base → full_backup`（`rolling` は使用しない）
  - today: `rolling → base → full_backup`（`rolling` 無ければ `base` から生成し保存）
- 指標付与: `compute_base_indicators()` が列名正規化・Adjusted Close 優先。High/Low/Close が欠損ならスキップ。
- Rolling 保守: `prune_rolling_if_needed()` を用い、`rolling/_meta.json` のメタを維持。アンカーは既定 SPY。
- 禁止事項: `data_cache/` 配下 CSV を直接読まない（常に `CacheManager.read()`）。

## 当日パイプライン（scripts/run_all_systems_today.py）

- 役割: 全システムの候補抽出 → エントリ判定 → 配分確定 → 結果保存/通知 →（任意）発注。
- 出力:
  - `results_csv/` にシステム別 CSV、連結結果、`daily_metrics.csv`（メトリクス追記: prefilter/candidates/entries など）
  - ログは `logs/`（`TODAY_SIGNALS_LOG_MODE` によりファイル切替）
- 主要フラグ（例）:
  - `--parallel`（並列実行）/ `--save-csv`（結果 CSV 保存）
  - 発注/ドライラン関連はブローカー設定と併用
- 並び順・結合: `common/today_signals.py` / `common/signal_merge.py` を参照。

## 実行コマンド（PowerShell）

```powershell
# 依存関係（開発ツールは requirements-dev.txt）
pip install -r requirements.txt

# UI 起動
streamlit run app_integrated.py

# 当日パイプライン（並列＆CSV 保存）
python scripts/run_all_systems_today.py --parallel --save-csv

# 日次キャッシュ更新（EODHD_API_KEY 必須）
python scripts/cache_daily_data.py

# テスト（決定性・オフライン）
pytest -q
pytest tests/test_headless_app.py tests/test_utils.py -q

# pre-commit（ruff/black/isort/mypy をコミット時に実行）
pre-commit install
```

## 設定/環境変数（config/settings.py::get_settings()）

- 優先度: JSON > YAML > .env（`get_settings(create_dirs=True)` で必要ディレクトリ自動作成）
- 資格情報: `EODHD_API_KEY`、Alpaca: `ALPACA_API_KEY`/`ALPACA_SECRET_KEY`、通知: `SLACK_WEBHOOK_URL` または `SLACK_BOT_TOKEN`+`SLACK_CHANNEL(_ID)`、`DISCORD_WEBHOOK_URL`
- ログ: `TODAY_SIGNALS_LOG_MODE=single|dated`（当日パイプラインのログファイル切替）
- 自動手仕舞い計画: `RUN_PLANNED_EXITS=off|open|close|auto`（`schedulers/next_day_exits.py` を当日パイプラインから実行）
- タイムゾーン: ログ/通知は JST。スケジューラ既定は `America/New_York`

## 開発規約・スタイル

- PEP8 厳守。関数/ファイルは snake_case、クラスは PascalCase。日本語文字列は UTF-8 のまま。
- インポート順序: 標準/サードパーティ/ローカル。
- `get_settings()` が提供するパスを尊重し、書き込みは `results_csv/`・`logs/`・`data_cache/` のみ。
- 配分ロジック（DEFAULT_ALLOCATIONS 等）と System7 の SPY アンカーは変更しない。

## テスト/CI/型

- テスト: すべてオフライン・決定性。外部 I/O は禁止。`common/testing.py::set_test_determinism` を参照。
- CI: `ruff/black/isort/mypy/bandit/pip-audit/pytest` を Python 3.10–3.12 で実行。
- 型: 段階的に mypy を厳格化。ネットワーク依存を排除。

## 運用シナリオ別クイック手順（Runbook）

- 初回セットアップ
  1. `pip install -r requirements.txt`
  2. `.env` に `EODHD_API_KEY`, Alpaca, Slack/Discord を設定
  3. `python scripts/cache_daily_data.py` で初期キャッシュ作成
  4. `streamlit run app_integrated.py` で UI 動作確認
- 当日朝の流れ
  1. `python scripts/cache_daily_data.py`
  2. `python scripts/run_all_systems_today.py --parallel --save-csv`
  3. `results_csv/` と 通知内容を確認 → 必要に応じて発注
- 障害時リカバリ
  - キャッシュ不整合: `common/cache_manager.py::prune_rolling_if_needed()` の呼び出し経路を実行（当日パイプラインで自動）。
  - SPY 原本復旧: `recover_spy_cache.py`（`data_cache/full_backup/`）。
  - MCP/Codacy 不可: VS Code の Copilot MCP 設定（Enable MCP servers）または CI/WSL で解析。

## 変更時のレビューチェックリスト

- 公開 API/CLI を壊していないか（フラグ追加は既定値で互換維持）。
- キャッシュ解決順（backtest: base→full_backup / today: rolling→base→full_backup）は守れているか。
- CSV 直読をしていないか（常に `CacheManager` 経由か）。
- System7 の SPY アンカーや配分ロジックを変更していないか。
- 新規 I/O は `get_settings()` 配下 (`results_csv/`, `logs/`, `data_cache/`) のみか。
- テストはオフライン・決定性を保っているか（ネットワーク呼び出しなし）。
- 型・品質: `mypy/ruff/black/isort` 前提の違反がないか。

## AI 編集ガイド（重要）

- 既存の public API と CLI フラグを壊さない。ファイル名/関数名のリネームは最小限。
- 不要な再フォーマットや同時大量変更を避ける。変更は論理的に小さく。
- 新規ファイルは `get_settings()` 管理下のパスにのみ作成。
- 変更後は可能なら対象テストを実行。実行できない場合は簡潔な再現手順を記載。
- 静的解析: Codacy CLI（MCP）を実行。Windows では WSL が必要。利用不能時はメンテへ通知。

## トラブルシューティング

- Codacy CLI（MCP）
  - Windows 単体では未対応。WSL または CI で実行してください。
  - MCP が利用不可: VS Code の Copilot MCP 設定（Enable MCP servers）を確認。
- キャッシュ不整合
  - `prune_rolling_if_needed()` で rolling 長を是正。メタは `rolling/_meta.json`。
  - SPY の原本復旧: `recover_spy_cache.py`。

不明点や小規模 PR の提案が必要な場合は、対象ファイルと意図を記載の上、相談してください。
