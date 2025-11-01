# Copilot instructions — quick reference (repository-specific)

目的: このリポジトリで AI エージェントが安全かつ即戦力で作業するための最小限かつ実用的なルール集。まずは必ず `docs/README.md` を確認して全体像（設計/運用/コマンド）を掴んでください。

## アーキテクチャ概要（把握のカギ）

- 7 システム構成：ロング(1,3,4,5)／ショート(2,6,7)。System7 は SPY ショートのヘッジ専用（特例）。
- UI: `apps/app_integrated.py`（統合タブ）/ `apps/app_today_signals.py`。
- 純ロジック: `core/systemX.py`、戦略ラッパ: `strategies/systemX_strategy.py`、UI タブ: `apps/systems/app_systemX.py`。
- 日次パイプライン: `scripts/run_all_systems_today.py`
  - フロー: symbols → load/cache → indicators → filters/setup → ranking/signals → allocation → save/notify。
- 最終配分: `core/final_allocation.py::finalize_allocation()`（slot/capital 両モード）。API 契約は変更しない。

## 必須ルール（破らない）

- キャッシュ I/O は必ず `common/cache_manager.py::CacheManager` 経由。`data_cache/`直参照・直接書込み禁止。
- `core/system7.py` は SPY 固定のヘッジ専用。ロジック変更や SPY 以外の割当は禁止。
- 設定は `config/settings.py::get_settings()`、環境は `config/environment.py::get_env_config()` を経由（`os.environ.get` 直読禁止）。
- テストで外部 API 呼び出し禁止。キャッシュデータ使用かモック/フィクスチャで代替。

## 開発ワークフロー（PowerShell 例）

- 依存: `pip install -r requirements.txt`
- UI: `python -m streamlit run apps/app_integrated.py`
- 当日パイプライン: `python scripts/run_all_systems_today.py --parallel --save-csv`
- 高速検証: `python scripts/run_all_systems_today.py --test-mode mini --skip-external --benchmark`
- 制御テスト: `python scripts/run_controlled_tests.py`（または `pytest -q tests/test_systems_controlled_all.py`）
- テスト: `pytest -q`（例: `pytest tests/test_system3.py::test_entry_rules`）
- Lint/Format: `ruff check --fix .` → `black .` → `isort .`（pre-commit 推奨）

## キャッシュ層とデータ流儀

- 層: `full_backup`(原本) → `base`(指標付与済/既定) → `rolling`(直近 N 日/当日用)。
- 解決順序: today は `rolling→base→full_backup`、backtest は `base→full_backup`。
- 直接 `data_cache/` を触らず、`CacheManager`/`common/cache_io.py` を使う。

## 診断・ログ・環境

- Diagnostics 共通キー: `setup_predicate_count`, `ranked_top_n_count`, `ranking_source`。
- 進捗監視: `ENABLE_PROGRESS_EVENTS=1` で UI から当日処理の進捗を追跡。
- 配分デバッグ: `get_env_config().allocation_debug` を有効化すると詳細ログ。
- スロット重複緩和: `slot_dedup_enabled` と `slot_max_rank_depth`（環境設定）。

## 代表 API と使用例

- 最終配分（契約厳守）: `core/final_allocation.py::finalize_allocation(per_system, strategies=?, positions=?, ...) -> (df, summary)`
  - 例: `--test-mode mini` で得たシステム別候補を `per_system={"system1": df1, ...}` として渡す。
- 指標/前処理: `common/indicators_precompute.py` / 利用は `common/indicator_access.py`。

## UI/E2E（自動スクショ）

- スクショ自動化: `tools/run_and_snapshot.ps1`（内部で Playwright 操作 → 撮影 → スナップショット）。
- 直接実行: `python tools/capture_ui_screenshot.py --url http://localhost:8501 --output results_images/today_signals_complete.png`。

## 実務の落とし穴（必読）

- slot モードで実株数が必要なら `include_trade_management=True` を指定（UI 連携は指定済み）。未指定だと shares 未算出で発注不可。
- `strategies` を渡さないと資本配分計算が機能しない場合がある。厳格運用は `ALLOCATION_REQUIRE_STRATEGIES=1` で強制。
- 現有ポジション集計は `symbol_system_map` の主系統（primary）に依存。未提供時は `common.symbol_map::load_symbol_system_map()` フォールバック。

## VS Code タスク（時短）

- VS Code から「Tasks: Run Task」で実行: “Run All Systems Today”, “Quick Test Run”, “Start Streamlit UI”。

## PR に必ず書くこと

1. 影響する入力/キャッシュ（例: `data_cache/rolling/system3.feather`） 2) 検証手順（`--test-mode mini` など具体コマンド） 3) UI 変更ならスクショ（`results_images/`）。

ヒント: 再現性が不安定なときは `common/testing.py::set_test_determinism()` を冒頭で呼び、まず `mini` / 制御テストで差分ゼロを確認してから本番相当へ広げてください。
