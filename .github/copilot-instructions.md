# AI アシスタント共通インストラクション（Copilot / Gemini / Antigravity）

このリポジトリで AI エージェントがすぐに生産的に動けるための要点集。まずは `docs/README.md` を開き、全体像と参照先を掴んでください。

> **重要**: チャット開始時に必ず `.agent/workflows/project-reference.md` を参照してください。

## 大枠の設計（何がどこにあるか）

- 7 システム構成：ロング(1,3,4,5)／ショート(2,6,7)。System7 は SPY ショートのヘッジ専用。
- **Streamlit UI**: `apps/app_integrated.py`（統合ダッシュボード）/ `apps/app_today_signals.py`。
- **Next.js UI**: `apps/dashboards/alpaca-next/`（モダンダッシュボード、port 3000）
- **FastAPI**: `apps/api/main.py`（API バックエンド、port 8000）
- 純ロジック: `core/systemX.py`、戦略ラッパ: `strategies/systemX_strategy.py`、UI タブ: `apps/systems/app_systemX.py`。
- 当日パイプライン: `scripts/run_all_systems_today.py`（symbols → cache/load → indicators → filters/setup → ranking/signals → allocation → save/notify）。
- 最終配分: `core/final_allocation.py::finalize_allocation()`（slot/capital 両モード、API 契約は変更しない）。

### ダッシュボード起動（統合スクリプト）

```powershell
.\Start-Dashboard.ps1  # FastAPI + Next.js 同時起動、ブラウザ自動オープン
```

## 新規実装モジュール（2024 Q4 Workstream #2-5）

### Workstream #2: Option B 本格展開

- **`common/system_candidates_utils.py`**: Phase-A 共通化ユーティリティ
  - `set_diagnostics_after_ranking()`: 診断更新の標準化
  - `normalize_candidates_by_date()`: 候補を {date: {symbol: payload}} 形式に正規化
  - `normalize_dataframe_to_by_date()`: DataFrame を正規化形式に変換
  - `choose_mode_date_for_latest_only()`: 最頻日選択（欠落銘柄耐性）
- **System3/5/6 統合**: `ENABLE_OPTION_B_SYSTEM[3|5|6]` feature flags で段階的ロールアウト

### Workstream #3: パフォーマンス最適化

- **`common/performance_optimization.py`** (420 行)
  - `get_optimal_worker_count()`: CPU 自動スケーリング（70% 利用率、上限 8）
  - `ParallelBacktestRunner`: 並列バックテスト実行（ProcessPool/ThreadPool 自動選択）
  - `PerformanceTimer`: コンテキストマネージャで実行時間計測
  - `DataFrameCache`: 読み取り専用ビューベースのキャッシュ

### Workstream #4: 診断・監視強化

- **`common/alert_framework.py`** (500 行)
  - `AlertManager`: 条件登録・イベント管理
  - `AlertCondition`: プラガブル条件定義（6 operators: >, >=, <, <=, ==, !=）
  - `AlertEvent`: イベント記録（4 重大度レベル、6 アクション型）
  - `AlertSeverity`: INFO, WARNING, ERROR, CRITICAL
  - `AlertAction`: LOG, LOG_WARNING, LOG_ERROR, NOTIFY, STOP, CUSTOM
  - `make_freshness_alert()`, `make_latency_alert()`, `make_memory_alert()`: 事前定義プリセット

### Workstream #5: ドキュメント完全化

- **`docs/TECHNICAL_SPECS.md`** (450 行)
  - System1-7 詳細仕様（エントリー条件、ランキング、指標）
  - Option-B 3 段階アーキテクチャ（入力検証 → 閾値適用 → 診断確定）
  - 診断標準コントラクト
  - トラブルシューティングガイド

## 絶対ルール（破らない）

- キャッシュ I/O は必ず `common/cache_manager.py::CacheManager` 経由。`data_cache/` を直接読まない/書かない。
- `core/system7.py` は SPY 固定ヘッジ。対象やロジックの拡張は禁止。
- 設定は `config/settings.py::get_settings()`、環境は `config/environment.py::get_env_config()` 経由（`os.environ.get` 直読禁止）。
- テストで外部 API 呼び出し禁止。キャッシュデータ or モック/フィクスチャを使う。

## 開発でよく使うコマンド（PowerShell）

- UI 起動: `python -m streamlit run apps/app_integrated.py`
- 当日パイプライン: `python scripts/run_all_systems_today.py --parallel --save-csv`
- 高速スモーク: `python scripts/run_all_systems_today.py --test-mode mini --skip-external --benchmark`
- 制御テスト一式: `python scripts/run_controlled_tests.py`（または `pytest -q tests/test_systems_controlled_all.py`）
- Lint/Format: `ruff check --fix .` → `black .` → `isort .`（pre-commit 推奨）

## キャッシュ層と解決順序（重要）

- 層: `full_backup`(原本) → `base`(指標付与済・既定) → `rolling`(直近 N 日・当日用)。
- 解決: today は `rolling→base→full_backup`、backtest は `base→full_backup`。I/O は `CacheManager`/`common/cache_io.py` 経由。

## 診断・進捗・デバッグ

- Diagnostics 共通キー: `setup_predicate_count`, `ranked_top_n_count`, `ranking_source`（全 System1–7）。
- 進捗監視: `ENABLE_PROGRESS_EVENTS=1` で UI の進捗タブから追跡。配分詳細は `get_env_config().allocation_debug`。

## 最終配分 API（契約）

- `finalize_allocation(per_system, strategies=?, positions=?, ...) -> (df, summary)` を維持。
- `--test-mode mini` の出力は `per_system={"system1": df1, ...}` で渡す。

## よくある落とし穴（回避策）

- slot モードの実株数が必要なら `include_trade_management=True` を忘れない（未指定だと shares 未算出）。
- 資本配分で `strategies` 未提供だと計算が無効化されることがある（厳格運用は `ALLOCATION_REQUIRE_STRATEGIES=1`）。
- 現有ポジション集計は主系統 `symbol_system_map.primary` に依存。未提供時は `common.symbol_map::load_symbol_system_map()` にフォールバック。

## ファイルの目印（まずここを見る）

- `apps/app_integrated.py`（UI） / `scripts/run_all_systems_today.py`（当日フロー）
- `common/cache_manager.py`（キャッシュ I/O） / `common/indicator_access.py`（指標参照）
- `core/systemX.py`（ロジック）/ `strategies/systemX_strategy.py`（ラッパ）/ `core/final_allocation.py`（配分）

## 参考と運用

- VS Code タスクから “Run All Systems Today” / “Quick Test Run” / “Start Streamlit UI” をすぐ呼べます。
- PR には「影響する入力/キャッシュ」「検証手順（例: `--test-mode mini`）」「UI 変更ならスクショ（`results_images/`）」を必ず添付。
- 詳細は `docs/README.md` と `docs/internal/AGENTS.md` を参照。再現性が不安定なときは `common/testing.py::set_test_determinism()` を冒頭で呼びます。
