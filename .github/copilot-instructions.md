# Copilot Instructions (Condensed)

目的: このリポジトリで AI エージェントが安全かつ即戦力で編集・追加を行うための最小必須知識。過度な抽象説明より「何を / どこで / どう守るか」。

## 0. 必須参照ドキュメント (Always Reference)

**最重要**: 質問・回答・編集前に必ず `docs/README.md` を参照すること。プロジェクト全体の構造・システム仕様・技術詳細・運用ガイドが統合された包括的なナビゲーションハブ。相互参照リンクで関連文書への効率的アクセスが可能。

- **統合目次**: [docs/README.md](docs/README.md) - 4 分野（クイックスタート・システム概要・技術文書・運用ガイド）
- **システム仕様**: [docs/systems/](docs/systems/) - System1-7 詳細仕様
- **技術詳細**: [docs/technical/](docs/technical/) - アーキテクチャ・指標・MCP 統合
- **処理フロー**: [docs/today_signal_scan/](docs/today_signal_scan/) - 8 フェーズ詳細
- **運用手順**: [docs/operations/](docs/operations/) - 自動実行・通知・監視

## 1. コア構造 / Entry Points

- UI: `apps/app_integrated.py`（統合タブ） / 当日シグナル UI 補助: `apps/app_today_signals.py`。
- 日次パイプライン: `scripts/run_all_systems_today.py`（8 フェーズ: symbols → load → shared indicators → filters(2-phase) → setup → signals → allocation → save/notify）。
- 戦略分離: ロジック `core/system{1..7}.py` / ラッパ `strategies/system*_strategy.py` / 統合 BT `common/integrated_backtest.py`。

## 2. データキャッシュ階層 (絶対ルール)

- 階層: `rolling`(直近 300 日, 今日用) → `base`(指標付与長期) → `full_backup`(原本)。
- 取得順: today = rolling→base→full_backup / backtest = base→full_backup。
- 直接 CSV 読み禁止: すべて `common/cache_manager.py::CacheManager` 経由 (Feather 優先, CSV フォールバック)。
- 指標キャッシュ: `data_cache/indicators_systemX_cache/`。

**重要な実装詳細**:

- `CacheManager._read_base_and_tail()`: base→full_backup のフォールバック + tail(330 行) で rolling 相当を生成。
- デュアルフォーマット: Feather 優先（74% サイズ削減）、CSV 自動フォールバック (`CacheFileManager`)。
- 重複列除去済み: `open/Open/OPEN` などの冗長列は PascalCase に統一（58→35 列）。

## 3. システム特性 / 不変条件

- ロング: 1,3,4,5 / ショート: 2,6,7。System7 = SPY 固定 (変更禁止)。
- Two-Phase: Filter 列判定 → Setup 列判定 → ランキング → 配分。
- 主なランキングキー例: S1=ROC200, S2=ADX7, S3=3 日下落, S4=RSI4 低, S5=ADX7, S6=6 日上昇。
- 配分: スロット/金額制 + `data/symbol_system_map.json`。`DEFAULT_ALLOCATIONS` を壊さない。

**Two-Phase 処理の実装**:

1. `common/today_filters.py`: `filter_systemX()` が Filter 列を生成・保存。
2. `common/system_setup_predicates.py`: `systemX_setup_predicate()` が Setup 条件を関数化。
3. `core/systemX.py`: `generate_systemX_candidates()` が Setup predicate でランキング対象を抽出。
4. `latest_only=True`: 当日シグナルは最終行のみ判定（O(銘柄数)）、backtest は全履歴走査。

**Diagnostics API (Phase0-7 導入)**:

- 全システム共通キー: `ranking_source`, `setup_predicate_count`, `final_top_n_count`。
- 候補生成関数は `(candidates, diagnostics)` タプルを返す。
- Snapshot export: `--test-mode mini` で `results_csv_test/diagnostics_snapshot_*.json` に出力。
- アクセス: `common/system_diagnostics.py::get_diagnostics_with_fallback()` で安全取得。

## 4. 設定 & 環境

- 優先順位: JSON > YAML > .env (`config/settings.py::get_settings`)。新規出力は `get_settings(create_dirs=True)` が返すパス配下のみ。
- 主要環境例: `COMPACT_TODAY_LOGS`, `ENABLE_PROGRESS_EVENTS`, `ROLLING_ISSUES_VERBOSE_HEAD`。

**設定読み込みの仕組み**:

- `get_settings()`: `@lru_cache` でシングルトン化、YAML/JSON/.env を統合。
- 検証: `config/schemas.py::validate_config_dict()` で JSON Schema バリデーション（オプション）。
- カスタマイズ: `config.yaml` をベース、秘匿値は `.env`、CI/テスト用は JSON 上書き。

**環境変数の統一管理** (2025 年 10 月 10 日導入):

- **完全リスト**: [docs/technical/environment_variables.md](docs/technical/environment_variables.md) - 全 40+環境変数の詳細（デフォルト値・使用例・警告）
- **型安全アクセス**: `config/environment.py::EnvironmentConfig` - dataclass で型安全な環境変数管理
- **使用パターン**:

  ```python
  from config.environment import get_env_config

  env = get_env_config()  # シングルトン
  if env.compact_logs:
      logger.setLevel(logging.WARNING)

  # バリデーション（危険な設定を検出）
  errors = env.validate()
  if errors:
      for err in errors:
          logger.warning(err)
  ```

- **禁止**: `os.environ.get()` の直接使用 → 必ず `get_env_config()` 経由でアクセス
- **機密情報**: `APCA_API_KEY_ID`, `SLACK_BOT_TOKEN` 等は `.env` で管理、リポジトリにコミット禁止

## 5. 開発ワークフロー (必須コマンド)

```powershell
pip install -r requirements.txt            # 初回
streamlit run apps/app_integrated.py       # UI
python scripts/run_all_systems_today.py --parallel --save-csv
python scripts/run_all_systems_today.py --test-mode mini --skip-external --benchmark  # 2秒高速検証
pytest -q                                   # 決定性テスト
pre-commit run --files <changed_files>
```

**テスト戦略**:

- `--test-mode mini`: 10 銘柄（超高速、2 秒）/ `quick`: 50 銘柄 / `sample`: 100 銘柄。
- `--skip-external`: NASDAQ Trader / pandas_market_calendars API をスキップ。
- `--benchmark`: パフォーマンス計測 JSON を `results_csv_test/` に出力。
- 決定性: `common/testing.py::set_test_determinism()` でシード固定 + freezegun 日時固定。

## 6. 守るべき禁止事項 / ガードレール

1. Public API / 既存 CLI フラグ / System7 SPY / DEFAULT_ALLOCATIONS を破壊変更しない。
2. 外部ネットワーク呼び出しをテスト経路に追加しない（`--test-mode` + `--skip-external` 互換保持）。
3. キャッシュ直接 I/O 禁止（必ず CacheManager）。
4. 新規巨大依存追加は避け、パフォーマンス影響はベンチマーク (`--benchmark`) で確認。

**具体例**:

- ❌ `pd.read_csv("data_cache/rolling/AAPL.csv")` → ⭕ `cache_manager.load_rolling("AAPL")`
- ❌ `core/system7.py` で SPY 以外の銘柄を許可 → ⭕ SPY 固定を維持
- ❌ テストに `requests.get()` 追加 → ⭕ キャッシュ済みデータで再現

## 7. 実装パターン

- Two-Phase: `today_filters.py` → Setup ラベル生成 → `today_signals.py` が抽出。
- ログ最適化: `COMPACT_TODAY_LOGS=1` で詳細を DEBUG へ。進捗は `ENABLE_PROGRESS_EVENTS=1` + `logs/progress_today.jsonl`。
- DataFrame 操作は重複列を増やさない (冗長列除去済み方針)。

**統一ログ出力パターン (2025 年 10 月 10 日導入)**:

- **推奨**: `common/logging_utils.py::SystemLogger` を使用

  ```python
  from common.logging_utils import SystemLogger

  sys_logger = SystemLogger.create("System3", logger=logger, log_callback=log_callback)
  sys_logger.info("処理開始", symbol_count=100)
  sys_logger.error("エラー発生", symbol="AAPL")
  ```

- **代替**: `common/error_handling.py::SystemErrorHandler` も同等の機能を提供
- **非推奨**: `logger.error()` のみ、`log_callback()` のみ、`print()` の直接使用

**戦略パターン (Strategy/Core 分離)**:

- `core/systemX.py`: 純粋関数ロジック（候補生成・指標計算なし）。
- `strategies/systemX_strategy.py`: `StrategyBase` 継承、`generate_candidates()` で core を呼び出し。
- `common/integrated_backtest.py`: 全システム統合 BT、`DEFAULT_ALLOCATIONS` で資金配分。

**配分ロジック (`core/final_allocation.py`)**:

- `finalize_allocation()`: スロット/金額制を統合、`data/symbol_system_map.json` でシステム固定銘柄を管理。
- Long/Short バケツ別配分: `DEFAULT_LONG_ALLOCATIONS` (S1/3/4/5) + `DEFAULT_SHORT_ALLOCATIONS` (S2/6/7)。
- ポジションサイズ: ATR ベース + `risk_pct=0.02` / `max_pct=0.10` 上限。

## 8. コードスタイル / 品質

- snake_case / PascalCase / 型ヒント推奨。`ruff` + `black`。決定性: `common/testing.py` 利用。
- 変更後: fast pipeline mini モード + `pytest -q` を想定。

**pre-commit フック**:

- pre-commit: `ruff` / `black` / `isort` / 基本チェック (trailing whitespace)。
- pre-push: mini パイプライン (core/common 変更時) + 品質集計 + **black 厳格チェック**。
- バイパス禁止: `--no-verify` は使わない（CI で失敗する）。

**型チェック (Windows UTF-8 対応)**:

- `tools/mypy_utf8_runner.py`: Windows cp932 エンコード例外を回避（UTF-8 強制）。
- 実行例: `python tools/mypy_utf8_runner.py core/system1.py --no-incremental`。

## 9. 追加時のチェックリスト (PR 前)

- [ ] Cache 経由のみか
- [ ] System7/SPY/CLI フラグへ影響なし
- [ ] mini テスト 2 秒パス / pytest パス
- [ ] 新規出力パスは settings 管理下
- [ ] ログ量増加なし / 必要なら COMPACT 対応コメント
- [ ] Diagnostics API: 候補生成関数が統一キー (`ranking_source`, `setup_predicate_count`, `final_top_n_count`) を返すか
- [ ] 重複列を増やしていないか（OHLCV は PascalCase 統一）
- [ ] 環境変数を追加した場合: `config/environment.py::EnvironmentConfig` と `docs/technical/environment_variables.md` の両方に追記
- [ ] `os.environ.get()` の直接使用を避け、`get_env_config()` 経由でアクセスしているか

**デバッグヒント**:

- 候補数不一致: `VALIDATE_SETUP_PREDICATE=1` で predicate vs Setup 列の差分ログを確認。
- キャッシュ問題: `ROLLING_ISSUES_VERBOSE_HEAD=5` で rolling キャッシュ問題の詳細を表示。
- 進捗監視: `ENABLE_PROGRESS_EVENTS=1` + Streamlit UI「当日シグナル」タブでリアルタイム追跡。
- 環境変数確認: `python -c "from config.environment import get_env_config; env = get_env_config(); env.print_env_summary()"` で現在の設定を表示。
- 本番環境警告: `env.validate()` が返すエラーリストで、テスト用設定（`MIN_DROP3D_FOR_TEST` 等）が本番で有効になっていないか確認。

不明点や曖昧な規約は PR 説明に背景を記述し合意形成してください。
