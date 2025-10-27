# Copilot instructions — quick reference (repository-specific)

目的: AI エージェントがこのリポジトリで安全かつ即戦力に編集・提案できるための最小ルール集。

必ず最初に読む: `docs/README.md`（設計・運用・コマンドの統合ハブ）。編集前はここで仕様・データフローと注意点を確認すること。

主要ポイント（要約）

- エントリ: `apps/app_integrated.py`, `apps/app_today_signals.py`（Streamlit UI）
- 日次パイプライン: `scripts/run_all_systems_today.py`（symbols → load → indicators → filters → setup → signals → allocation → save/notify）
- 戦略分離: 低レベルロジックは `core/systemX.py`、戦略は `strategies/systemX_strategy.py`

必須ルール（破らないでください）

- キャッシュ I/O は常に `common/cache_manager.py::CacheManager` を経由する（直接 `data_cache/` を参照・書き換えない）。
- `core/system7.py`（SPY 固定）を改変しない。System7 はヘッジ専用で特別扱いです。
- 設定は必ず `config/settings.py::get_settings()` 経由、環境変数は `config/environment.py::get_env_config()` を通す（直接 `os.environ.get()` 禁止）。

重要ファイル（参照例）

- 処理フロー: `scripts/run_all_systems_today.py`
- 最終配分: `core/final_allocation.py` (`finalize_allocation(...)` の呼び出し規約を尊重)
- キャッシュ: `common/cache_manager.py`, `common/cache_io.py`
- UI テスト/スクリーンショット: `tools/capture_ui_screenshot.py`, `tools/run_and_snapshot.ps1`, Playwright 設定 `playwright.config.ts`

よく使うコマンド（PowerShell 例）

- 依存インストール: `pip install -r requirements.txt`
- UI 起動: `& .\venv\Scripts\python.exe -m streamlit run apps/app_integrated.py`
- 当日一式（本番相当）: `python scripts/run_all_systems_today.py --parallel --save-csv`
- 速い検証: `python scripts/run_all_systems_today.py --test-mode mini --skip-external --benchmark`
- テスト: `python -m pytest -q`（個別: `pytest tests/test_system3.py::test_entry_rules`）

デバッグ・検証ループ（推奨ワークフロー）

- 変更後は `--test-mode mini` で素早く動作確認。UI 変更は Playwright ベースの `tools/run_and_snapshot.ps1` でスクリーンショットを取り、`snapshots/` と `imgdiff_report.html` を確認する。
- 再現性問題が疑われる場合、`common/testing.py::set_test_determinism()` を呼んで安定化を試みる。

開発スタイルと CI 前提

- Lint/format: `ruff`/`black`/`isort` を `pre-commit` で回す。CI は `ruff/black/mypy/pytest` を実行します。
- 外部 API 呼び出しはテストで禁止。キャッシュ済みデータ（`data_cache/`）を使うか、モック/フィクスチャを使うこと。

実装時の小さな契約（短く）

- 入出力: 公開関数に型ヒントをつける。戻り値はドキュメント化する。
- エラー: 入力データ不備は早期に ValueError を投げる。外部 I/O の失敗は例外をラップして上位に伝える。

変更を提案したら必ず添える情報（PR 必須項目）

1. どの入力ファイル / キャッシュに影響するか（例: `data_cache/rolling/system3.feather`）
2. 期待されるテストコマンド（`--test-mode mini` の例）
3. UI 変更があればスクリーンショット（`results_images/`）

その他: 不明点があれば、編集対象ファイルと実行したいコマンドを教えてください。こちらで patch を作り、`--test-mode mini` で確認して差分を示します。

```

```
