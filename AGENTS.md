# Repository Guidelines

## Project Structure & Module Organization
- エントリ: `app_integrated.py`（Streamlit UI。System1–7 タブ統合）
- 戦略: `strategies/`（例: `system1_strategy.py`, `base_strategy.py`）
- 共通: `common/`（`backtest_utils.py`, `ui_components.py` 等）
- 設定: `config/settings.py`（`.env` を読む `get_settings()`）
- データ/出力: `data_cache/`, `results_csv/`, `logs/`（git 追跡外）
- テスト: `tests/`（`test_system1.py` … `test_system7.py` ほか）
- ツール/スクリプト: `tools/`, `scripts/`

## Build, Test, and Development Commands
- 依存関係: `pip install -r requirements.txt`（開発は `requirements-dev.txt`）
- UI 起動: `streamlit run app_integrated.py`
- データキャッシュ: `python scripts/cache_daily_data.py`（`EODHD_API_KEY` 必須）
- 単体実行: `python -m strategies.system1_strategy`
- テスト: `pytest -q`（例: `pytest tests/test_system3.py::test_entry_rules`）
- 事前チェック: `pre-commit install` → `pre-commit run --files tests/test_headless_app.py tests/test_utils.py tests/app_smoke.py`

## Coding Style & Naming Conventions
- PEP 8、インデント 4 スペース。公開 API は型ヒント推奨。
- 命名: 関数/ファイルは `snake_case`、クラスは `PascalCase`。
- インポート順: 標準/サードパーティ/ローカル。循環依存を回避。
- Lint/Format: `ruff`/`black`/`isort` を `pre-commit` で実行。
- Docstring は入出力と前提を簡潔に記述。

## Testing Guidelines
- フレームワーク: `pytest`。決定性を重視（`common/testing.py` の `set_test_determinism()`）。
- ネットワーク呼び出し禁止。キャッシュ済みデータを使用。
- 実行例: `pytest tests/test_headless_app.py tests/test_utils.py -q`。
- フィクスチャ: `freezegun` で時刻固定、`monkeypatch` で外部 I/O を遮断。

## Commit & Pull Request Guidelines
- コミット: 命令形・現在形、72 文字以内。
  - 例: `feat(strategies): add SMA/EMA crossover for System2`
  - 例: `fix(common): guard empty price series`
- PR: 目的/背景、関連 Issue、検証手順、UI 変更時はスクリーンショット。テスト合格・警告ゼロ必須。

## Security & Configuration Tips
- 秘密情報は `.env` に保存（例: `EODHD_API_KEY`）。コミット禁止。
- パス作成は `get_settings(create_dirs=True)` を利用。I/O は `data_cache/`, `results_csv/`, `logs/` 配下に限定。
- 言語方針: 回答・コメントは日本語。日本語文字列は UTF‑8 のまま保持。

## Cache Policy / CI / Type Checking
- キャッシュ層: `base/`・`rolling/`・`full_backup/`。リゾルバ: backtest は `base→full_backup`、today は `rolling→base→full_backup`。`data_cache/`直下の CSV を直接参照しない（`CacheManager`/`load_base_cache()` を経由）。
- 復旧: `recover_spy_cache.py` は `data_cache/full_backup/` に保存。
- 型: `mypy .`（段階的に厳格化）。
- CI: GitHub Actions で `ruff/black/isort/mypy/bandit/pip-audit/pytest` を Py3.10–3.12 で実行。
- ブランチ/バージョン: `feat/*`, `fix/*`, `chore/*`。SemVer、`CHANGELOG.md` 更新。

