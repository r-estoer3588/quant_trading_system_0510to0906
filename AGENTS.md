# Repository Guidelines

## Project Structure & Module Organization
- エントリ: `app_integrated.py`（Streamlit UI、System1–7 のタブ統合）
- 戦略: `strategies/`（例: `system1_strategy.py`, `base_strategy.py`）
- 共通: `common/`（例: `backtest_utils.py`, `ui_components.py`）
- 設定: `config/settings.py`（`.env` を読む `get_settings()`）
- データ/出力: `data_cache/`, `results_csv/`, `logs/`（git 追跡外）
- テスト: `tests/`（`test_system1.py` … `test_system7.py`）
- ツール/スクリプト: `tools/`, `scripts/`（インポート解析、キャッシュ更新 等）

## Build, Test, and Development Commands
- 依存関係: `pip install -r requirements.txt`
- 開発ツール: `pip install -r requirements-dev.txt`
- UI 起動: `streamlit run app_integrated.py`
- データキャッシュ: `python scripts/cache_daily_data.py`（`EODHD_API_KEY` 必須）
- テスト: `pytest -q`（例: 集中実行 `pytest tests/test_system3.py::test_entry_rules`）
- 単体モジュール実行: `python -m strategies.system1_strategy`

## Coding Style & Naming Conventions
- インデント 4 スペース、PEP 8 準拠。公開 API は型ヒント推奨。
- 命名: ファイル/関数は `snake_case`、クラスは `PascalCase`。
- インポート順: 標準/サードパーティ/ローカル。循環依存は回避。
- ドキュメンテーション: 入出力と前提を簡潔に docstring で記述。

### Lint/Format（pre-commit）
- 導入: `pre-commit install`
- 使用: コミット時に `ruff`/`black`/`isort`/基本フックが自動実行。

## Testing Guidelines
- Codex タスク後は以下のコマンドでテストを実行し、異常がないか確認すること:
  - `pre-commit run --files tests/test_headless_app.py tests/test_utils.py tests/app_smoke.py`
  - `pytest tests/test_headless_app.py tests/test_utils.py -q`
- フレームワーク: `pytest`。決定性重視（乱数シード・日付固定）。
- ネットワーク呼び出しは避け、キャッシュ済みデータを使用。
- 実行: `pytest -q`。失敗時は対象を絞ってデバッグ。
 - 便利関数: `common/testing.py` の `set_test_determinism()` を活用。
 - フィクスチャ例: `freezegun` で日時固定、`monkeypatch` で外部I/O遮断。

## Commit & Pull Request Guidelines
- コミット: 命令形・現在形、件名 72 文字以内。
  - 例: `feat(strategies): add SMA/EMA crossover for System2`
  - 例: `fix(common): guard empty price series`
- PR: 目的/背景、関連 Issue、検証手順、（UI 変更時）スクリーンショット。
- チェック: テスト合格、新規警告なし、環境変数変更時は README/設定更新。

## Security & Configuration Tips
- 秘密情報は `.env` に保存しコミットしない（必須: `EODHD_API_KEY`）。
- パス作成は `get_settings(create_dirs=True)` を利用。
- I/O は `data_cache/`, `results_csv/`, `logs/` の配下に限定。

## Language Policy
- 原則: すべての回答は日本語で提供してください。
- 例外: 識別子や外部 API 名は英語準拠を維持して構いません。

## Architecture Overview（任意）
- Streamlit で System1–7 をタブ統合。戦略は `strategies/`、共通ロジックは `common/`。
- 計算結果・ログは git 追跡外ディレクトリに保存し、再現性と安全性を確保。

## Type Checking
- `mypy` を利用。基本は寛容設定（未型定義の関数は可）。段階的に厳格化を推奨。
- 実行: `mypy .`

## CI
- GitHub Actions (`.github/workflows/ci.yml`) で `ruff/black/isort/mypy/bandit/pip-audit/pytest` を実行。
- Python 3.10–3.12 のマトリクスで検証。

## Branch & Versioning
- ブランチ: `feat/*`, `fix/*`, `chore/*`。
- バージョニング: SemVer。`CHANGELOG.md` を更新。
