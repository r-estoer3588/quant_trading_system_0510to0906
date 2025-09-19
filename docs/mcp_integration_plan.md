# MCP サーバー連携計画

## 概要
- `tools/mcp_server/` に Model Context Protocol (MCP) 対応の Python サーバーを実装しました。
- 既存の `results_csv/`・`data_cache/`・`config/`・`scripts/` を直接操作する MCP ツールを提供し、VS Code からの対話オートメーションを実現します。
- バックテスト実行では `common.backtest_utils.simulate_trades_with_risk` を使用し、既存ロジックと整合した結果を返します。

## MCP サーバー構成
```
tools/mcp_server/
├── __init__.py
├── __main__.py          # python -m tools.mcp_server エントリポイント
├── operations.py        # シグナル取得・バックテスト・ファイル操作などの本体実装
└── server.py            # MCP Server 定義（ツール一覧と call_tool ディスパッチ）
```
- `operations.py` では pandas/YAML/AST/サブプロセス等を用いて既存基盤と連携。
- `server.py` は MCP の `list_tools` / `call_tool` ハンドラを定義し、構造化レスポンスを返却。
- `tools/mcp_server/__main__.py` で CLI 実行 (`python -m tools.mcp_server`) をサポート。

## 提供ツール一覧
| ツール名 | 主な役割 |
| --- | --- |
| `hello` | 挨拶レスポンス（接続テスト）。 |
| `fetch_signals` | `results_csv`・`data_cache/signals` から最新シグナルを取得。 |
| `run_backtest` | 指定システムでデータを読み込み、`simulate_trades_with_risk` によるバックテスト実行。 |
| `summarize_performance` | 任意のトレード CSV を読み込み勝率・シャープレシオ等を集計。 |
| `read_config` / `write_config` | `config/*.yaml` を安全に読み書き（マージ or 上書き）。 |
| `execute_daily_run` | 新規追加の `scripts/daily_run.py` を起動し、日次バッチを実行。 |
| `search_project` | プロジェクト横断のファイル検索。 |
| `find_symbol_references` | 関数・クラス定義/参照の AST 横断探索。 |
| `run_pytest` | `pytest` のサブプロセス実行。 |
| `analyze_imports` | Python ファイルの import 依存関係解析。 |
| `run_python_file` | 任意の Python ファイルを実行し標準出力を取得。 |
| `write_text_file` | テキストファイルの生成・上書き・追記。 |
| `analyze_error_log` | ログファイル末尾からエラーサマリを抽出。 |

## トレードシステム連携
- `collect_signals()` で `results_csv/*.csv` と `signals_dir` を走査し、システム別フィルタや欠損ソース情報を返します。
- `run_backtest()` は `config/settings.py` からシンボルユニバース・初期資金を取得し、必要に応じて `load_base_cache()` でデータをロード。`Strategy` クラスを通じて候補生成後、`simulate_trades_with_risk` を呼び出します。
- 生成したトレード結果は `common.performance_summary.summarize()` で評価し、MCP から構造化レスポンスを取得できます。
- `scripts/daily_run.py` では `cache_daily_data → compute_today_signals → build_metrics_report → notify_metrics` を連続実行し、`schedulers/runner.py` から `daily_run` タスクとして呼び出せます。

## VS Code 連携
- `.vscode/settings.json` に以下のエントリを追加済みです。
- `.vscode/settings.json` では、以下のように自作サーバーに加えてブラウザ・ファイル操作系の MCP サーバーも登録しています。
  ```json
  "mcp.servers": {
    "quant-trading-mcp": {
      "command": "python",
      "args": ["-m", "tools.mcp_server"],
      "cwd": "${workspaceFolder}"
    },
    "playwright": {
      "command": "npx",
      "args": ["@playwright/mcp@latest"],
      "cwd": "${workspaceFolder}"
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "${workspaceFolder}"],
      "cwd": "${workspaceFolder}"
    }
  }
  ```
- VS Code の **MCP: Use Tool** から `hello` や `run_backtest` を呼び出してレスポンスを確認できます。`playwright` サーバーを使うとブラウザ自動化、`filesystem` サーバーで安全なファイル操作が可能です。

## セットアップ手順とコマンド例
```bash
# 依存関係 (開発ツール含む)
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install mcp  # requirements-dev に追加済み

# MCP サーバー単体起動
env PYTHONPATH=. python -m tools.mcp_server

# 日次バッチ手動実行
python -m scripts.daily_run

# スケジューラ実行 (config の cron に従う)
python -m schedulers.runner
```

## 拡張設計指針
1. **ディレクトリ構造**
   - MCP 関連コードは `tools/mcp_server/` に集約。追加ツールは `operations.py` に小さな関数として実装し、複雑化する場合はサブモジュールへ分割する。
2. **開発/本番切り替え**
   - `config/config.yaml` の `scheduler.jobs` で `daily_run` を平日朝に実行。開発環境では `.env` や `RESULTS_DIR` で出力パスを切り替え、`get_settings(create_dirs=True)` を通じて安全にディレクトリ作成。
3. **API キーとセキュリティ**
   - `EODHD_API_KEY` 等の機密情報は `.env` 管理。MCP ツールはリポジトリ外のパスを弾く (`_as_repo_path`) ことで書き込み先を限定。
4. **ロギングと監視**
   - `scripts/daily_run.py` は `common.logging_utils.setup_logging` を使用し、`logs/` へ日次ログを出力。`analyze_error_log` ツールでログチェックを自動化。
5. **将来拡張案**
   - プロンプトテンプレートを追加し、UI タスク (`streamlit`) の自動テスト実行を MCP 経由でトリガー。
   - `FastMCP` (HTTP/SSE) への置き換えにより、クラウド常駐型の MCP サーバーを構築。
   - Alpaca/Slack 等の外部 API 呼び出しを専用ツールとして追加し、`config/settings.py` の `strategies` パラメータを MCP 経由で更新。

## 参考
- 追加スケジューラタスク: `config/config.yaml` に `daily_run` を登録済み。`schedulers/runner.py` の `TASKS` から `scripts/daily_run.py` を呼び出します。
- 日次実行結果は `logs/` および `results_csv/` に出力され、MCP ツール経由で即時確認できます。
