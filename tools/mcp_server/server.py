from __future__ import annotations

import json
from typing import Any
import asyncio

from typing import TYPE_CHECKING as _TYPE_CHECKING

try:
    import anyio  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    anyio = None  # type: ignore

try:
    from mcp import types  # type: ignore
    from mcp.server import NotificationOptions, Server, stdio  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    if _TYPE_CHECKING:  # pragma: no cover - typing only
        from mcp import types  # type: ignore
        from mcp.server import NotificationOptions, Server, stdio  # type: ignore
    else:
        from types import SimpleNamespace as _SimpleNamespace

        class _Tool:
            def __init__(self, **kwargs: Any) -> None:  # pragma: no cover - stub
                for k, v in kwargs.items():
                    setattr(self, k, v)

        class _TextContent:
            def __init__(self, **kwargs: Any) -> None:  # pragma: no cover - stub
                for k, v in kwargs.items():
                    setattr(self, k, v)

        types = _SimpleNamespace(Tool=_Tool, TextContent=_TextContent)  # type: ignore

        class NotificationOptions:  # pragma: no cover - stub
            def __init__(self, *_, **__):
                pass

        class _StdIo:
            class _Ctx:
                async def __aenter__(self):
                    return (None, None)

                async def __aexit__(self, exc_type, exc, tb):
                    return False

            def stdio_server(self):
                return self._Ctx()

        class Server:  # pragma: no cover - stub
            def __init__(self, *_, **__):
                pass

            def list_tools(self):
                def _decorator(f):
                    return f

                return _decorator

            def call_tool(self):
                def _decorator(f):
                    return f

                return _decorator

            def create_initialization_options(self, *_, **__):
                return {}

            async def run(self, *_, **__):
                return None

        stdio = _StdIo()

from . import operations as ops
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .operations import BacktestResult  # type: ignore


def _text(message: str) -> list[Any]:
    if not hasattr(types, "TextContent"):
        return [{"type": "text", "text": message}]
    return [types.TextContent(type="text", text=message)]


async def _run_sync(func, *args, **kwargs):
    if anyio is not None:
        return await anyio.to_thread.run_sync(lambda: func(*args, **kwargs))
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


def _format_summary(summary: dict[str, Any]) -> str:
    lines = ["バックテスト結果"]
    for key in (
        "trades",
        "total_return",
        "win_rate",
        "max_drawdown",
        "sharpe",
        "sortino",
        "profit_factor",
        "cagr",
    ):
        value = summary.get(key)
        if value is None:
            continue
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.4f}")
        else:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def create_server() -> Server:
    server = Server(
        "quant-trading-mcp",
        version="0.1.0",
        instructions=(
            "米国株自動売買システムとバックテスト基盤を操作する開発支援MCPサーバーです。"
            "results_csv や data_cache からのデータ取得、シミュレーション、"
            "pytest 実行などを提供します。"
        ),
    )

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:  # noqa: WPS430 - MCP API
        return [
            types.Tool(
                name="hello",
                description="指定した名前へ挨拶を返します。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "宛先となる名前"},
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
            ),
            types.Tool(
                name="fetch_signals",
                description="results_csv や data_cache のシグナルを取得します。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "system": {
                            "type": "string",
                            "description": "system1 などのシステム名 (任意)",
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 500,
                            "default": 20,
                            "description": "取得するシグナル件数",
                        },
                    },
                    "additionalProperties": False,
                },
            ),
            types.Tool(
                name="run_backtest",
                description=(
                    "指定システムで simulate_trades_with_risk を用いた バックテストを実行します。"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "system": {"type": "string", "description": "system1〜system7"},
                        "capital": {"type": "number", "description": "初期資金 (任意)"},
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "対象銘柄リスト (任意)",
                        },
                        "start_date": {
                            "type": "string",
                            "description": "YYYY-MM-DD 形式の開始日 (任意)",
                        },
                        "end_date": {
                            "type": "string",
                            "description": "YYYY-MM-DD 形式の終了日 (任意)",
                        },
                        "max_symbols": {
                            "type": "integer",
                            "description": "シンボル最大数 (任意)",
                        },
                        "rebuild_cache": {
                            "type": "boolean",
                            "description": "キャッシュ再構築を許可するか",
                            "default": False,
                        },
                    },
                    "required": ["system"],
                    "additionalProperties": False,
                },
            ),
            types.Tool(
                name="summarize_performance",
                description="既存のトレードCSVからパフォーマンス指標を集計します。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "trades_path": {
                            "type": "string",
                            "description": "トレード履歴CSVへのパス",
                        },
                        "capital": {
                            "type": "number",
                            "description": "初期資金 (任意)",
                        },
                    },
                    "required": ["trades_path"],
                    "additionalProperties": False,
                },
            ),
            types.Tool(
                name="read_config",
                description="config ディレクトリ内の YAML を読み込みます。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "例: config.yaml",
                        }
                    },
                    "required": ["filename"],
                    "additionalProperties": False,
                },
            ),
            types.Tool(
                name="write_config",
                description="config YAML の上書きまたはマージ更新を行います。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "content": {"type": "string", "description": "YAML文字列"},
                        "updates": {"type": "object", "description": "辞書形式の更新内容"},
                        "mode": {
                            "type": "string",
                            "enum": ["merge", "overwrite"],
                            "default": "merge",
                        },
                    },
                    "required": ["filename"],
                    "additionalProperties": False,
                },
            ),
            types.Tool(
                name="execute_daily_run",
                description="scripts.daily_run を起動し日次バッチを実行します。",
                inputSchema={"type": "object", "additionalProperties": False},
            ),
            types.Tool(
                name="search_project",
                description="プロジェクト内でテキスト検索を実行します。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "file_globs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "glob パターンのリスト",
                        },
                        "max_results": {
                            "type": "integer",
                            "default": 100,
                            "minimum": 1,
                        },
                        "case_sensitive": {"type": "boolean", "default": False},
                    },
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
            ),
            types.Tool(
                name="find_symbol_references",
                description="関数・クラス定義および参照箇所を横断検索します。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                    },
                    "required": ["symbol"],
                    "additionalProperties": False,
                },
            ),
            types.Tool(
                name="run_pytest",
                description="pytest を実行し結果を取得します。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "args": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "pytest へ渡す追加引数",
                        }
                    },
                    "additionalProperties": False,
                },
            ),
            types.Tool(
                name="analyze_imports",
                description="Python ファイルの import 依存関係を解析します。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "ファイルパス"},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            ),
            types.Tool(
                name="run_python_file",
                description="任意の Python ファイルを実行し標準出力を取得します。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "args": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "コマンドライン引数",
                        },
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            ),
            types.Tool(
                name="write_text_file",
                description="テキストファイルの生成・追記・上書きを行います。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "mode": {
                            "type": "string",
                            "enum": ["overwrite", "append"],
                            "default": "overwrite",
                        },
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            ),
            types.Tool(
                name="analyze_error_log",
                description="ログファイルのエラーを解析し概要を要約します。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "tail": {
                            "type": "integer",
                            "minimum": 10,
                            "default": 200,
                            "description": "解析対象とする末尾行数",
                        },
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> Any:  # noqa: WPS430
        if name == "hello":
            target = arguments.get("name", "world")
            return _text(f"Hello, {target}")

        if name == "fetch_signals":
            result = await _run_sync(
                ops.collect_signals,
                system=arguments.get("system"),
                limit=int(arguments.get("limit", 20)),
            )
            text = json.dumps(result, ensure_ascii=False, indent=2)
            return (_text(text), result)

        if name == "run_backtest":
            result: BacktestResult = await _run_sync(
                ops.run_backtest,
                arguments["system"],
                capital=arguments.get("capital"),
                symbols=arguments.get("symbols"),
                start_date=arguments.get("start_date"),
                end_date=arguments.get("end_date"),
                max_symbols=arguments.get("max_symbols"),
                rebuild_cache=bool(arguments.get("rebuild_cache", False)),
            )
            structured = {
                "summary": result.summary,
                "trades": result.trades,
                "logs": result.logs,
                "missingSymbols": result.missing_symbols,
            }
            summary_text = _format_summary(result.summary)
            return (_text(summary_text), structured)

        if name == "summarize_performance":
            structured = await _run_sync(
                ops.summarize_performance,
                arguments["trades_path"],
                capital=arguments.get("capital"),
            )
            summary_text = _format_summary(structured.get("summary", {}))
            return (_text(summary_text), structured)

        if name == "read_config":
            data = await _run_sync(ops.read_config_yaml, arguments["filename"])
            return (_text(json.dumps(data, ensure_ascii=False, indent=2)), data)

        if name == "write_config":
            data = await _run_sync(
                ops.write_config_yaml,
                arguments["filename"],
                content=arguments.get("content"),
                updates=arguments.get("updates"),
                mode=arguments.get("mode", "merge"),
            )
            return (_text(json.dumps(data, ensure_ascii=False, indent=2)), data)

        if name == "execute_daily_run":
            result = await _run_sync(ops.execute_daily_run)
            text = "\n".join(
                [
                    f"command: {result['command']}",
                    f"returncode: {result['returncode']}",
                    "--- stdout ---",
                    result.get("stdout", "").strip(),
                    "--- stderr ---",
                    result.get("stderr", "").strip(),
                ]
            )
            return (_text(text), result)

        if name == "search_project":
            matches = await _run_sync(
                ops.search_project_files,
                arguments["pattern"],
                file_globs=arguments.get("file_globs"),
                max_results=int(arguments.get("max_results", 100)),
                case_sensitive=bool(arguments.get("case_sensitive", False)),
            )
            text = json.dumps(matches, ensure_ascii=False, indent=2)
            return (_text(text), {"matches": matches})

        if name == "find_symbol_references":
            result = await _run_sync(ops.find_symbol_references, arguments["symbol"])
            return (_text(json.dumps(result, ensure_ascii=False, indent=2)), result)

        if name == "run_pytest":
            result = await _run_sync(ops.run_pytest, arguments.get("args"))
            text = "\n".join(
                [
                    f"command: {result['command']}",
                    f"returncode: {result['returncode']}",
                    "--- stdout ---",
                    result.get("stdout", "").strip(),
                    "--- stderr ---",
                    result.get("stderr", "").strip(),
                ]
            )
            return (_text(text), result)

        if name == "analyze_imports":
            result = await _run_sync(ops.analyze_imports, arguments["path"])
            text_payload = json.dumps(result, ensure_ascii=False, indent=2)
            return (_text(text_payload), {"imports": result})

        if name == "run_python_file":
            result = await _run_sync(
                ops.run_python_file,
                arguments["path"],
                arguments.get("args"),
            )
            text = "\n".join(
                [
                    f"command: {result['command']}",
                    f"returncode: {result['returncode']}",
                    "--- stdout ---",
                    result.get("stdout", "").strip(),
                    "--- stderr ---",
                    result.get("stderr", "").strip(),
                ]
            )
            return (_text(text), result)

        if name == "write_text_file":
            result = await _run_sync(
                ops.write_text_file,
                arguments["path"],
                arguments["content"],
                mode=arguments.get("mode", "overwrite"),
            )
            return (_text(json.dumps(result, ensure_ascii=False, indent=2)), result)

        if name == "analyze_error_log":
            result = await _run_sync(
                ops.analyze_error_log,
                arguments["path"],
                tail=int(arguments.get("tail", 200)),
            )
            return (_text(json.dumps(result, ensure_ascii=False, indent=2)), result)

        raise ValueError(f"未知のツールです: {name}")

    return server


async def main() -> None:
    server = create_server()
    init_opts = server.create_initialization_options(
        notification_options=NotificationOptions(),
    )
    async with stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init_opts)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    anyio.run(main)
