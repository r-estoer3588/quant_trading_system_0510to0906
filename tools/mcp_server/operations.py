from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import yaml

from common.backtest_utils import simulate_trades_with_risk
from common.cache_manager import load_base_cache
from common.performance_summary import summarize
from config.settings import Settings, get_settings
from strategies.system1_strategy import System1Strategy
from strategies.system2_strategy import System2Strategy
from strategies.system3_strategy import System3Strategy
from strategies.system4_strategy import System4Strategy
from strategies.system5_strategy import System5Strategy
from strategies.system6_strategy import System6Strategy
from strategies.system7_strategy import System7Strategy


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class BacktestResult:
    summary: dict[str, Any]
    trades: list[dict[str, Any]]
    logs: list[str]
    missing_symbols: list[str]


def _load_settings() -> Settings:
    return get_settings(create_dirs=True)


def _as_repo_path(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    if not p.is_absolute():
        candidate = (REPO_ROOT / p).resolve()
    else:
        candidate = p.resolve()
    try:
        candidate.relative_to(REPO_ROOT)
    except ValueError as exc:  # pragma: no cover - safety guard
        raise ValueError(f"Path '{candidate}' はリポジトリ外を指しています") from exc
    return candidate


def _default_results_dir() -> Path:
    settings = _load_settings()
    configured = getattr(settings.outputs, "results_csv_dir", Path("results_csv"))
    path = Path(configured)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _default_cache_signals_dir() -> Path:
    settings = _load_settings()
    configured = getattr(
        settings.outputs,
        "signals_dir",
        Path("data_cache") / "signals",
    )
    path = Path(configured)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_symbol_universe(limit: int | None = None) -> list[str]:
    universe_path = REPO_ROOT / "data" / "universe_auto.txt"
    symbols: list[str] = []
    if universe_path.exists():
        try:
            with universe_path.open("r", encoding="utf-8") as fh:
                symbols = [line.strip() for line in fh if line.strip()]
        except Exception:
            symbols = []
    if not symbols:
        symbols = ["SPY"]
    if limit is not None:
        symbols = symbols[:limit]
    return symbols


def _ensure_spy(symbols: Iterable[str]) -> list[str]:
    out = list(dict.fromkeys(sym.upper() for sym in symbols))
    if "SPY" not in out:
        out.insert(0, "SPY")
    return out


def _normalize_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.to_pydatetime().isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if pd.isna(value):
        return None
    return value


def _records_from_df(
    df: pd.DataFrame, limit: int | None = None
) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    if limit is not None:
        df = df.head(limit)
    records = []
    for rec in df.to_dict(orient="records"):
        normalized = {k: _normalize_value(v) for k, v in rec.items()}
        records.append(normalized)
    return records


def collect_signals(system: str | None = None, limit: int = 20) -> dict[str, Any]:
    system_key = system.lower() if system else None
    records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    missing_sources: list[str] = []

    results_dir = _default_results_dir()
    cache_dir = _default_cache_signals_dir()

    def _matches_system(path: Path) -> bool:
        if system_key is None:
            return True
        if system_key in path.stem.lower():
            return True
        return False

    csv_files = sorted(
        results_dir.glob("*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for csv_path in csv_files:
        if not _matches_system(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            missing_sources.append(str(csv_path.relative_to(REPO_ROOT)))
            continue
        if system_key and "system" in df.columns:
            df = df[df["system"].astype(str).str.lower() == system_key]
        if df.empty:
            continue
        subset = df.head(max(limit - len(records), 0)) if limit else df
        if subset.empty:
            continue
        records.extend(_records_from_df(subset))
        sources.append(
            {
                "path": str(csv_path.relative_to(REPO_ROOT)),
                "rows": int(len(subset)),
            }
        )
        if limit and len(records) >= limit:
            records = records[:limit]
            break

    if limit is None or len(records) < limit:
        if cache_dir.exists():
            for path in sorted(cache_dir.glob("**/*")):
                if path.is_dir():
                    continue
                if path.suffix.lower() not in {".csv", ".json"}:
                    continue
                if not _matches_system(path):
                    continue
                try:
                    if path.suffix.lower() == ".json":
                        with path.open("r", encoding="utf-8") as fh:
                            payload = json.load(fh)
                        df = pd.DataFrame(payload)
                    else:
                        df = pd.read_csv(path)
                except Exception:
                    missing_sources.append(str(path.relative_to(REPO_ROOT)))
                    continue
                if system_key and "system" in df.columns:
                    df = df[df["system"].astype(str).str.lower() == system_key]
                if df.empty:
                    continue
                subset = df.head(max(limit - len(records), 0)) if limit else df
                if subset.empty:
                    continue
                records.extend(_records_from_df(subset))
                sources.append(
                    {
                        "path": str(path.relative_to(REPO_ROOT)),
                        "rows": int(len(subset)),
                    }
                )
                if limit and len(records) >= limit:
                    records = records[:limit]
                    break

    return {
        "signals": records,
        "sources": sources,
        "missing_sources": missing_sources,
    }


STRATEGY_REGISTRY = {
    "system1": System1Strategy,
    "system2": System2Strategy,
    "system3": System3Strategy,
    "system4": System4Strategy,
    "system5": System5Strategy,
    "system6": System6Strategy,
    "system7": System7Strategy,
}


def _filter_by_date(
    df: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df
    if start is not None:
        out = out[out.index >= start]
    if end is not None:
        out = out[out.index <= end]
    return out


def run_backtest(
    system: str,
    *,
    capital: float | None = None,
    symbols: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_symbols: int | None = None,
    rebuild_cache: bool = False,
) -> BacktestResult:
    settings = _load_settings()
    capital_val = (
        capital if capital is not None else float(settings.backtest.initial_capital)
    )
    system_key = system.lower()
    if system_key not in STRATEGY_REGISTRY:
        raise ValueError(f"未知のシステム '{system}' です")
    strategy_cls = STRATEGY_REGISTRY[system_key]
    strategy = strategy_cls()

    start_ts = pd.to_datetime(start_date) if start_date else None
    end_ts = pd.to_datetime(end_date) if end_date else None

    if symbols:
        selected = [sym.upper() for sym in symbols]
    else:
        limit = max_symbols or settings.backtest.max_symbols
        selected = _load_symbol_universe(limit)
    selected = _ensure_spy(selected)

    data_dict: dict[str, pd.DataFrame] = {}
    missing_symbols: list[str] = []
    for sym in selected:
        try:
            df = load_base_cache(sym, rebuild_if_missing=rebuild_cache)
        except Exception:
            df = None
        if df is None or df.empty:
            missing_symbols.append(sym)
            continue
        df = df.sort_index()
        df = _filter_by_date(df, start_ts, end_ts)
        if df.empty:
            missing_symbols.append(sym)
            continue
        data_dict[sym] = df

    if not data_dict:
        raise RuntimeError("有効な価格データが取得できませんでした")

    log_messages: list[str] = []

    def _log(*msgs: Any) -> None:
        if not msgs:
            return
        message = " ".join(str(m) for m in msgs)
        log_messages.append(message)

    def _progress(*args: Any) -> None:
        if not args:
            return
        try:
            done = int(args[0]) if len(args) >= 1 else 0
        except Exception:
            done = 0
        try:
            total = int(args[1]) if len(args) >= 2 else 0
        except Exception:
            total = 0
        log_messages.append(f"progress {done}/{total}")

    def _skip(*msgs: Any) -> None:
        if not msgs:
            return
        _log(*msgs)

    try:
        prepared = strategy.prepare_data(
            data_dict,
            progress_callback=_progress,
            log_callback=_log,
            skip_callback=_skip,
        )
    except TypeError:
        prepared = strategy.prepare_data(data_dict)
    if prepared is None:
        raise RuntimeError("prepare_data が失敗しました")

    spy_df = prepared.get("SPY") if isinstance(prepared, dict) else None
    try:
        candidates = strategy.generate_candidates(
            prepared,
            progress_callback=_progress,
            log_callback=_log,
        )
    except TypeError:
        if spy_df is not None:
            candidates = strategy.generate_candidates(prepared, market_df=spy_df)
        else:
            candidates = strategy.generate_candidates(prepared)

    merged_df = None
    if isinstance(candidates, tuple) and len(candidates) == 2:
        candidates_by_date, merged_df = candidates
    else:
        candidates_by_date = candidates

    if not isinstance(candidates_by_date, dict) or not candidates_by_date:
        raise RuntimeError("候補データが取得できませんでした")

    filtered_candidates: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    for date_key, entries in candidates_by_date.items():
        ts = pd.to_datetime(date_key)
        if start_ts and ts < start_ts:
            continue
        if end_ts and ts > end_ts:
            continue
        normalized_entries: list[dict[str, Any]] = []
        for entry in entries:
            entry_copy = dict(entry)
            entry_copy.setdefault("symbol", entry.get("ticker") or entry.get("Symbol"))
            entry_copy.setdefault("entry_date", ts)
            normalized_entries.append(entry_copy)
        if normalized_entries:
            filtered_candidates[ts] = normalized_entries

    if not filtered_candidates:
        raise RuntimeError("日付条件に合致する候補がありませんでした")

    trades_df, _ = simulate_trades_with_risk(
        filtered_candidates,
        prepared if isinstance(prepared, dict) else data_dict,
        capital_val,
        strategy,
        on_progress=lambda done, total, _start=None: _progress(done, total),
        on_log=_log,
    )

    summary, enriched = summarize(trades_df, capital_val)
    summary_dict = {k: _normalize_value(v) for k, v in summary.to_dict().items()}
    trades_records = _records_from_df(enriched)

    if merged_df is not None and not isinstance(merged_df, pd.DataFrame):
        try:
            merged_df = pd.DataFrame(merged_df)
        except Exception:
            merged_df = None
    if isinstance(merged_df, pd.DataFrame) and not merged_df.empty:
        preview_json = merged_df.head(3).to_json(orient="records")
        log_messages.append(f"merged candidates preview: {preview_json}")

    if missing_symbols:
        log_messages.append(
            "missing symbols: " + ", ".join(sorted(set(missing_symbols)))
        )

    return BacktestResult(
        summary=summary_dict,
        trades=trades_records,
        logs=log_messages,
        missing_symbols=missing_symbols,
    )


def summarize_performance(
    trades_path: str, capital: float | None = None
) -> dict[str, Any]:
    path = _as_repo_path(trades_path)
    if not path.exists():
        raise FileNotFoundError(f"ファイルが存在しません: {path}")
    df = pd.read_csv(path)
    for col in ("entry_date", "exit_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    settings = _load_settings()
    initial_cap = (
        capital if capital is not None else float(settings.backtest.initial_capital)
    )
    summary, enriched = summarize(df, initial_cap)
    return {
        "summary": {k: _normalize_value(v) for k, v in summary.to_dict().items()},
        "trades": _records_from_df(enriched),
    }


def read_config_yaml(filename: str) -> dict[str, Any]:
    path = _as_repo_path(Path("config") / filename)
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルが存在しません: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = value
    return base


def write_config_yaml(
    filename: str,
    *,
    content: str | None = None,
    updates: dict[str, Any] | None = None,
    mode: str = "merge",
) -> dict[str, Any]:
    if content is None and updates is None:
        raise ValueError("content または updates のいずれかが必要です")
    path = _as_repo_path(Path("config") / filename)
    existing: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            existing = yaml.safe_load(fh) or {}
    if content is not None and mode == "overwrite":
        with path.open("w", encoding="utf-8") as fh:
            fh.write(content)
        result = yaml.safe_load(content) if content.strip() else {}
    else:
        if content is not None:
            updates = yaml.safe_load(content) or {}
        updates = updates or {}
        merged = _deep_merge(dict(existing), updates)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(merged, fh, allow_unicode=True, sort_keys=False)
        result = merged
    return result


def execute_daily_run() -> dict[str, Any]:
    cmd = [sys.executable, "-m", "scripts.daily_run"]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "command": " ".join(cmd),
    }


def search_project_files(
    pattern: str,
    *,
    file_globs: Sequence[str] | None = None,
    max_results: int = 100,
    case_sensitive: bool = False,
) -> list[dict[str, Any]]:
    default_globs = ["*.py", "*.md", "*.yaml", "*.yml", "*.json"]
    globs = list(file_globs) if file_globs else default_globs
    flags = 0 if case_sensitive else re.IGNORECASE
    regex = re.compile(pattern, flags)
    results: list[dict[str, Any]] = []
    for glob in globs:
        for path in REPO_ROOT.rglob(glob):
            is_hidden = any(
                part.startswith(".") and part not in {".vscode"}
                for part in path.parts
            )
            if is_hidden:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    results.append(
                        {
                            "path": str(path.relative_to(REPO_ROOT)),
                            "line": line_no,
                            "text": line.strip(),
                        }
                    )
                    if len(results) >= max_results:
                        return results
    return results


def find_symbol_references(symbol: str) -> dict[str, list[dict[str, Any]]]:
    definitions: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for path in REPO_ROOT.rglob("*.py"):
        if any(part.startswith(".") for part in path.parts):
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except Exception:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ) and node.name == symbol:
                definitions.append(
                    {
                        "path": str(path.relative_to(REPO_ROOT)),
                        "line": node.lineno,
                        "type": type(node).__name__,
                    }
                )
            elif isinstance(node, ast.Name) and node.id == symbol:
                references.append(
                    {
                        "path": str(path.relative_to(REPO_ROOT)),
                        "line": node.lineno,
                        "context": _extract_line(source, node.lineno),
                    }
                )
    return {"definitions": definitions, "references": references}


def _extract_line(source: str, lineno: int) -> str:
    lines = source.splitlines()
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].strip()
    return ""


def run_pytest(args: Sequence[str] | None = None) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "pytest"]
    if args:
        cmd.extend(args)
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "command": " ".join(cmd),
    }


def analyze_imports(path: str) -> list[dict[str, Any]]:
    target = _as_repo_path(path)
    if target.is_dir():
        raise ValueError("ファイルパスを指定してください")
    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.append(
                {
                    "module": None,
                    "names": [alias.name for alias in node.names],
                    "lineno": node.lineno,
                }
            )
        elif isinstance(node, ast.ImportFrom):
            imports.append(
                {
                    "module": node.module,
                    "names": [alias.name for alias in node.names],
                    "lineno": node.lineno,
                }
            )
    return imports


def run_python_file(path: str, args: Sequence[str] | None = None) -> dict[str, Any]:
    target = _as_repo_path(path)
    cmd = [sys.executable, str(target)]
    if args:
        cmd.extend(args)
    proc = subprocess.run(
        cmd,
        cwd=target.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "command": " ".join(cmd),
    }


def write_text_file(
    path: str, content: str, *, mode: str = "overwrite"
) -> dict[str, Any]:
    target = _as_repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode not in {"overwrite", "append"}:
        raise ValueError("mode は overwrite または append を指定してください")
    file_mode = "w" if mode == "overwrite" else "a"
    with target.open(file_mode, encoding="utf-8") as fh:
        fh.write(content)
        if not content.endswith("\n"):
            fh.write("\n")
    return {
        "path": str(target.relative_to(REPO_ROOT)),
        "bytes_written": len(content.encode("utf-8")),
        "mode": mode,
    }


def analyze_error_log(path: str, tail: int = 200) -> dict[str, Any]:
    target = _as_repo_path(path)
    if not target.exists():
        raise FileNotFoundError(f"ログファイルが存在しません: {target}")
    lines = deque(maxlen=tail)
    with target.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            lines.append(line.rstrip("\n"))
    error_segments: list[str] = []
    current: list[str] = []
    for line in lines:
        if "Traceback" in line:
            if current:
                error_segments.append("\n".join(current))
                current = []
            current.append(line)
        elif line.strip().startswith("File ") and current:
            current.append(line)
        elif "ERROR" in line:
            if current:
                error_segments.append("\n".join(current))
                current = []
            error_segments.append(line)
        elif current:
            current.append(line)
    if current:
        error_segments.append("\n".join(current))
    counter = Counter(error_segments)
    summary = [
        {
            "count": count,
            "sample": segment.splitlines()[0] if segment else "",
            "details": segment,
        }
        for segment, count in counter.most_common()
    ]
    return {
        "entries": summary,
        "tail": tail,
        "path": str(target.relative_to(REPO_ROOT)),
    }
