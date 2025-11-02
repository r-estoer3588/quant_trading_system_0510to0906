"""Rebuild rolling cache for selected symbols and report indicator presence.

Usage:
    python tools/rebuild_rolling_for_symbols.py \
        --symbols BINI CARM ... [--workers 1] [--dry-run]

This script is conservative and works on local caches only (no external API
calls). It re-creates rolling files using the existing extractor and then
checks whether indicators such as ``drop3d`` and ``atr_ratio`` were written.
Results are saved as JSON in ``results_csv_test/rebuild_rolling_report_<ts>.json``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

# Ensure repository root is importable (tools/ scripts follow same pattern)
try:
    _HERE = Path(__file__).resolve().parents[1]
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
except Exception:
    pass

from common.cache_manager import CacheManager
from config.settings import get_settings
from scripts.build_rolling_with_indicators import extract_rolling_from_full

DEFAULT_SYMBOLS = ["BINI", "CARM", "CURIW", "ENFY", "PWM", "PXSAW", "SEV"]


def _last_non_nan_value(s: pd.Series | None) -> float | None:
    try:
        if s is None:
            return None
        s2 = pd.to_numeric(s, errors="coerce").dropna()
        if s2.empty:
            return None
        return float(s2.iloc[-1])
    except Exception:
        return None


def inspect_rolling_for_symbols(
    symbols: list[str], cm: CacheManager, out_path: Path
) -> dict[str, Any]:
    rc_dir = cm.rolling_dir
    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "symbols": {},
    }

    for sym in symbols:
        entry: dict[str, Any] = {
            "file_used": None,
            "exists": False,
            "rows": None,
            "columns": None,
            "drop3d_present": False,
            "last_drop3d": None,
            "atr_ratio_present": False,
            "last_atr_ratio": None,
            "error": None,
            "file_mtime": None,
        }
        try:
            # Prefer feather when present for performance
            feather_p = rc_dir / f"{sym}.feather"
            csv_p = rc_dir / f"{sym}.csv"
            chosen = None
            if feather_p.exists():
                chosen = feather_p
            elif csv_p.exists():
                chosen = csv_p

            if chosen is not None:
                entry["file_used"] = chosen.name
                entry["exists"] = True
                try:
                    entry["file_mtime"] = datetime.fromtimestamp(
                        chosen.stat().st_mtime
                    ).isoformat()
                except Exception:
                    entry["file_mtime"] = None

            # Read via CacheManager (normalised helpers applied)
            df = cm.read(sym, "rolling")
            if df is None:
                entry["error"] = "no_rolling_after_rebuild"
                summary["symbols"][sym] = entry
                continue

            entry["rows"] = int(len(df))
            try:
                cols = list(df.columns)
            except Exception:
                cols = []
            entry["columns"] = cols
            # Column names are lowercased by read path; check with lower-case
            lc = set([str(c).lower() for c in cols])
            entry["drop3d_present"] = "drop3d" in lc
            if entry["drop3d_present"]:
                entry["last_drop3d"] = _last_non_nan_value(df.get("drop3d"))
            entry["atr_ratio_present"] = "atr_ratio" in lc or "atr_pct" in lc
            if entry["atr_ratio_present"]:
                # prefer atr_ratio
                atr_val = _last_non_nan_value(df.get("atr_ratio") or df.get("atr_pct"))
                entry["last_atr_ratio"] = atr_val

        except Exception as e:  # pragma: no cover - defensive
            entry["error"] = f"inspect_error:{type(e).__name__}:{e}"

        summary["symbols"][sym] = entry

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception:
        pass
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Rebuild rolling cache and report whether key indicators were written."
        )
    )
    p.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help=(
            "Symbols to rebuild (space separated). " "Default: the 7 inspected symbols"
        ),
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers passed to extractor (default 1)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call extractor; only inspect existing rolling files",
    )
    p.add_argument(
        "--report-dir",
        type=str,
        default="results_csv_test",
        help="Output directory for JSON report",
    )
    args = p.parse_args(argv)

    settings = get_settings(create_dirs=True)
    cm = CacheManager(settings)

    symbols = [s.strip() for s in args.symbols if s and isinstance(s, str)]
    report_dir = Path(args.report_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"rebuild_rolling_report_{ts}.json"

    if not args.dry_run:
        print("Rebuilding rolling cache for: " + ", ".join(symbols))
        print(f"workers={args.workers}")
        try:
            stats = extract_rolling_from_full(
                cm,
                symbols=symbols,
                log=print,
                nan_warnings=True,
                workers=args.workers,
                adaptive=False,
            )
        except Exception as e:  # pragma: no cover - defensive catch
            print(f"Extraction failed: {e}")
            stats = None
    else:
        print("Dry-run: skipping extraction, only inspecting current rolling files")
        stats = None

    summary = inspect_rolling_for_symbols(symbols, cm, report_path)
    if stats is not None:
        try:
            summary["extract_stats"] = stats.to_dict()
        except Exception:
            summary["extract_stats"] = None

    print(f"Wrote JSON report: {report_path}")
    # Print concise per-symbol summary to stdout
    for sym, info in summary.get("symbols", {}).items():
        ok = "OK" if info.get("drop3d_present") else "MISSING-drop3d"
        rows = info.get("rows")
        last = info.get("last_drop3d")
        mtime = info.get("file_mtime")
        print(f"{sym}: {ok} rows={rows} last_drop3d={last} file_mtime={mtime}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
