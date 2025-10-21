"""Bulk recompute indicators for rolling cache.

Intended use: run as an ops/maintenance task (scheduled or manual) to repair
any rolling files that are missing required indicators. This avoids doing
per-read repairs which can slow down normal runtime behavior.

Usage:
  python scripts/recompute_rolling_bulk.py --workers 4 --symbols-file mylist.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from pathlib import Path as _Path
from typing import Any

# Ensure repository root is importable when running script directly
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.cache_manager import CacheManager  # noqa: E402 (sys.path set above)
from config.settings import get_settings  # noqa: E402 (sys.path manipulation above)


def _process_one(
    cm: CacheManager,
    symbol: str,
    execute: bool = False,
    backup: bool = False,
) -> tuple[str, bool, str | None, dict]:
    try:
        df = cm.read(symbol, "rolling")
        if df is None or df.empty:
            return (symbol, False, "no_data", {})

        # detect missing/NaN indicators
        required = ("drop3d", "atr_ratio", "dollarvolume20")
        missing = [c for c in required if c not in df.columns or df[c].isna().all()]
        meta: dict = {"missing": missing}
        if not missing:
            meta["would_update"] = False
            return (symbol, True, None, meta)

        # Recompute using existing cache manager helper
        recomputed = cm._recompute_indicators(df)
        # Validate
        ok = True
        for c in required:
            if c not in recomputed.columns or recomputed[c].dropna().empty:
                ok = False
                break

        # Prepare a small sample of recomputed values for review
        try:
            last_row = recomputed.tail(1).iloc[0].to_dict()
        except Exception:
            last_row = {}
        meta["recomputed_sample"] = {k: last_row.get(k) for k in ("drop3d", "atr_ratio", "dollarvolume20")}

        if not ok:
            meta["would_update"] = False
            return (symbol, False, "recompute_failed", meta)

        # If execute flag is not set, do not persist; just report what would be done
        if not execute:
            meta["would_update"] = True
            return (symbol, True, None, meta)

        # Optionally backup existing rolling files before overwrite
        if backup:
            try:
                from shutil import copy

                roll_dir = cm.rolling_dir
                for suf in (".feather", ".csv"):
                    p = roll_dir / f"{symbol}{suf}"
                    if p.exists():
                        dst = roll_dir / "backup"
                        dst.mkdir(parents=True, exist_ok=True)
                        copy(p, dst / p.name)
            except Exception:
                # Backup errors shouldn't block the update
                pass

        # Persist recomputed DataFrame
        try:
            cm.write_atomic(recomputed, symbol, "rolling")
            meta["would_update"] = True
            return (symbol, True, None, meta)
        except Exception as e:
            meta["would_update"] = False
            return (symbol, False, f"write_error:{e}", meta)
    except Exception as e:
        return (symbol, False, f"error:{e}", {})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Bulk recompute rolling indicators")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--symbols-file",
        type=str,
        default="",
        help="File with one symbol per line",
    )
    p.add_argument("--symbols", nargs="*", help="Symbols to process (overrides file)")
    p.add_argument(
        "--execute",
        action="store_true",
        help="If set, persist recomputed rolling files. Default: dry-run (no writes)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path for the JSON report (overrides default path)",
    )
    p.add_argument(
        "--backup",
        action="store_true",
        help="When --execute is used, back up existing rolling files before overwrite",
    )
    args = p.parse_args(argv)

    settings = get_settings(create_dirs=False)
    cm = CacheManager(settings)

    if args.symbols:
        symbols = args.symbols
    elif args.symbols_file:
        symbols = [s.strip() for s in Path(args.symbols_file).read_text().splitlines() if s.strip()]
    else:
        # Discover all symbols from full cache
        from common.symbols_manifest import load_symbol_manifest

        symbols = load_symbol_manifest(cm.full_dir) or []

    if not symbols:
        print("No symbols to process")
        return 0

    results: dict[str, Any] = {
        "total": len(symbols),
        "processed": 0,
        "updated": 0,  # actually updated on disk (execute=True)
        "would_update": 0,  # would be updated in dry-run
        "errors": {},
        "per_symbol": {},
    }
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_process_one, cm, sym, args.execute, args.backup): sym for sym in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                s_sym, success, message, meta = fut.result()
                results["processed"] += 1
                results["per_symbol"][sym] = {
                    "success": success,
                    "message": message,
                    "meta": meta,
                }
                if meta.get("would_update"):
                    results["would_update"] += 1
                if args.execute and success and meta.get("would_update"):
                    results["updated"] += 1
                if not success:
                    results["errors"][sym] = message
            except Exception as e:
                results["errors"][sym] = f"exception:{e}"

    if args.output:
        out_path = Path(args.output)
    else:
        now_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"recompute_rolling_bulk_report_{now_ts}.json"
        out_path = Path("results_csv_test") / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(results, ensure_ascii=False, indent=2)
    out_path.write_text(payload, encoding="utf-8")
    print("Wrote:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
