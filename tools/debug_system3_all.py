"""Run System3 filter debug across the whole rolling cache.

Usage:
    python tools/debug_system3_all.py [--out OUT_CSV] [--workers N]
        [--max-symbols N] [--test-mode MODE]

This script reads each per-symbol rolling cache file from data_cache/rolling,
applies the System3 pre-filter (_system3_conditions) and collects per-symbol
reasons plus a handful of last-row indicator values. When FILTER_DEBUG is
set (the script enables it by default) the underlying filter attaches
per-symbol reasons to df.attrs which are preserved in the CSV output.

Outputs:
    - CSV (rows per symbol):
            symbol,file_used,low_ok,avgvol_ok,atr_ok,reason,close,
            dollarvolume20,atr_ratio,drop3d,sma150,error
  - Prints a short summary of reason counts and a few examples.

Notes:
  - This can take minutes for a large cache. Use --max-symbols to limit.
  - Default worker parallelism is 8; set --workers 1 for sequential execution.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd

# Make sure repository root is importable when running from tools/
try:
    _HERE = Path(__file__).resolve().parents[1]
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
except Exception:
    pass


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="System3 filter debug over rolling cache")
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output CSV path (default: "
            "results_csv_test/system3_filter_debug_all_<ts>.csv)"
        ),
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel worker threads (default: 8)",
    )
    p.add_argument(
        "--max-symbols", type=int, default=0, help="Limit processed symbols (0 = all)"
    )
    p.add_argument(
        "--test-mode",
        type=str,
        default=None,
        help="Set TEST_MODE env var for filter test-mode (e.g. mini)",
    )
    return p.parse_args()


def _safe_read_df(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    """Read a feather or csv at path. Returns (df, error_message)."""
    try:
        if path.suffix.lower() == ".feather":
            try:
                df = pd.read_feather(path)
                return df, None
            except Exception:
                # fallback to CSV read
                pass
        # CSV fallback (attempt to read index column if present)
        try:
            df = pd.read_csv(path)
            return df, None
        except Exception as e:
            return None, f"read_csv_error: {e}"
    except Exception as e:
        return None, f"read_error: {e}"


def _process_symbol(
    symbol: str,
    rc_dir: Path,
    sys3_fn: Any,
    pick_fn: Any,
    last_scalar_fn: Any,
    thr_drop: float | None,
) -> dict:
    row: dict[str, Any] = {
        "symbol": symbol,
        "file_used": None,
        "low_ok": None,
        "avgvol_ok": None,
        "atr_ok": None,
        "reason": None,
        "close": None,
        "dollarvolume20": None,
        "atr_ratio": None,
        "drop3d": None,
        "sma150": None,
        "error": None,
    }
    feather_p = rc_dir / f"{symbol}.feather"
    csv_p = rc_dir / f"{symbol}.csv"
    chosen = None
    if feather_p.exists():
        chosen = feather_p
    elif csv_p.exists():
        chosen = csv_p
    else:
        row["error"] = "no_cache_file"
        return row

    row["file_used"] = chosen.name
    df, err = _safe_read_df(chosen)
    if df is None:
        row["error"] = err
        return row

    try:
        # Call System3 pre-filter; this will attach debug reasons to df.attrs
        low_ok, av_ok, atr_ok = sys3_fn(df)
        row["low_ok"] = bool(low_ok)
        row["avgvol_ok"] = bool(av_ok)
        row["atr_ok"] = bool(atr_ok)
    except Exception as e:
        row["error"] = f"filter_error: {e}"
        return row

    # Preferred: reason from df.attrs (set when FILTER_DEBUG=1)
    try:
        if hasattr(df, "attrs"):
            reasons = df.attrs.get("_fdbg_reasons3")
            if isinstance(reasons, list) and reasons:
                row["reason"] = reasons[-1]
    except Exception:
        pass

    # Fallback reason derivation
    if not row["reason"]:
        if row["low_ok"] is False:
            row["reason"] = "low_fail"
        elif row["avgvol_ok"] is False:
            row["reason"] = "avgvol_fail"
        elif row["atr_ok"] is False:
            # try to inspect ATR series to distinguish missing vs below
            atr_val = last_scalar_fn(
                pick_fn(df, ["ATR_Ratio", "ATR_Pct", "atr_ratio", "atr_pct"])
            )
            if atr_val is None:
                row["reason"] = "atr_missing"
            else:
                row["reason"] = "atr_below"
        else:
            # all pre-filters passed (setup predicate may still fail on drop3d)
            # check drop3d to provide additional signal
            drop_val = last_scalar_fn(pick_fn(df, ["Drop3D", "drop3d", "drop_3d"]))
            try:
                if drop_val is None:
                    row["reason"] = "drop3d_missing"
                elif thr_drop is not None and float(drop_val) < float(thr_drop):
                    row["reason"] = "drop3d_below_thr"
                else:
                    row["reason"] = "pass_pre_filters"
            except Exception:
                row["reason"] = "pass_pre_filters"

    # Collect last-row values for primary columns (use helpers for robust retrieval)
    try:
        row["close"] = last_scalar_fn(pick_fn(df, ["Close", "close"]))
    except Exception:
        row["close"] = None
    try:
        row["dollarvolume20"] = last_scalar_fn(
            pick_fn(
                df,
                [
                    "DollarVolume20",
                    "dollarvolume20",
                    "dollar_volume20",
                    "DV20",
                ],
            )
        )
    except Exception:
        row["dollarvolume20"] = None
    try:
        row["atr_ratio"] = last_scalar_fn(
            pick_fn(df, ["ATR_Ratio", "ATR_Pct", "atr_ratio", "atr_pct"])  # noqa: E501
        )
    except Exception:
        row["atr_ratio"] = None
    try:
        row["drop3d"] = last_scalar_fn(pick_fn(df, ["Drop3D", "drop3d", "drop_3d"]))
    except Exception:
        row["drop3d"] = None
    try:
        row["sma150"] = last_scalar_fn(pick_fn(df, ["SMA150", "sma150", "sma_150"]))
    except Exception:
        row["sma150"] = None

    return row


def main() -> None:
    args = _parse_args()
    if args.test_mode:
        os.environ.setdefault("TEST_MODE", str(args.test_mode))
    # Enable filter debug by default for this diagnostic run
    os.environ.setdefault("FILTER_DEBUG", "1")

    rc_dir = Path("data_cache/rolling")
    if not rc_dir.exists():
        print(f"Rolling cache directory not found: {rc_dir}")
        raise SystemExit(1)

    # Import filter helpers after setting environment vars so get_env_config sees them
    try:
        from common.today_filters import _last_scalar, _pick_series, _system3_conditions
    except Exception as e:
        print("Failed to import common.today_filters:", e)
        raise

    # Collect unique symbol stems (prefer feather when present)
    stems = sorted(
        {p.stem for p in rc_dir.iterdir() if p.suffix.lower() in (".feather", ".csv")}
    )
    total = len(stems)
    if args.max_symbols and int(args.max_symbols) > 0:
        max_symbols = int(args.max_symbols)
    else:
        max_symbols = None
    if max_symbols is not None:
        stems = stems[:max_symbols]
    print(f"Found {total} symbols in rolling cache;")
    print(f"Processing {len(stems)} symbols (workers={args.workers})")

    if args.out:
        out_p = Path(args.out)
    else:
        out_p = (
            Path("results_csv_test")
            / f"system3_filter_debug_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "symbol",
        "file_used",
        "low_ok",
        "avgvol_ok",
        "atr_ok",
        "reason",
        "close",
        "dollarvolume20",
        "atr_ratio",
        "drop3d",
        "sma150",
        "error",
    ]

    # Determine drop3d threshold from env (if set via env.get_env_config or fallback)
    thr_drop = None
    try:
        from config.environment import get_env_config

        env = get_env_config()
        if env is not None and env.min_drop3d_for_test is not None:
            thr_drop = float(env.min_drop3d_for_test)
    except Exception:
        thr_drop = None

    results: list[dict[str, Any]] = []

    # Process with thread pool for better throughput on I/O bound workload
    workers = int(args.workers) if args.workers and int(args.workers) > 0 else 1
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for sym in stems:
            fut = ex.submit(
                _process_symbol,
                sym,
                rc_dir,
                _system3_conditions,
                _pick_series,
                _last_scalar,
                thr_drop,
            )
            futures[fut] = sym
        processed = 0
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {"symbol": sym, "error": f"exception_in_worker: {e}"}
            results.append(row)
            processed += 1
            if processed % 200 == 0:
                print(f"Processed {processed}/{len(stems)}")

    # Write CSV
    try:
        with out_p.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r.get(k) for k in fieldnames})
        print(f"Wrote per-symbol debug CSV: {out_p}")
    except Exception as e:
        print("Failed to write CSV:", e)

    # Print summary counts
    counts: dict[str, int] = {}
    samples: dict[str, list[str]] = {}
    for r in results:
        rc = r.get("reason") or str(r.get("error") or "unknown")
        counts[rc] = counts.get(rc, 0) + 1
        samples.setdefault(rc, []).append(str(r.get("symbol") or ""))

    print("\n=== Reason distribution ===")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        sample_list = ",".join((samples.get(k) or [])[:5])
        print(f"{k}: {v} (examples: {sample_list})")

    # Save a small JSON summary next to CSV
    try:
        summary = {
            "generated_at": datetime.now().isoformat(),
            "total_processed": len(results),
            "reason_counts": counts,
            "csv": str(out_p),
        }
        summary_p = out_p.with_suffix(".summary.json")
        with summary_p.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        print(f"Wrote summary JSON: {summary_p}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
