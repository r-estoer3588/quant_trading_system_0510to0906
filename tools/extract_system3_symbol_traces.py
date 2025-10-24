"""Extract per-symbol recent rows and FILTER_DEBUG reasons for system3 excluded symbols.

Usage:
    python tools/extract_system3_symbol_traces.py --snapshot <snapshot.json> [--out-dir <dir>] [--rows N]

This reads the diagnostics snapshot, parses `system3` -> `exclude_symbols`, and for
each symbol loads the rolling cache DataFrame from `data_cache/rolling/{SYM}.feather` or
`{SYM}.csv`. It writes a text file per symbol containing the last N rows (as CSV)
and the `df.attrs['_fdbg_reasons3']` list if present.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from pathlib import Path as _Path
from typing import Any

import pandas as pd

# ensure repo root is on sys.path
ROOT = _Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from common.today_filters import _system3_conditions
except Exception:
    _system3_conditions = None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract system3 symbol traces")
    p.add_argument(
        "--snapshot",
        required=True,
        help="Path to diagnostics snapshot JSON",
    )
    p.add_argument(
        "--out-dir",
        default="results_csv_test/system3_symbol_traces_20251022",
        help="Directory to write per-symbol outputs",
    )
    p.add_argument(
        "--rows",
        type=int,
        default=20,
        help="Number of tail rows to save per symbol",
    )
    return p.parse_args()


def _parse_setlike(s: Any) -> list[str]:
    # snapshot stores sets as strings like "{'A','B'}" — try ast.literal_eval
    if s is None:
        return []
    if isinstance(s, (list, tuple, set)):
        return list(map(str, s))
    if isinstance(s, str):
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple, set)):
                return [str(x) for x in val]
        except Exception:
            # fallback: simple split on commas inside braces
            t = s.strip()
            if t.startswith("{") and t.endswith("}"):
                inner = t[1:-1].strip()
                if not inner:
                    return []
                parts = [p.strip().strip("'\"") for p in inner.split(",")]
                return [p for p in parts if p]
    return []


def _read_df_for_symbol(
    sym: str, rc_dir: Path
) -> tuple[pd.DataFrame | None, str | None]:
    feather_p = rc_dir / f"{sym}.feather"
    csv_p = rc_dir / f"{sym}.csv"
    if feather_p.exists():
        try:
            return pd.read_feather(feather_p), str(feather_p)
        except Exception:
            pass
    if csv_p.exists():
        try:
            return pd.read_csv(csv_p), str(csv_p)
        except Exception:
            pass
    return None, None


def main() -> None:
    args = _parse_args()
    snap_p = Path(args.snapshot)
    if not snap_p.exists():
        print("Snapshot not found:", snap_p)
        raise SystemExit(2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with snap_p.open("r", encoding="utf-8") as fh:
        snap = json.load(fh)

    # find system3 block
    systems = snap.get("systems") or []
    sys3 = None
    for s in systems:
        if s.get("system_id") == "system3":
            sys3 = s
            break
    if not sys3:
        print("system3 block not found in snapshot")
        raise SystemExit(3)

    extra = sys3.get("diagnostics_extra") or {}
    excl = extra.get("exclude_symbols") or {}
    drop_set = _parse_setlike(excl.get("drop3d"))
    close_set = _parse_setlike(excl.get("close_vs_sma150"))
    symbols = sorted(set(drop_set + close_set))
    print(f"Found {len(symbols)} symbols in snapshot exclude_symbols")

    rc_dir = Path("data_cache/rolling")
    for sym in symbols:
        df, src = _read_df_for_symbol(sym, rc_dir)
        out_p = out_dir / f"{sym}.txt"
        with out_p.open("w", encoding="utf-8") as fh:
            fh.write(f"symbol: {sym}\n")
            fh.write(f"source_file: {src}\n\n")
            if df is None:
                fh.write("NO DATAFRAME FOUND\n")
                continue
            # Ensure persistent debug column exists in the tail: prefer column
            # `_dbg_reasons3`. If attrs do not have _fdbg_reasons3, attempt to
            # call _system3_conditions (best-effort) to populate debug reasons.
            try:
                if "_dbg_reasons3" not in df.columns:
                    # If attrs already contain debug reasons, use them
                    last_reason = None
                    if hasattr(df, "attrs") and df.attrs.get("_fdbg_reasons3"):
                        try:
                            ar = df.attrs.get("_fdbg_reasons3")
                            if isinstance(ar, (list, tuple)) and ar:
                                last_reason = ar[-1]
                            else:
                                last_reason = ar
                        except Exception:
                            last_reason = None
                        try:
                            df = df.copy()
                            df["_dbg_reasons3"] = last_reason
                        except Exception:
                            pass
                    else:
                        # Try to compute reasons by calling _system3_conditions on a copy
                        try:
                            if _system3_conditions is not None:
                                tmp = df.copy()
                                try:
                                    _system3_conditions(tmp)
                                    # if attrs populated, extract last reason
                                    if hasattr(tmp, "attrs"):
                                        ar2 = tmp.attrs.get("_fdbg_reasons3")
                                        if ar2:
                                            if isinstance(ar2, (list, tuple)) and ar2:
                                                last_reason = ar2[-1]
                                            else:
                                                last_reason = ar2
                                            df = df.copy()
                                            df["_dbg_reasons3"] = last_reason
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        # fallback: ensure column exists
                        if "_dbg_reasons3" not in df.columns:
                            try:
                                df = df.copy()
                                df["_dbg_reasons3"] = None
                            except Exception:
                                pass
            except Exception:
                pass

            # Dump last N rows as CSV
            try:
                tail = df.tail(args.rows)
                fh.write("--- tail rows (csv) ---\n")
                tail.to_csv(fh, index=False)
                fh.write("\n")
            except Exception as e:
                fh.write(f"FAILED_TO_WRITE_TAIL: {e}\n")

            # Dump debug reasons if present
            try:
                reasons = None
                if hasattr(df, "attrs"):
                    reasons = df.attrs.get("_fdbg_reasons3")
                fh.write("--- _fdbg_reasons3 ---\n")
                fh.write(repr(reasons) + "\n")
            except Exception as e:
                fh.write(f"FAILED_TO_READ_REASONS: {e}\n")

    print("Wrote per-symbol trace files to:", out_dir)


if __name__ == "__main__":
    main()
