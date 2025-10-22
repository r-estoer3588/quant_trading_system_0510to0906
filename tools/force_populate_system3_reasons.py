"""Force-populate df.attrs['_fdbg_reasons3'] by calling _system3_conditions with
FILTER_DEBUG enabled, for symbols listed in a diagnostics snapshot.

This script overwrites files in results_csv_test/system3_symbol_traces_20251022/ with
the same format as `extract_system3_symbol_traces.py`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import sys

# ensure repo root is on sys.path so `import common` works when running from tools/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument(
        "--out-dir",
        default="results_csv_test/system3_symbol_traces_20251022",
    )
    p.add_argument("--rows", type=int, default=20)
    return p.parse_args()


def _read_df(sym: str, rc_dir: Path) -> tuple[pd.DataFrame | None, str | None]:
    feather = rc_dir / f"{sym}.feather"
    csvp = rc_dir / f"{sym}.csv"
    if feather.exists():
        try:
            return pd.read_feather(feather), str(feather)
        except Exception:
            pass
    if csvp.exists():
        try:
            return pd.read_csv(csvp), str(csvp)
        except Exception:
            pass
    return None, None


def main() -> None:
    args = _parse_args()
    snap_p = Path(args.snapshot)
    if not snap_p.exists():
        print("snapshot missing", snap_p)
        raise SystemExit(2)
    with snap_p.open("r", encoding="utf-8") as fh:
        snap = json.load(fh)

    systems = snap.get("systems") or []
    sys3 = next((s for s in systems if s.get("system_id") == "system3"), None)
    if not sys3:
        print("no system3 in snapshot")
        raise SystemExit(3)
    excl = (sys3.get("diagnostics_extra") or {}).get("exclude_symbols") or {}

    # simple parsing like extractor
    def parse_setlike(s: Any) -> list[str]:
        if s is None:
            return []
        if isinstance(s, (list, tuple, set)):
            return list(map(str, s))
        if isinstance(s, str):
            s2 = s.strip()
            if s2.startswith("{") and s2.endswith("}"):
                inner = s2[1:-1].strip()
                if not inner:
                    return []
                parts = [p.strip().strip("'\"") for p in inner.split(",")]
                return [p for p in parts if p]
        return []

    drop = parse_setlike(excl.get("drop3d"))
    close = parse_setlike(excl.get("close_vs_sma150"))
    symbols = sorted(set(drop + close))
    rc_dir = Path("data_cache/rolling")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # import filter after ensuring env
    os.environ["FILTER_DEBUG"] = "1"
    try:
        from common.today_filters import _system3_conditions
    except Exception as e:
        print("failed to import system3_conditions:", e)
        raise

    for sym in symbols:
        df, src = _read_df(sym, rc_dir)
        out_p = out_dir / f"{sym}.txt"
        with out_p.open("w", encoding="utf-8") as fh:
            fh.write(f"symbol: {sym}\n")
            fh.write(f"source_file: {src}\n\n")
            if df is None:
                fh.write("NO DATAFRAME FOUND\n")
                continue
            # call system3 conditions to populate df.attrs when FILTER_DEBUG=1
            try:
                _system3_conditions(df)
            except Exception as e:
                fh.write(f"_system3_conditions error: {e}\n")
            try:
                tail = df.tail(args.rows)
                fh.write("--- tail rows (csv) ---\n")
                tail.to_csv(fh, index=False)
                fh.write("\n")
            except Exception as e:
                fh.write(f"FAILED_TO_WRITE_TAIL: {e}\n")
            try:
                reasons = None
                if hasattr(df, "attrs"):
                    reasons = df.attrs.get("_fdbg_reasons3")
                fh.write("--- _fdbg_reasons3 ---\n")
                fh.write(repr(reasons) + "\n")
            except Exception as e:
                fh.write(f"FAILED_TO_READ_REASONS: {e}\n")

    print("done")


if __name__ == "__main__":
    main()
