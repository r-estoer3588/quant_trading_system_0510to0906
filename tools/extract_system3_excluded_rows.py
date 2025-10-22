#!/usr/bin/env python3
"""
Extract rows from a system3 debug CSV for symbols listed in a diagnostics snapshot.

Usage:
  python tools/extract_system3_excluded_rows.py \
      --snapshot results_csv_test/diagnostics_test/diagnostics_snapshot_20251022_135717.json \
      --csv results_csv_test/system3_filter_debug_all_20251022_minatr0.03.csv

Prints a compact CSV to stdout with selected columns for the excluded symbols.
"""
import argparse
import ast
import json
import sys
from pathlib import Path

import pandas as pd


def parse_set_string(s):
    # snapshot stores exclude_symbols as Python set string like "{'A','B'}"
    if not s:
        return []
    try:
        return sorted(list(ast.literal_eval(s)))
    except Exception:
        # fallback: crude parsing
        s2 = s.strip().lstrip("{\n").rstrip("}\n")
        parts = [p.strip().strip("'\" ") for p in s2.split(",") if p.strip()]
        return sorted(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument("--csv", required=True)
    args = p.parse_args()

    snap_path = Path(args.snapshot)
    csv_path = Path(args.csv)
    if not snap_path.exists():
        print("snapshot not found:", snap_path, file=sys.stderr)
        sys.exit(2)
    if not csv_path.exists():
        print("csv not found:", csv_path, file=sys.stderr)
        sys.exit(2)

    snap = json.loads(snap_path.read_text())

    # find system3 block
    sys3 = None
    for s in snap.get("systems", []):
        if s.get("system_id") == "system3":
            sys3 = s
            break
    if sys3 is None:
        print("system3 not found in snapshot", file=sys.stderr)
        sys.exit(2)

    extra = sys3.get("diagnostics_extra", {})
    exclude = extra.get("exclude_symbols", {})
    drop3d_set = parse_set_string(exclude.get("drop3d"))
    close_set = parse_set_string(exclude.get("close_vs_sma150"))

    target = sorted(set(drop3d_set + close_set))
    if not target:
        print("no excluded symbols found in snapshot")
        return

    # load CSV and filter
    df = pd.read_csv(csv_path)
    # Normalize symbol column
    if "symbol" not in df.columns:
        print("csv missing 'symbol' column", file=sys.stderr)
        sys.exit(2)

    found = df[df["symbol"].isin(target)].copy()
    # select useful columns
    cols = [
        c
        for c in [
            "symbol",
            "reason",
            "close",
            "sma150",
            "dollarvolume20",
            "atr_ratio",
            "drop3d",
        ]
        if c in found.columns
    ]
    if found.empty:
        print("No matching rows found in CSV for snapshot symbols")
        # still print the target list
        print("\n".join(target))
        return

    out = found[cols].sort_values("symbol")
    # print as CSV to stdout
    print(out.to_csv(index=False))


if __name__ == "__main__":
    main()
