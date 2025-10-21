#!/usr/bin/env python3
"""Replay finalize_allocation using persisted per-system feather files.

Reads files matching results_csv_test/per_system_*.feather, sets
ALLOCATION_DEBUG=1 in-process, calls finalize_allocation and prints
the ALLOC_DEBUG logs and summary to stdout.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd


def main() -> None:
    # Ensure allocator debug mode is enabled for this process
    os.environ.setdefault("ALLOCATION_DEBUG", "1")

    # Configure simple logging to stdout to capture ALLOC_DEBUG messages
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    repo_root = Path(__file__).resolve().parents[1]
    search_dir = repo_root / "results_csv_test"
    pattern = search_dir / "per_system_*.feather"
    files = sorted(pattern.parent.glob(pattern.name))

    if not files:
        print(f"No per-system feather files found at {search_dir!s}")
        return

    per_system: dict[str, pd.DataFrame] = {}
    for p in files:
        try:
            df = pd.read_feather(p)
        except Exception as e:
            print(f"Failed to read {p}: {e}")
            continue
        key = p.stem.replace("per_system_", "").strip().lower()
        per_system[key] = df
        print(f"Loaded {p.name}: rows={len(df)} columns={list(df.columns)[:6]}...")
        if len(df) > 0:
            print(f" sample: {dict(df.iloc[0].to_dict())}")

    # Defer import until after env/logging set
    try:
        # Ensure repo root is on sys.path so package imports like `core.*` work
        import sys

        sys.path.insert(0, str(repo_root))
        from core.final_allocation import (
            finalize_allocation,
            to_allocation_summary_dict,
        )
    except Exception as e:
        print(f"Could not import finalize_allocation: {e}")
        return

    print("\nCalling finalize_allocation with ALLOCATION_DEBUG=1...\n")
    try:
        final_df, summary = finalize_allocation(
            per_system, include_trade_management=False
        )
    except Exception as e:
        print(f"finalize_allocation raised exception: {e}")
        return

    print("\n=== Allocation Summary ===")
    try:
        summary_dict = to_allocation_summary_dict(summary)
        print(json.dumps(summary_dict, indent=2, default=str, ensure_ascii=False))
    except Exception:
        print(repr(summary))

    print("\n=== Final frame (top rows) ===")
    try:
        if final_df is None or final_df.empty:
            print("(no rows allocated)")
        else:
            # show only a few columns for brevity
            cols = [
                c
                for c in [
                    "symbol",
                    "system",
                    "shares",
                    "position_value",
                    "entry_price",
                    "stop_price",
                    "side",
                ]
                if c in final_df.columns
            ]
            print(final_df[cols].head(50).to_string(index=False))
    except Exception as e:
        print(f"Could not print final_df: {e}")


if __name__ == "__main__":
    main()
