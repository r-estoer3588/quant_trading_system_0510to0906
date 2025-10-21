"""Reproduce finalize_allocation using candidates reconstructed from logs.

This script builds minimal DataFrames for system1 and system4 using the sample
rows found in logs/today_signals_20251021_1415.log and calls finalize_allocation
with ALLOCATION_DEBUG output enabled.

Run:
    $env:ALLOCATION_DEBUG='1'; python tools/repro_allocation_from_logs.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def make_row(symbol, system, entry_price, stop_price, score_key, score, atr20=None, atr10=None, atr40=None):
    return {
        "symbol": symbol,
        "system": system,
        "side": "long",
        "signal_type": "buy",
        "entry_date": pd.Timestamp("2025-10-21"),
        "entry_price": entry_price,
        "stop_price": stop_price,
        "score_key": score_key,
        "score": score,
        "score_rank": None,
        "score_rank_total": None,
        "reason": "repro",
        "atr10": atr10,
        "atr20": atr20,
        "atr40": atr40,
    }


def main():
    from core.final_allocation import finalize_allocation

    # Reconstruct candidates from the log's sample rows
    system1_rows = [
        make_row("SPY", "system1", 671.3, 636.5, "roc200", 14.54, atr20=6.96),
        make_row("AA", "system1", 38.96, 29.16, "roc200", 3.12, atr20=1.47),
        # third system1 candidate -- not present in final CSV, reconstruct from logs
        make_row("AACB", "system1", 12.0, 9.0, "roc200", 2.5, atr20=0.8),
    ]

    system4_rows = [
        make_row("A", "system4", 143.0, 138.155, "rsi4", 80.42, atr20=3.23, atr40=3.31),
        make_row("SPY", "system4", 671.3, 659.075, "rsi4", 68.31, atr20=6.96, atr40=6.58),
    ]

    per_system = {
        "system1": pd.DataFrame(system1_rows),
        "system4": pd.DataFrame(system4_rows),
    }

    final_df, summary = finalize_allocation(
        per_system,
        strategies=None,
        positions=None,
        symbol_system_map=None,
        long_allocations={"system1": 0.5, "system4": 0.5},
        short_allocations={},
        slots_long=5,
        slots_short=0,
        include_trade_management=False,
    )

    print("final_df:\n", final_df)
    print("summary.final_count:", summary.final_count)
    print("summary.final_symbols:", summary.final_symbols)


if __name__ == "__main__":
    main()
