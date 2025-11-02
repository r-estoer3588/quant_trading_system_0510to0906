from __future__ import annotations

"""Quick debug runner: call get_today_signals for system1 and system4 and print attrs.

Run from repo root:
    python tools/debug_today_signals_quick.py

This script calls each strategy's get_today_signals and prints
`entry_skip_counts`, `entry_skip_details`, `entry_skip_samples` and a sample row.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import traceback

import pandas as pd

from strategies.system1_strategy import System1Strategy
from strategies.system4_strategy import System4Strategy


def run_one(StrategyCls, name: str) -> None:
    print(f"--- RUN {name} ---")
    stg = StrategyCls()
    try:
        df = stg.get_today_signals(
            {}, market_df=None, today=pd.Timestamp("2025-10-20"), log_callback=print
        )
    except Exception as e:
        print(f"{name} get_today_signals raised: {e}")
        traceback.print_exc()
        return

    if df is None or getattr(df, "empty", True):
        print(f"{name} -> empty DataFrame")
        return

    print(f"{name} -> rows: {len(df)}")
    for key in ("entry_skip_counts", "entry_skip_details", "entry_skip_samples"):
        print(f"attrs[{key}]:", df.attrs.get(key))

    print("-- sample row --")
    try:
        print(df.iloc[0].to_dict())
    except Exception:
        pass


if __name__ == "__main__":
    run_one(System1Strategy, "system1")
    run_one(System4Strategy, "system4")
    print("done")
