"""Quick debug helper to call finalize_allocation with minimal inputs.

Run from the repository root:

    python tools/debug_finalize_allocation.py

It will print the returned final_df shape and the AllocationSummary.system_diagnostics
so you can quickly see why allocations may be empty.
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

# Ensure repository root is on sys.path so 'core' package imports resolve when
# running this script directly.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def make_dummy_candidate(symbol: str, entry: float = 100.0, atr: float = 1.0) -> dict:
    return {
        "symbol": symbol,
        "entry_price": entry,
        "stop_price": entry - atr * 2,
        "atr10": atr,
        "Close": entry,
        "score": 1.0,
    }


def main() -> None:
    # Import here so sys.path modification above takes effect for package imports
    from core.final_allocation import finalize_allocation

    # Create minimal per_system dict with some candidates
    per_system = {
        "system1": pd.DataFrame([make_dummy_candidate("AAPL", 150.0, 1.5)]),
        "system3": pd.DataFrame([make_dummy_candidate("MSFT", 300.0, 2.0)]),
    }

    # Omit strategies and symbol_system_map to reproduce omission cases
    final_df, summary = finalize_allocation(
        per_system,
        strategies=None,
        positions=None,
        symbol_system_map=None,
        long_allocations={"system1": 0.5, "system3": 0.5},
        short_allocations={},
        slots_long=5,
        slots_short=0,
        include_trade_management=False,
    )

    print("final_df.shape:", getattr(final_df, "shape", None))
    print("final_count:", summary.final_count)
    print("final_counts:", summary.final_counts)
    print("final_symbols:", summary.final_symbols)
    print("final_long_count:", summary.final_long_count)
    print("final_short_count:", summary.final_short_count)
    print("system_diagnostics:")
    print(json.dumps(summary.system_diagnostics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
