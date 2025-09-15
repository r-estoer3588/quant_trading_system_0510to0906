"""Precompute shared indicators cache for the current universe.

Run outside market hours to warm the shared_indicators cache so that
runtime precomputation becomes differential and faster.
"""

from __future__ import annotations

import os

import pandas as pd

from common.indicators_precompute import precompute_shared_indicators
from common.universe import load_universe_file
from common.data_loader import load_price
from common.logging_utils import setup_logging
from config.settings import get_settings


def main() -> None:
    settings = get_settings(create_dirs=True)
    setup_logging(settings)

    symbols = load_universe_file()
    if not symbols:
        print("No universe loaded; nothing to precompute.")
        return

    # Load minimal price data for all symbols
    basic_data: dict[str, pd.DataFrame] = {}
    for s in symbols:
        try:
            df = load_price(s) or None
            if df is None or df.empty:
                continue
            basic_data[s] = df
        except Exception:
            continue

    if not basic_data:
        print("No data available; nothing to precompute.")
        return

    # Force parallel (can be limited by env THREADS_DEFAULT)
    os.environ.setdefault("PRECOMPUTE_PARALLEL", "1")
    os.environ.setdefault("PRECOMPUTE_PARALLEL_THRESHOLD", "0")

    precompute_shared_indicators(
        basic_data,
        log=lambda m: None,  # keep quiet in warm-up
        parallel=True,
        max_workers=getattr(settings, "THREADS_DEFAULT", 12),
    )
    print(f"Precomputed shared indicators for {len(basic_data)} symbols.")


if __name__ == "__main__":
    main()
