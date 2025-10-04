"""Seed minimal SPY rolling cache for today pipeline.

- Writes SPY.csv to data_cache/rolling via CacheManager (no direct CSV IO)
- Generates deterministic OHLCV for ~260 business days to satisfy SMA200
- Safe for local/dev use; does not call external APIs

Usage:
    python tools/seed_spy_cache.py

Notes:
- Respects project settings paths via get_settings(create_dirs=True)
- Keeps columns in lowercase (CacheManager normalizes on read)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common.cache_manager import CacheManager
from config.settings import get_settings


def _make_synthetic_spy(days: int = 260) -> pd.DataFrame:
    """Create deterministic synthetic daily OHLCV series.

    - date: last `days` business days ending today (tz-naive)
    - close: gently trending upward with small sinusoidal wiggle
    - open/high/low derived from close with small spreads
    - volume: constant-ish with slight noise
    """

    # Use business days to align with typical caches
    end = pd.Timestamp.today().normalize()
    idx = pd.bdate_range(end=end, periods=max(210, int(days)))  # ensure >= 210 for SMA200

    # Build a smooth, positive close series
    t = np.arange(len(idx))
    base = 400.0  # arbitrary starting level
    trend = t * 0.25  # slow uptrend
    wiggle = 2.0 * np.sin(t / 9.0)  # small oscillation
    close = base + trend + wiggle

    # Derive OHLC around close
    spread = 0.5 + 0.02 * np.cos(t / 7.0)
    open_ = close - 0.1
    high = close + spread
    low = close - spread

    volume = (5_000_000 + (100_000 * np.cos(t / 5.0))).astype(int)

    df = pd.DataFrame(
        {
            "date": idx,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    # CacheManager expects lowercase base columns; keep as-is
    return df


def seed_spy_rolling(days: int = 260) -> Path:
    """Write SPY rolling cache (CSV or Feather chosen by settings, auto->CSV).

    Returns the path where the cache was written (best-effort).
    """

    settings = get_settings(create_dirs=True)
    cm = CacheManager(settings)

    # Prepare synthetic OHLCV
    df = _make_synthetic_spy(days=days)

    # Write to rolling profile via CacheManager (atomic + formatting)
    cm.write_atomic(df, "SPY", profile="rolling")

    # Resolve the actually chosen path (auto -> csv by default when new)
    path = cm.file_manager.detect_path(cm.rolling_dir, "SPY")
    return path


def main() -> None:
    path = seed_spy_rolling(days=260)
    print(f"✅ Seeded SPY rolling cache at: {path}")


if __name__ == "__main__":
    main()
