#!/usr/bin/env python3
"""Fake symbol data harness for System4/System5 filter debugging.

Steps:
1. Load feather files from data_cache/test_symbols.
2. Synthesize missing indicator columns required by _system4/_system5 conditions:
   - DollarVolume50: average of (Close * Volume) last 50 rows (or boosted for S4 pass symbols).
   - HV50: constant 20 (within 10~40 pass band) for symbols containing 'S4'; 50 for others (fail band) to show discrimination.
   - AvgVolume50: average Volume last 50 rows.
   - ATR_Pct: 0.05 (>2.5%) for symbols containing 'S5'; 0.01 otherwise.
3. Run filter_system4 / filter_system5 with stats collection and print detailed per-symbol condition tuples.

Usage:
  python scripts/debug_system45_with_fake.py

Environment flags respected:
  DEBUG_SYSTEM_FILTERS=1  (will cause today_filters debug prints if filters invoked elsewhere; here we print explicitly)
"""
from __future__ import annotations

import pandas as pd

from common.today_filters import (
    _system4_conditions,
    _system5_conditions,
    filter_system4,
    filter_system5,
)
from config.settings import get_settings


def load_fake_frames():
    st = get_settings()
    d = st.DATA_CACHE_DIR / "test_symbols"
    if not d.exists():
        raise SystemExit(f"test_symbols dir missing: {d}")
    frames: dict[str, pd.DataFrame] = {}
    for f in d.glob("*.feather"):
        try:
            df = pd.read_feather(f)
        except Exception as e:  # pragma: no cover - diagnostic path
            print(f"❌ read fail {f.name}: {e}")
            continue
        frames[f.stem] = df
    return frames


def ensure_columns(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """Add / override needed columns (vectorized constant fill is fine)."""
    # Volume may not be first 5 printed; assume exists.
    if "Volume" not in df.columns:
        # Fallback: create synthetic volume if absent.
        df["Volume"] = 1_000_000
    tail = df.tail(50)
    try:
        raw_dv50 = float((tail["Close"] * tail["Volume"]).mean())
    except Exception:
        raw_dv50 = 0.0
    dv50 = raw_dv50
    if "S4" in symbol and dv50 < 100_000_000:
        # Boost to guarantee pass for System4 designated symbols.
        dv50 = 150_000_000.0
    if "DollarVolume50" not in df.columns:
        df["DollarVolume50"] = dv50  # constant; cheaper than per-row recompute
    # HV50 discrimination
    hv_val = 20.0 if "S4" in symbol else 50.0
    if "HV50" not in df.columns:
        df["HV50"] = hv_val
    # AvgVolume50
    try:
        av50 = float(tail["Volume"].mean())
    except Exception:
        av50 = 1_000_000.0
    if "AvgVolume50" not in df.columns:
        df["AvgVolume50"] = av50
    # ATR_Pct discrimination
    atr_val = 0.05 if "S5" in symbol else 0.01
    if "ATR_Pct" not in df.columns and "ATR_Ratio" not in df.columns:
        df["ATR_Pct"] = atr_val
    return df


def main():
    frames = load_fake_frames()
    enriched: dict[str, pd.DataFrame] = {}
    for sym, df in frames.items():
        enriched[sym] = ensure_columns(sym, df.copy())
    # Focus symbol groups
    target_syms_s4 = [s for s in enriched if "S4" in s]
    target_syms_s5 = [s for s in enriched if "S5" in s]

    print("=== System4 candidate symbols:", target_syms_s4)
    for sym in target_syms_s4:
        dv_ok, hv_ok = _system4_conditions(enriched[sym])
        print(
            f"[S4 cond] {sym}: dv_ok={dv_ok} hv_ok={hv_ok}  (DollarVolume50={enriched[sym]['DollarVolume50'].iloc[-1]:.2f} HV50={enriched[sym]['HV50'].iloc[-1]:.2f})"
        )

    print("\n=== System5 candidate symbols:", target_syms_s5)
    for sym in target_syms_s5:
        av_ok, dv_ok, atr_ok = _system5_conditions(enriched[sym])
        atr_series = None
        if "ATR_Pct" in enriched[sym].columns:
            atr_series = enriched[sym]["ATR_Pct"]
        elif "ATR_Ratio" in enriched[sym].columns:
            atr_series = enriched[sym]["ATR_Ratio"]
        atr_last = (
            float(atr_series.iloc[-1])
            if atr_series is not None and len(atr_series)
            else float("nan")
        )
        print(
            f"[S5 cond] {sym}: av_ok={av_ok} dv_ok={dv_ok} atr_ok={atr_ok} (AvgVolume50={enriched[sym]['AvgVolume50'].iloc[-1]:.0f} DV50={enriched[sym]['DollarVolume50'].iloc[-1]:.2f} ATR_Pct={atr_last:.3f})"
        )

    # Run filters with stats using only target groups (simulate pre-filter symbol universe subset)
    from collections import defaultdict

    stats4: dict[str, int] = defaultdict(int)
    stats5: dict[str, int] = defaultdict(int)
    passed4 = filter_system4(target_syms_s4, enriched, stats4)
    passed5 = filter_system5(target_syms_s5, enriched, stats5)

    print("\n--- Filter Aggregates ---")
    print("System4 stats:", dict(stats4), "passed symbols:", passed4)
    print("System5 stats:", dict(stats5), "passed symbols:", passed5)

    if not passed4:
        print("⚠️ EXPECTED at least one System4 pass; verify DV50/HV50 injection logic.")
    if not passed5:
        print("⚠️ EXPECTED at least one System5 pass; verify ATR_Pct injection logic.")


if __name__ == "__main__":  # pragma: no cover
    main()
