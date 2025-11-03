#!/usr/bin/env python3
"""生成されたテストシンボルの値確認"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from common.cache_manager import CacheManager
from config.settings import get_settings

cache = CacheManager(get_settings())

# 各パターンから1つずつサンプル確認
test_symbols = [
    "SETUP_PASS_S1_00",
]

for sym in test_symbols:
    print(f"\n=== {sym} ===")
    df_full = cache.read(sym, "full")
    if df_full is not None and not df_full.empty:
        last = df_full.iloc[-1]
        print(f"\nFULL (rows={len(df_full)}):")
        print(f"  Close: {last.get('close', last.get('Close', 'N/A'))}")
        print(f"  SMA25: {last.get('sma25', 'N/A')}")
        print(f"  SMA50: {last.get('sma50', 'N/A')}")
        print(f"  ROC200: {last.get('roc200', 'N/A')}")

    df_rolling = cache.read(sym, "rolling")
    if df_rolling is not None and not df_rolling.empty:
        last = df_rolling.iloc[-1]
        print(f"\nROLLING (rows={len(df_rolling)}):")
        print(f"  Close: {last.get('close', last.get('Close', 'N/A'))}")
        print(f"  SMA25: {last.get('sma25', 'N/A')}")
        print(f"  SMA50: {last.get('sma50', 'N/A')}")
        print(f"  ROC200: {last.get('roc200', 'N/A')}")
