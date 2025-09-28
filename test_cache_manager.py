#!/usr/bin/env python3

import os
import sys

# プロジェクトルートディレクトリを追加
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from common.cache_manager import CacheManager
from config.settings import get_settings

print("Testing CacheManager SPY read functionality...")

settings = get_settings()
cache_manager = CacheManager(settings)

# 各プロファイルでSPYを試す
for profile in ["full", "base", "rolling"]:
    print(f"\n=== Profile: {profile} ===")
    try:
        data = cache_manager.read("SPY", profile)
        if data is not None and not data.empty:
            print(f"✅ Successfully read {len(data)} rows")
            print(f"Date range: {data['date'].min()} to {data['date'].max()}")
            print(f"Columns: {len(data.columns)} columns")
            print(f"Recent dates: {data['date'].tail(3).tolist()}")
        else:
            print("❌ No data or empty DataFrame")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
