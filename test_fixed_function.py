#!/usr/bin/env python3
"""Test the actual _prepare_rolling_frame function with the fixes."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.build_rolling_with_indicators import _prepare_rolling_frame
from common.cache_manager import CacheManager
from config.settings import get_settings


def test_fixed_function():
    """Test the fixed _prepare_rolling_frame function."""
    print("🔍 修正後の _prepare_rolling_frame 関数をテスト")

    settings = get_settings(create_dirs=True)
    cm = CacheManager(settings)

    # Get a sample symbol
    csv_path = next(cm.full_dir.glob("*.csv"))
    symbol = csv_path.stem
    print(f"📊 シンボル: {symbol}")

    # Read raw data
    df = cm.read(symbol, "full")
    print(f"🔴 Raw データ: {len(df.columns)} 列")
    print(f"📝 Raw 列名: {list(df.columns)[:15]}...")

    # Test the fixed function
    result = _prepare_rolling_frame(df, target_days=330)

    if result is None:
        print("❌ 関数がNoneを返しました")
        return

    print(f"✅ 結果: {len(result.columns)} 列")
    print(f"📝 結果列名: {list(result.columns)[:15]}...")

    # Check for duplicates
    all_cols = result.columns.tolist()
    seen = set()
    duplicates = []
    for col in all_cols:
        if col.lower() in seen:
            duplicates.append(col)
        else:
            seen.add(col.lower())

    if duplicates:
        print(f"🔴 重複列が残存: {duplicates}")
    else:
        print("✅ 重複列なし!")


if __name__ == "__main__":
    test_fixed_function()
