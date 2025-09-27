#!/usr/bin/env python3
"""Test cache_daily_data.py fix by creating new cache entries."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from common.cache_manager import CacheManager
from common.indicators_common import add_indicators
from config.settings import get_settings


def test_new_cache_format():
    """Test creating new cache with PascalCase columns."""
    print("🔍 新しいキャッシュ形式をテスト")

    settings = get_settings(create_dirs=True)
    cm = CacheManager(settings)

    # Simulate EODHD API data format (as returned by get_eodhd_data)
    test_data = {
        "Date": pd.date_range("2024-01-01", periods=100),
        "Open": [100.0 + i * 0.5 for i in range(100)],
        "High": [101.0 + i * 0.5 for i in range(100)],
        "Low": [99.0 + i * 0.5 for i in range(100)],
        "Close": [100.5 + i * 0.5 for i in range(100)],
        "AdjClose": [100.5 + i * 0.5 for i in range(100)],
        "Volume": [1000000 + i * 1000 for i in range(100)],
    }

    df = pd.DataFrame(test_data).set_index("Date")

    print(f"📊 元データ形式: {list(df.columns)}")

    # Apply indicators as cache_daily_data.py would do
    full_df = add_indicators(df.copy())

    print(f"🧮 指標追加後: {len(full_df.columns)} 列")
    print(f"📝 指標追加後列名: {list(full_df.columns)[:15]}...")

    # Apply the fixed processing (no .rename(columns=str.lower))
    df_reset = full_df.reset_index()

    print(f"💾 保存形式: {len(df_reset.columns)} 列")
    print(f"📝 保存形式列名: {list(df_reset.columns)[:15]}...")

    # Save to test location
    test_path = cm.full_dir / "TEST_NEW_FORMAT.csv"
    df_reset.to_csv(test_path, index=False)
    print(f"✅ テストファイル保存: {test_path}")

    # Load it back and check
    loaded = pd.read_csv(test_path)
    print(f"📂 ロード後: {len(loaded.columns)} 列")
    print(f"📝 ロード後列名: {list(loaded.columns)[:15]}...")

    # Check for duplicates
    all_cols = loaded.columns.tolist()
    seen = set()
    duplicates = []
    for col in all_cols:
        if col.lower() in seen:
            duplicates.append(col)
        else:
            seen.add(col.lower())

    if duplicates:
        print(f"🔴 重複列発見: {duplicates}")
    else:
        print("✅ 重複列なし!")

    # Test with build_rolling_with_indicators
    from scripts.build_rolling_with_indicators import _prepare_rolling_frame

    result = _prepare_rolling_frame(loaded, target_days=50)
    if result is not None:
        print(f"🔧 rolling処理後: {len(result.columns)} 列")

        # Check for duplicates after rolling processing
        all_cols = result.columns.tolist()
        seen = set()
        duplicates = []
        for col in all_cols:
            if col.lower() in seen:
                duplicates.append(col)
            else:
                seen.add(col.lower())

        if duplicates:
            print(f"🔴 rolling処理後に重複列: {duplicates}")
        else:
            print("✅ rolling処理後も重複列なし!")
    else:
        print("❌ rolling処理が失敗")


if __name__ == "__main__":
    test_new_cache_format()
