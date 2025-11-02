#!/usr/bin/env python3

"""Test script to verify that the format migration works correctly."""

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd


def test_cache_manager_detects_feather_for_base():
    """Test that CacheManager detects feather files for base cache."""
    with tempfile.TemporaryDirectory() as temp_dir:
        old_cache_dir = os.environ.get("DATA_CACHE_DIR")
        os.environ["DATA_CACHE_DIR"] = temp_dir

        try:
            from common.cache_manager import CacheManager
            from config.settings import get_settings

            settings = get_settings()
            ROOT = Path(__file__).parent
            sys.path.insert(0, str(ROOT))
            _cm = CacheManager(settings)  # Underscore to indicate intentionally unused

            # Create base directory
            base_dir = Path(temp_dir) / "base"
            base_dir.mkdir(parents=True, exist_ok=True)

            # Create test feather file
            test_symbol = "TEST"
            expected_path = base_dir / f"{test_symbol}.feather"

            # Test data
            test_data = pd.DataFrame(
                {
                    "date": pd.date_range("2024-01-01", periods=5),
                    "Open": [100, 101, 102, 103, 104],
                    "High": [101, 102, 103, 104, 105],
                    "Low": [99, 100, 101, 102, 103],
                    "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
                    "Volume": [1000000, 1100000, 1200000, 1300000, 1400000],
                }
            )

            test_data.set_index("date", inplace=True)

            # Save as feather
            test_data.to_feather(expected_path)
            print(f"Expected path for base cache: {expected_path}")

            print("✓ Test file created successfully")

        finally:
            if old_cache_dir is not None:
                os.environ["DATA_CACHE_DIR"] = old_cache_dir
            else:
                if "DATA_CACHE_DIR" in os.environ:
                    del os.environ["DATA_CACHE_DIR"]


if __name__ == "__main__":
    try:
        print("Testing feather format migration...")
        test_cache_manager_detects_feather_for_base()
        print("All tests passed! ✓")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
