#!/usr/bin/env python3
"""Debug script to isolate create_dirs hang."""

from pathlib import Path
import time


def test_dir_creation():
    """Test directory creation step by step."""
    from config.settings import get_settings

    print("1. Getting settings without create_dirs...")
    start = time.time()
    settings = get_settings(create_dirs=False)
    print(f"   Done in {time.time() - start:.3f}s")

    dirs_to_create = [
        settings.DATA_CACHE_DIR,
        settings.DATA_CACHE_RECENT_DIR,
        settings.RESULTS_DIR,
        settings.LOGS_DIR,
        settings.outputs.signals_dir,
    ]

    print("2. Testing individual directory creation...")
    for i, p in enumerate(dirs_to_create, 1):
        try:
            print(f"   {i}. Creating {p}...")
            start = time.time()
            Path(p).mkdir(parents=True, exist_ok=True)
            elapsed = time.time() - start
            print(f"      Success in {elapsed:.3f}s")
        except Exception as e:
            print(f"      Error: {e}")

    print("3. Testing get_settings(create_dirs=True)...")
    start = time.time()
    get_settings(create_dirs=True)
    elapsed = time.time() - start
    print(f"   Done in {elapsed:.3f}s")


if __name__ == "__main__":
    test_dir_creation()
