#!/usr/bin/env python3
"""Simple alternative pipeline test."""

from strategies.system1_strategy import System1Strategy
from strategies.system2_strategy import System2Strategy
from strategies.system3_strategy import System3Strategy
from strategies.system4_strategy import System4Strategy
from strategies.system5_strategy import System5Strategy
from strategies.system6_strategy import System6Strategy
from strategies.system7_strategy import System7Strategy
from config.settings import get_settings
from pathlib import Path


def test_basic_pipeline():
    """Test basic pipeline functionality without corrupted script."""
    print("1. Testing settings...")
    settings = get_settings(create_dirs=False)

    # Create required directories
    Path(settings.DATA_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.LOGS_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.outputs.signals_dir).mkdir(parents=True, exist_ok=True)
    print("✓ Settings and directories OK")

    print("2. Testing strategy initialization...")
    strategies = [
        System1Strategy(),
        System2Strategy(),
        System3Strategy(),
        System4Strategy(),
        System5Strategy(),
        System6Strategy(),
        System7Strategy(),
    ]
    print(f"✓ All {len(strategies)} strategies initialized")

    print("3. Success! Core components are working.")
    return True


if __name__ == "__main__":
    test_basic_pipeline()
