#!/usr/bin/env python3
"""Debug pipeline imports step by step."""

import time

print("Testing import sequence...")

modules = [
    ("pandas", "import pandas as pd"),
    ("common.cache_manager", "from common.cache_manager import CacheManager"),
    ("config.settings", "from config.settings import get_settings"),
    (
        "strategies.system1_strategy",
        "from strategies.system1_strategy import System1Strategy",
    ),
    (
        "strategies.system2_strategy",
        "from strategies.system2_strategy import System2Strategy",
    ),
    (
        "strategies.system3_strategy",
        "from strategies.system3_strategy import System3Strategy",
    ),
    (
        "strategies.system4_strategy",
        "from strategies.system4_strategy import System4Strategy",
    ),
    (
        "strategies.system5_strategy",
        "from strategies.system5_strategy import System5Strategy",
    ),
    (
        "strategies.system6_strategy",
        "from strategies.system6_strategy import System6Strategy",
    ),
    (
        "strategies.system7_strategy",
        "from strategies.system7_strategy import System7Strategy",
    ),
]

for name, import_stmt in modules:
    print(f"Importing {name}...")
    start = time.time()
    try:
        exec(import_stmt)
        elapsed = time.time() - start
        print(f"  ✓ {elapsed:.3f}s")
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ✗ {elapsed:.3f}s - {e}")
        break

print("All imports completed!")
