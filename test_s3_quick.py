"""Quick test for system3 top-off logic with full dataset"""

from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from common.cache_manager import CacheManager
from common.universe import load_universe_file
from config.settings import get_settings
from core.system3 import generate_candidates_system3


def test_system3():
    settings = get_settings(create_dirs=True)
    cm = CacheManager(settings)

    syms = load_universe_file()
    print(f"Loading {len(syms)} symbols...")

    data = {}
    for i, sym in enumerate(syms):
        if i % 500 == 0:
            print(f"  {i}/{len(syms)}")
        try:
            df = cm.read(sym, "rolling")
            if df is not None and not df.empty:
                data[sym] = df
        except Exception:
            continue

    print(f"Loaded {len(data)} symbols\n")

    log_messages = []

    def log_cb(msg):
        log_messages.append(msg)
        if "DEBUG_S3" in msg:
            print(msg)

    result = generate_candidates_system3(
        data,
        top_n=10,
        latest_only=True,
        latest_mode_date="2025-10-07",
        log_callback=log_cb,
    )

    candidates = result[0]
    total = sum(len(v) for v in candidates.values())

    print(f"\n✅ Result: {total} total candidates")
    print("Expected: 10")

    if candidates:
        for date, syms_dict in sorted(candidates.items()):
            syms_list = list(syms_dict.keys())
            print(f"  {date}: {len(syms_dict)} - {syms_list}")

    return total


if __name__ == "__main__":
    count = test_system3()
    exit(0 if count == 10 else 1)
