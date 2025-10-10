"""Quick test for system3 top-off logic"""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from common.cache_manager import CacheManager
from common.universe import load_universe_file
from config.settings import get_settings
from core.system3 import generate_candidates_system3


def test_system3_topoff():
    settings = get_settings(create_dirs=True)
    cm = CacheManager(settings)

    # Load first 200 symbols
    syms = load_universe_file()[:200]
    print(f"Testing with {len(syms)} symbols")

    # Load data using read method
    data = {}
    for sym in syms:
        try:
            df = cm.read(sym, "rolling")
            if df is not None and not df.empty:
                data[sym] = df
        except Exception:
            continue

    print(f"Loaded {len(data)} symbols with data")

    # Generate candidates
    log_messages = []

    def log_cb(msg):
        log_messages.append(msg)
        if "DEBUG_S3" in msg:
            print(f"  {msg}")

    result = generate_candidates_system3(
        data,
        top_n=10,
        latest_only=True,
        latest_mode_date="2025-10-07",
        log_callback=log_cb,
    )

    candidates, entry_map = result[0], result[1]

    print(f"\nResult: {len(candidates)} candidates")
    print("Expected: 10 (if STUpass >= 10)")

    # Show debug logs
    print("\nDebug logs:")
    for msg in log_messages:
        if "DEBUG_S3" in msg or "system3" in msg.lower():
            print(f"  {msg}")

    # Show candidates
    if candidates:
        print("\nCandidates:")
        for date, syms_dict in sorted(candidates.items()):
            print(f"  {date}: {len(syms_dict)} symbols - {list(syms_dict.keys())[:5]}...")


if __name__ == "__main__":
    test_system3_topoff()
