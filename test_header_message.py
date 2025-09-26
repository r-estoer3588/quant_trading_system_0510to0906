#!/usr/bin/env python3
"""
Test script for the header message functionality in execute_today_signals.
"""

import sys
from pathlib import Path

from app_today_signals import RunConfig, execute_today_signals

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_header_message():
    """Test that the header message is displayed correctly."""
    print("Testing header message functionality...")

    # Create a minimal RunConfig for testing
    test_config = RunConfig(
        symbols=["AAPL", "MSFT", "GOOGL"],  # Small set for testing
        scan_missing_only=True,  # Use debug mode to avoid full processing
        save_csv=False,
        notify=False,
        run_parallel=False,
        capital_long=100000.0,
        capital_short=50000.0,
        csv_name_mode="plain",
    )

    try:
        # This should display the header messages we added
        result = execute_today_signals(test_config)
        print(f"Test completed successfully. Debug mode: {result.debug_mode}")
        print(f"Log lines count: {len(result.log_lines)}")

        # Print the first few log lines to verify header messages
        print("\nFirst 10 log lines:")
        for i, line in enumerate(result.log_lines[:10]):
            print(f"  {i+1}: {line}")

        return True
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_header_message()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
