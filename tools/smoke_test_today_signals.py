# tools/smoke_test_today_signals.py
"""簡易スモークテスト
- system1 の Today Signals を呼び、ログ出力と zero_reason を確認する
"""
import traceback
import os
import sys

# Ensure repo root is on sys.path for local imports when executing this script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# local imports (after sys.path adjustment)
from strategies.system1_strategy import System1Strategy
from common.today_signals import get_today_signals_for_strategy
from common.utils_spy import get_spy_with_indicators


def main():
    # ログコールバック
    def log_cb(msg):
        print(f"[LOG] {msg}")

    try:
        strat = System1Strategy()
        # Try to obtain SPY from cache/helpers and pass as market_df
        spy_df = None
        try:
            spy_df = get_spy_with_indicators()
        except Exception:
            spy_df = None

        sample_data = {}
        print("Starting smoke test: system1")
        df = get_today_signals_for_strategy(
            strat,
            sample_data,
            market_df=spy_df,
            log_callback=log_cb,
        )
        print("Result DataFrame:")
        print(df)

    except Exception:
        print("Exception during smoke test:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
