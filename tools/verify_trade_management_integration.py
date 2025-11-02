#!/usr/bin/env python3
"""Verify that trade management is actually integrated into allocation."""
# ruff: noqa: E402

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.final_allocation import (  # noqa: E402
    finalize_allocation,
    load_symbol_system_map,
)

# Import strategies so finalize_allocation has access to per-system sizing funcs
from strategies.system1_strategy import System1Strategy  # noqa: E402
from strategies.system2_strategy import System2Strategy  # noqa: E402
from strategies.system3_strategy import System3Strategy  # noqa: E402
from strategies.system4_strategy import System4Strategy  # noqa: E402
from strategies.system5_strategy import System5Strategy  # noqa: E402
from strategies.system6_strategy import System6Strategy  # noqa: E402
from strategies.system7_strategy import System7Strategy  # noqa: E402


def create_sample_market_data(symbol: str, days: int = 100) -> pd.DataFrame:
    """Create sample market data with required indicators."""
    # Use business days and ensure last day is today's date
    end_date = pd.Timestamp.now().normalize()
    dates = pd.bdate_range(end=end_date, periods=days)

    # Base price data
    close_prices = 100 + np.random.randn(days).cumsum()
    close_prices = np.maximum(close_prices, 1)  # Ensure positive

    data = {
        "date": dates,
        "Open": close_prices * (1 + np.random.randn(days) * 0.01),
        "High": close_prices * (1 + abs(np.random.randn(days)) * 0.02),
        "Low": close_prices * (1 - abs(np.random.randn(days)) * 0.02),
        "Close": close_prices,
        "Volume": np.random.randint(1000000, 10000000, days),
        "atr10": close_prices * 0.02,
        "atr20": close_prices * 0.025,
        "atr40": close_prices * 0.03,
    }

    df = pd.DataFrame(data)
    # Ensure date is proper DatetimeIndex
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date", drop=False)
    return df


def test_trade_management_integration():
    """Test that trade management information is added to allocation results."""
    print("=== Testing Trade Management Integration ===\n")

    # Create sample candidate data
    candidates_s1 = pd.DataFrame(
        [
            {"symbol": "AAPL", "score": 15.5, "shares": 100},
            {"symbol": "MSFT", "score": 12.3, "shares": 50},
        ]
    )

    candidates_s2 = pd.DataFrame(
        [
            {"symbol": "TSLA", "score": 8.7, "shares": 30},
        ]
    )

    per_system = {
        "system1": candidates_s1,
        "system2": candidates_s2,
    }

    # Create market data for each symbol
    market_data_dict = {
        "AAPL": create_sample_market_data("AAPL"),
        "MSFT": create_sample_market_data("MSFT"),
        "TSLA": create_sample_market_data("TSLA"),
    }

    # Set signal date to latest date in market data
    signal_date = market_data_dict["AAPL"].index[-1]

    print(f"Signal date: {signal_date}")
    print(f"Market data available for: {list(market_data_dict.keys())}\n")

    # Test 1: Without trade management
    print("1. Testing WITHOUT trade management:")
    # Create strategies mapping and symbol_system_map to mirror production callers
    strategy_objs = [
        System1Strategy(),
        System2Strategy(),
        System3Strategy(),
        System4Strategy(),
        System5Strategy(),
        System6Strategy(),
        System7Strategy(),
    ]
    strategies = {getattr(s, "SYSTEM_NAME", "").lower(): s for s in strategy_objs}
    try:
        symbol_system_map = load_symbol_system_map()
    except Exception:
        symbol_system_map = {}

    final_df_without, summary_without = finalize_allocation(
        per_system,
        long_allocations={"system1": 1.0},
        short_allocations={"system2": 1.0},
        slots_long=2,
        slots_short=1,
        include_trade_management=False,
        strategies=strategies,
        symbol_system_map=symbol_system_map,
    )

    print(f"   Columns WITHOUT trade management: {list(final_df_without.columns)}")
    print(f"   Rows: {len(final_df_without)}\n")

    # Test 2: With trade management (default behavior)
    print("2. Testing WITH trade management:")
    try:
        final_df_with, summary_with = finalize_allocation(
            per_system,
            long_allocations={"system1": 1.0},
            short_allocations={"system2": 1.0},
            slots_long=2,
            slots_short=1,
            market_data_dict=market_data_dict,
            signal_date=signal_date,
            include_trade_management=True,
            strategies=strategies,
            symbol_system_map=symbol_system_map,
        )

        print(f"   Columns WITH trade management: {list(final_df_with.columns)}")
        print(f"   Rows: {len(final_df_with)}\n")

        # Check for trade management columns
        trade_mgmt_columns = [
            "entry_type",
            "entry_price_final",
            "stop_price",
            "profit_target_price",
            "use_trailing_stop",
            "trailing_stop_pct",
            "max_holding_days",
            "entry_atr",
            "risk_per_share",
            "total_risk",
        ]

        print("3. Checking for trade management columns:")
        found_columns = []
        missing_columns = []
        for col in trade_mgmt_columns:
            if col in final_df_with.columns:
                found_columns.append(col)
                print(f"   ✅ {col}")
            else:
                missing_columns.append(col)
                print(f"   ❌ {col} (MISSING)")

        print("\n4. Sample data with trade management:")
        if not final_df_with.empty:
            # Display first row with trade management info
            row = final_df_with.iloc[0]
            print(f"\n   Symbol: {row.get('symbol', 'N/A')}")
            print(f"   System: {row.get('system', 'N/A')}")
            print(f"   Entry Type: {row.get('entry_type', 'N/A')}")

            entry_price = row.get("entry_price_final", "N/A")
            if isinstance(entry_price, (int, float)):
                print(f"   Entry Price: ${entry_price:.2f}")
            else:
                print(f"   Entry Price: {entry_price}")

            stop_price = row.get("stop_price", "N/A")
            if isinstance(stop_price, (int, float)):
                print(f"   Stop Price: ${stop_price:.2f}")
            else:
                print(f"   Stop Price: {stop_price}")

            profit_target = row.get("profit_target_price")
            if pd.notna(profit_target) and isinstance(profit_target, (int, float)):
                print(f"   Profit Target: ${profit_target:.2f}")

            print(f"   Trailing Stop: {row.get('use_trailing_stop', False)}")
            if row.get("use_trailing_stop"):
                trailing_pct = row.get("trailing_stop_pct", 0)
                if isinstance(trailing_pct, (int, float)):
                    print(f"   Trailing %: {trailing_pct * 100:.1f}%")

            print(f"   Max Hold Days: {row.get('max_holding_days', 'N/A')}")

            risk_per_share = row.get("risk_per_share", "N/A")
            if isinstance(risk_per_share, (int, float)):
                print(f"   Risk/Share: ${risk_per_share:.2f}")
            else:
                print(f"   Risk/Share: {risk_per_share}")

            total_risk = row.get("total_risk", "N/A")
            if isinstance(total_risk, (int, float)):
                print(f"   Total Risk: ${total_risk:.2f}")
            else:
                print(f"   Total Risk: {total_risk}")

        print("\n=== Test Summary ===")
        if found_columns:
            print("✅ Trade management IS integrated!")
            print(
                "   Found "
                + str(len(found_columns))
                + "/"
                + str(len(trade_mgmt_columns))
                + " expected columns"
            )
            if missing_columns:
                print(f"   Missing: {missing_columns}")
            return True
        else:
            print("❌ Trade management NOT integrated!")
            print("   No trade management columns found")
            return False

    except Exception as e:
        print(f"❌ Error during integration test: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_trade_management_integration()
    if success:
        print("\n🎉 Trade Management Integration: VERIFIED")
        sys.exit(0)
    else:
        print("\n❌ Trade Management Integration: NOT WORKING")
        sys.exit(1)
