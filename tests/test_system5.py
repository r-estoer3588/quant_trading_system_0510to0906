import pandas as pd
import pytest

from strategies.constants import FALLBACK_EXIT_DAYS_DEFAULT
from strategies.system5_strategy import System5Strategy


@pytest.fixture
def dummy_data():
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    df = pd.DataFrame(
        {
            "Open": [100] * 100,
            "High": [101] * 100,
            "Low": [99] * 100,
            "Close": [100] * 100,
            "Volume": [1_000_000] * 100,
        },
        index=dates,
    )
    return {"DUMMY": df}


def test_minimal_indicators(dummy_data):
    strategy = System5Strategy()
    processed = strategy.prepare_minimal_for_test(dummy_data)
    assert "SMA100" in processed["DUMMY"].columns


def test_placeholder_run(dummy_data):
    strategy = System5Strategy()
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    df = pd.DataFrame(
        {
            "Open": [100, 100, 100, 99],
            "High": [100, 100, 100, 99],
            "Low": [100, 90, 90, 99],
            "Close": [100, 97, 99, 99],
            "Volume": [1_000_000] * 4,
            "ATR10": [1, 1, 1, 1],
        },
        index=dates,
    )
    prepared = {"DUMMY": df}
    entry_date = dates[1]
    candidates = {entry_date: [{"symbol": "DUMMY", "entry_date": entry_date}]}
    trades = strategy.run_backtest(prepared, candidates, capital=10_000)
    assert not trades.empty
    assert "pnl" in trades.columns


def test_entry_rule_limit_buy():
    strategy = System5Strategy()
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    df = pd.DataFrame(
        {
            "Open": [100, 100],
            "High": [101, 101],
            "Low": [99, 99],
            "Close": [100, 100],
            "ATR10": [1, 1],
        },
        index=dates,
    )
    candidate = {"symbol": "DUMMY", "entry_date": dates[1]}
    entry = strategy.compute_entry(df, candidate, current_capital=10_000)
    assert entry == (97.0, pytest.approx(94.0))


def test_system5_profit_target_exits_next_open():
    strategy = System5Strategy()
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    df = pd.DataFrame(
        {
            "Open": [100, 100, 110, 120, 120],
            "High": [100, 101, 100, 121, 121],
            "Low": [99, 99, 95, 119, 119],
            "Close": [100, 100, 99, 120, 120],
            "ATR10": [1, 1, 1, 1, 1],
        },
        index=dates,
    )
    candidate = {"symbol": "DUMMY", "entry_date": dates[1]}
    entry_price, stop_price = strategy.compute_entry(df, candidate, 10_000)
    entry_idx = df.index.get_loc(dates[1])

    exit_price, exit_date = strategy.compute_exit(
        df, entry_idx, entry_price, stop_price
    )

    assert exit_date == dates[3]
    assert exit_price == pytest.approx(float(df.iloc[3]["Open"]))


def test_system5_stop_exit_uses_stop_price_same_day():
    strategy = System5Strategy()
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    df = pd.DataFrame(
        {
            "Open": [100, 100, 100, 100],
            "High": [100, 101, 98, 100],
            "Low": [99, 99, 90, 100],
            "Close": [100, 100, 95, 100],
            "ATR10": [1, 1, 1, 1],
        },
        index=dates,
    )
    candidate = {"symbol": "DUMMY", "entry_date": dates[1]}
    entry_price, stop_price = strategy.compute_entry(df, candidate, 10_000)
    entry_idx = df.index.get_loc(dates[1])

    exit_price, exit_date = strategy.compute_exit(
        df, entry_idx, entry_price, stop_price
    )

    assert exit_date == dates[2]
    assert exit_price == pytest.approx(stop_price)


def test_system5_fallback_exit_next_open_after_six_days():
    strategy = System5Strategy()
    fallback_days = strategy.config.get(
        "fallback_exit_after_days", FALLBACK_EXIT_DAYS_DEFAULT
    )
    periods = fallback_days + 3  # entry day + fallback window + next day
    dates = pd.date_range("2024-01-01", periods=periods, freq="B")
    highs = [97] * periods
    lows = [95] * periods
    df = pd.DataFrame(
        {
            "Open": [100 + i for i in range(periods)],
            "High": highs,
            "Low": lows,
            "Close": [100] * periods,
            "ATR10": [1] * periods,
        },
        index=dates,
    )
    candidate = {"symbol": "DUMMY", "entry_date": dates[1]}
    entry_price, stop_price = strategy.compute_entry(df, candidate, 10_000)
    entry_idx = df.index.get_loc(dates[1])

    exit_price, exit_date = strategy.compute_exit(
        df, entry_idx, entry_price, stop_price
    )

    expected_idx = entry_idx + fallback_days + 1
    assert exit_date == dates[expected_idx]
    assert exit_price == pytest.approx(float(df.iloc[expected_idx]["Open"]))
