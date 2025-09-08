import pandas as pd
import pytest

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
