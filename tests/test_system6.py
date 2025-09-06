import pandas as pd
import pytest
from strategies.system6_strategy import System6Strategy


@pytest.fixture
def dummy_data():
    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    df = pd.DataFrame(
        {
            "Open": [100] * 50,
            "High": [101] * 50,
            "Low": [99] * 50,
            "Close": [100] * 50,
            "Volume": [1_000_000] * 50,
        },
        index=dates,
    )
    return {"DUMMY": df}


def test_minimal_indicators(dummy_data):
    strategy = System6Strategy()
    processed = strategy.prepare_minimal_for_test(dummy_data)
    assert "ATR10" in processed["DUMMY"].columns


def test_placeholder_run(dummy_data):
    strategy = System6Strategy()
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    df = pd.DataFrame(
        {
            "Open": [100, 100, 100, 100],
            "High": [100, 100, 100, 100],
            "Low": [100, 100, 100, 100],
            "Close": [100, 105, 90, 95],
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
