import pandas as pd
import pytest
from strategies.system7_strategy import System7Strategy


@pytest.fixture
def dummy_data():
    dates = pd.date_range("2024-01-01", periods=70, freq="B")
    df = pd.DataFrame(
        {
            "Open": [100] * 70,
            "High": [101] * 70,
            "Low": [99] * 70,
            "Close": [100] * 70,
            "Volume": [1_000_000] * 70,
        },
        index=dates,
    )
    return {"SPY": df}


def test_minimal_indicators(dummy_data):
    strategy = System7Strategy()
    processed = strategy.prepare_minimal_for_test(dummy_data)
    assert "ATR50" in processed["SPY"].columns


def test_placeholder_run(dummy_data):
    strategy = System7Strategy()
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    df = pd.DataFrame(
        {
            "Open": [100, 100, 100, 100],
            "High": [100, 100, 110, 100],
            "Low": [100, 100, 100, 100],
            "Close": [100, 100, 100, 100],
            "Volume": [1_000_000] * 4,
            "ATR50": [1, 1, 1, 1],
            "max_70": [150, 150, 150, 150],
        },
        index=dates,
    )
    prepared = {"SPY": df}
    entry_date = dates[1]
    candidates = {entry_date: [{"symbol": "SPY", "entry_date": entry_date, "ATR50": 1}]}
    trades = strategy.run_backtest(prepared, candidates, capital=10_000)
    assert not trades.empty
    assert "pnl" in trades.columns
