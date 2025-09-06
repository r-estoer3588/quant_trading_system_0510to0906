import pandas as pd
import pytest
from strategies.system2_strategy import System2Strategy


@pytest.fixture
def dummy_data():
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    df = pd.DataFrame(
        {
            "Open": [100] * 100,
            "High": [101] * 100,
            "Low": [99] * 100,
            "Close": [100] * 100,
            "Volume": [2_000_000] * 100,
        },
        index=dates,
    )
    return {"DUMMY": df}


def test_minimal_indicators(dummy_data):
    strategy = System2Strategy()
    processed = strategy.prepare_minimal_for_test(dummy_data)
    assert isinstance(processed, dict)
    assert "DUMMY" in processed
    assert "RSI3" in processed["DUMMY"].columns


def test_placeholder_run(dummy_data):
    strategy = System2Strategy()
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "Open": [100, 110, 110],
            "High": [101, 112, 120],
            "Low": [99, 108, 108],
            "Close": [100, 109, 109],
            "Volume": [2_000_000] * 3,
            "ATR10": [1, 1, 1],
        },
        index=dates,
    )
    prepared = {"DUMMY": df}
    entry_date = dates[1]
    candidates = {entry_date: [{"symbol": "DUMMY", "entry_date": entry_date}]}
    trades = strategy.run_backtest(prepared, candidates, capital=10_000)
    assert not trades.empty
    assert "pnl" in trades.columns
