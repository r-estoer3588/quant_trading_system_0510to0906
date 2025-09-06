import pandas as pd
import pytest
from strategies.system1_strategy import System1Strategy


@pytest.fixture
def dummy_data():
    dates = pd.date_range("2024-01-01", periods=250, freq="B")
    df = pd.DataFrame(
        {
            "Open": [100 + i * 0.1 for i in range(250)],
            "High": [101 + i * 0.1 for i in range(250)],
            "Low": [99 + i * 0.1 for i in range(250)],
            "Close": [100 + i * 0.1 for i in range(250)],
            "Volume": [1_000_000] * 250,
        },
        index=dates,
    )
    return {"DUMMY": df}


def test_prepare_data(dummy_data):
    strategy = System1Strategy()
    processed = strategy.prepare_data(dummy_data)
    assert isinstance(processed, dict)
    assert "DUMMY" in processed
    assert "SMA25" in processed["DUMMY"].columns


def test_placeholder_run(dummy_data):
    strategy = System1Strategy()
    # 最小限のデータセットでバックテストを実行
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "Open": [100, 110, 110],
            "High": [101, 112, 112],
            "Low": [99, 108, 100],
            "Close": [100, 111, 111],
            "Volume": [1_000_000] * 3,
            "ATR20": [1, 1, 1],
        },
        index=dates,
    )
    prepared = {"DUMMY": df}
    entry_date = dates[1]
    candidates = {entry_date: [{"symbol": "DUMMY", "entry_date": entry_date}]}
    trades = strategy.run_backtest(prepared, candidates, capital=10_000)
    assert not trades.empty
    assert "pnl" in trades.columns
