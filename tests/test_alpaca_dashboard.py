from types import SimpleNamespace

from app_alpaca_dashboard import _positions_to_df
import pandas as pd


def test_positions_to_df():
    positions = [
        SimpleNamespace(
            symbol="AAPL", qty="10", avg_entry_price="100", current_price="110", unrealized_pl="100"
        ),
        SimpleNamespace(
            symbol="MSFT", qty="5", avg_entry_price="200", current_price="190", unrealized_pl="-50"
        ),
    ]
    df = _positions_to_df(positions)
    assert isinstance(df, pd.DataFrame)
    assert list(df["symbol"]) == ["AAPL", "MSFT"]
    assert list(df["qty"]) == ["10", "5"]
