from types import SimpleNamespace

from app_alpaca_dashboard import _group_by_system, _positions_to_df
import pandas as pd


def test_positions_to_df():
    positions = [
        SimpleNamespace(
            symbol="AAPL",
            qty="10",
            avg_entry_price="100",
            current_price="110",
            unrealized_pl="100",
        ),
        SimpleNamespace(
            symbol="MSFT",
            qty="5",
            avg_entry_price="200",
            current_price="190",
            unrealized_pl="-50",
        ),
    ]
    df = _positions_to_df(positions)
    assert isinstance(df, pd.DataFrame)
    assert list(df["銘柄"]) == ["AAPL", "MSFT"]
    assert list(df["数量"]) == ["10", "5"]


def test_group_by_system():
    df = pd.DataFrame(
        {
            "銘柄": ["AAPL", "MSFT", "AAPL"],
            "数量": ["10", "5", "2"],
            "現在値": ["110", "190", "100"],
        }
    )
    mapping = {"AAPL": "system1", "MSFT": "system2"}
    grouped = _group_by_system(df, mapping)
    assert set(grouped) == {"system1", "system2"}
    s1 = grouped["system1"]
    # 10*110 + 2*100 = 1300
    assert s1["評価額"].sum() == 1300.0
