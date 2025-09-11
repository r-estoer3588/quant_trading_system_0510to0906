from __future__ import annotations

import pandas as pd

from common.profit_protection import evaluate_positions


class DummyPos:
    def __init__(
        self,
        symbol: str,
        *,
        side: str = "long",
        qty: int = 1,
        current_price: float = 100.0,
        plpc: float = 0.0,
        entry_date: pd.Timestamp | None = None,
    ) -> None:
        self.symbol = symbol
        self.side = side
        self.qty = qty
        self.current_price = current_price
        self.unrealized_plpc = plpc
        if entry_date is not None:
            self.entry_date = entry_date


def test_evaluate_positions(monkeypatch):
    def fake_load_price(symbol: str, cache_profile: str = "rolling") -> pd.DataFrame:
        idx = pd.date_range("2023-01-01", periods=70)
        return pd.DataFrame({"High": range(1, 71)}, index=idx)

    monkeypatch.setattr("common.profit_protection.load_price", fake_load_price)

    now = pd.Timestamp.utcnow().normalize()
    positions = [
        DummyPos("SPY", side="short"),
        DummyPos("AAA", side="long", plpc=0.05),
        DummyPos("BBB", side="short", plpc=0.04),
        DummyPos("CCC", side="long", entry_date=now - pd.Timedelta(days=3)),
        DummyPos("DDD", side="long", entry_date=now - pd.Timedelta(days=6)),
        DummyPos("EEE", side="short", plpc=0.05),
        DummyPos("FFF", side="short", entry_date=now - pd.Timedelta(days=3)),
        DummyPos("GGG", side="short", entry_date=now - pd.Timedelta(days=2)),
    ]

    df = evaluate_positions(positions)
    judge = dict(zip(df["symbol"], df["judgement"], strict=True))

    assert judge["SPY"].startswith("70日高値更新")
    assert judge["AAA"] == "4%利益→翌日大引けで手仕舞い"
    assert judge["BBB"] == "4%利益→翌日大引けで手仕舞い"
    assert judge["CCC"] == "3日経過→翌日大引けで手仕舞い"
    assert judge["DDD"] == "6日経過→翌日寄りで手仕舞い"
    assert judge["EEE"] == "5%利益→翌日大引けで手仕舞い"
    assert judge["FFF"] == "3日経過→大引けで手仕舞い"
    assert judge["GGG"] == "2日経過→大引けで手仕舞い"


def test_evaluate_positions_load_failure(monkeypatch):
    """Data load failure should result in judgement failure."""

    def raise_load_price(symbol: str, cache_profile: str = "rolling") -> pd.DataFrame:
        raise RuntimeError("boom")

    monkeypatch.setattr("common.profit_protection.load_price", raise_load_price)

    positions = [DummyPos("SPY", side="short")]

    df = evaluate_positions(positions)
    judge = dict(zip(df["symbol"], df["judgement"], strict=True))
    assert judge["SPY"] == "判定失敗"
