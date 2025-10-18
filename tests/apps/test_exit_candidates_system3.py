import numpy as np
import pandas as pd

# 対象関数だけをインポート
from apps.app_today_signals import _evaluate_position_for_exit
from strategies.system3_strategy import System3Strategy


class DummyPos:
    def __init__(self, symbol: str, qty: int, side: str):
        self.symbol = symbol
        self.qty = qty
        self.side = side


def _make_price_df(start_date: str = "2025-10-01", days: int = 12) -> pd.DataFrame:
    idx = pd.date_range(start_date, periods=days, freq="B")
    # 単調増加の価格系列を作成
    close = np.linspace(10, 12, num=days)
    open_ = close * 1.001
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    atr10 = np.full((days,), 0.2)
    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "ATR10": atr10,
        },
        index=idx,
    )
    return df


def test_evaluate_position_for_exit_system3_includes_entry_and_stop(monkeypatch):
    sym = "TEST3"
    # entryは直近営業日の2日前想定
    entry_date = pd.Timestamp("2025-10-09")

    # load_price をモンキーパッチして、full/rollingどちらでも同じDFを返す
    price_df = _make_price_df("2025-10-01", days=12)

    def _fake_load_price(symbol: str, cache_profile: str = "rolling"):
        assert symbol == sym or symbol == "SPY"
        # SPY用にも適当なDFを返す
        if symbol == "SPY":
            return _make_price_df("2025-10-01", days=12)
        return price_df

    import apps.app_today_signals as m

    monkeypatch.setattr(m, "load_price", _fake_load_price)

    pos = DummyPos(symbol=sym, qty=10, side="long")
    entry_map = {sym: entry_date.strftime("%Y-%m-%d")}
    symbol_system_map = {sym: "system3"}
    latest_trading_day = None
    strategy_classes = {"system3": System3Strategy}

    result = _evaluate_position_for_exit(
        pos,
        entry_map,
        symbol_system_map,
        latest_trading_day,
        strategy_classes,
    )
    assert result is not None
    system, pos_side, qty, when, row_base, is_today_exit = result

    assert system == "system3"
    assert qty == 10
    # ここが今回の修正の本質: entry/stopが含まれ、正の値
    assert "entry_price" in row_base and isinstance(row_base["entry_price"], (int, float))
    assert "stop_price" in row_base and isinstance(row_base["stop_price"], (int, float))
    assert row_base["entry_price"] > 0
    assert row_base["stop_price"] > 0 or row_base["stop_price"] < row_base["entry_price"]
