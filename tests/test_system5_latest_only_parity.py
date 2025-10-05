from __future__ import annotations

import pandas as pd

from core.system5 import generate_candidates_system5


def _make_prepared(symbol: str, dates: pd.DatetimeIndex, adx_vals: list[float]) -> pd.DataFrame:
    assert len(dates) == len(adx_vals)
    return pd.DataFrame(
        {
            "Close": [120.0 + i for i in range(len(dates))],
            "adx7": adx_vals,
            "setup": [True] * len(dates),
            "atr_pct": [2.0] * len(dates),
        },
        index=dates,
    )


def test_system5_latest_only_parity_latest_day():
    dates = pd.date_range("2024-05-13", periods=4, freq="B")
    latest = dates[-1]
    prepared = {
        # 最終日の adx7: 60, 55, 45, 36 (>=35.01 想定で 35 しきい値超過), 34 (除外)
        "AAA": _make_prepared("AAA", dates, [20, 30, 40, 60]),
        "BBB": _make_prepared("BBB", dates, [19, 28, 38, 55]),
        "CCC": _make_prepared("CCC", dates, [18, 27, 37, 45]),
        "DDD": _make_prepared("DDD", dates, [17, 26, 36, 36]),
        "EEE": _make_prepared("EEE", dates, [16, 25, 34, 34]),  # 閾値以下で除外
    }

    top_n = 3
    fast_by_date, fast_df = generate_candidates_system5(prepared, top_n=top_n, latest_only=True)
    assert fast_df is not None
    full_by_date, full_df = generate_candidates_system5(prepared, top_n=top_n, latest_only=False)
    assert full_df is not None and latest in full_by_date

    fast_syms = list(fast_df[fast_df["date"] == latest]["symbol"])  # adx7 desc
    full_syms = list(full_by_date[latest].keys())
    expected = ["AAA", "BBB", "CCC"]  # 60 > 55 > 45
    assert fast_syms == expected
    assert full_syms == expected

    fast_map = {r.symbol: r for r in fast_df.itertuples() if r.date == latest}
    full_map = full_by_date[latest]
    for sym in expected:
        assert float(fast_map[sym].adx7) == float(full_map[sym]["adx7"])  # type: ignore[index]
        assert float(fast_map[sym].close) == float(full_map[sym]["close"])  # type: ignore[index]
