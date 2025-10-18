from __future__ import annotations

import pandas as pd

from core.system3 import generate_candidates_system3


def _make_prepared(symbol: str, dates: pd.DatetimeIndex, drop_vals: list[float]) -> pd.DataFrame:
    assert len(dates) == len(drop_vals)
    return pd.DataFrame(
        {
            "Close": [70.0 + i for i in range(len(dates))],
            "drop3d": drop_vals,
            "setup": [True] * len(dates),
            "atr_ratio": [0.8] * len(dates),
        },
        index=dates,
    )


def test_system3_latest_only_parity_latest_day():
    dates = pd.date_range("2024-04-08", periods=4, freq="B")
    latest = dates[-1]
    prepared = {
        # 最終日の drop3d: 0.50, 0.40, 0.30, 0.10 (<0.125 は除外)
        "AAA": _make_prepared("AAA", dates, [0.20, 0.25, 0.30, 0.50]),
        "BBB": _make_prepared("BBB", dates, [0.15, 0.18, 0.28, 0.40]),
        "CCC": _make_prepared("CCC", dates, [0.14, 0.16, 0.24, 0.30]),
        "DDD": _make_prepared("DDD", dates, [0.13, 0.14, 0.15, 0.10]),  # 閾値未満で除外
    }

    top_n = 3
    fast_by_date, fast_df = generate_candidates_system3(prepared, top_n=top_n, latest_only=True)
    assert fast_df is not None
    full_by_date, full_df = generate_candidates_system3(prepared, top_n=top_n, latest_only=False)
    assert full_df is not None and latest in full_by_date

    fast_syms = list(fast_df[fast_df["date"] == latest]["symbol"])  # drop3d desc
    full_syms = list(full_by_date[latest].keys())
    expected = ["AAA", "BBB", "CCC"]  # 0.50 > 0.40 > 0.30
    assert fast_syms == expected
    assert full_syms == expected

    fast_map = {r.symbol: r for r in fast_df.itertuples() if r.date == latest}
    full_map = full_by_date[latest]
    for sym in expected:
        assert float(fast_map[sym].drop3d) == float(full_map[sym]["drop3d"])  # type: ignore[index]
        assert float(fast_map[sym].close) == float(full_map[sym]["close"])  # type: ignore[index]
