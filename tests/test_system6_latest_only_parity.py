from __future__ import annotations

import pandas as pd

from core.system6 import generate_candidates_system6


def _make_prepared(symbol: str, dates: pd.DatetimeIndex, ret_vals: list[float]) -> pd.DataFrame:
    """Construct a minimal 'prepared' DataFrame satisfying System6 full-path requirements.

    All symbols share identical non-ranking feature values except for return_6d.
    Only the latest row has setup=True so full path yields a single entry_date matching
    fast path's latest date.
    """
    assert len(dates) == len(ret_vals)
    n = len(dates)
    data = {
        "Open": [30.0] * n,
        "High": [31.0] * n,
        "Low": [29.5] * n,
        "Close": [30.5 + i * 0.1 for i in range(n)],
        "Volume": [1_000_000] * n,
        "atr10": [1.0] * n,
        "dollarvolume50": [20_000_000] * n,
        "return_6d": ret_vals,
        "UpTwoDays": [0] * n,
        "filter": [1] * n,  # pass filter
        "setup": [False] * (n - 1) + [True],  # only latest day triggers
        "hv50": [0.20] * n,
    }
    return pd.DataFrame(data, index=dates)


def test_system6_latest_only_parity_latest_day():
    dates = pd.date_range("2024-06-10", periods=5, freq="B")
    prepared = {
        "AAA": _make_prepared("AAA", dates, [0.05, 0.07, 0.10, 0.12, 0.25]),
        "BBB": _make_prepared("BBB", dates, [0.04, 0.06, 0.09, 0.11, 0.20]),
        "CCC": _make_prepared("CCC", dates, [0.03, 0.05, 0.08, 0.10, 0.18]),
        "DDD": _make_prepared("DDD", dates, [0.02, 0.04, 0.06, 0.07, 0.05]),
        "EEE": _make_prepared("EEE", dates, [0.01, 0.02, 0.03, 0.04, 0.01]),
    }

    top_n = 3
    fast_by_date, fast_df = generate_candidates_system6(prepared, top_n=top_n, latest_only=True)
    assert fast_df is not None
    # fast_path の最新日を抽出
    fast_latest = max(fast_by_date.keys())

    # full path では全期間 setup=True なので複数日が生成され得る。最新日の集合を比較。
    full_by_date, full_df = generate_candidates_system6(prepared, top_n=top_n, latest_only=False)
    assert full_df is None  # full path は None 仕様
    assert fast_latest in full_by_date

    # fast path symbols ordered (we added rank but we rely on return_6d ordering)
    fast_syms_ordered = sorted(
        fast_by_date[fast_latest].items(), key=lambda kv: kv[1]["return_6d"], reverse=True
    )
    fast_syms = [sym for sym, _ in fast_syms_ordered]

    # full path latest date top_n sorted by return_6d
    full_latest_syms_payload = full_by_date[fast_latest]
    full_sorted = sorted(
        full_latest_syms_payload.items(), key=lambda kv: kv[1]["return_6d"], reverse=True
    )[:top_n]
    full_syms_top = [sym for sym, _ in full_sorted]

    expected = ["AAA", "BBB", "CCC"]  # 0.25 > 0.20 > 0.18
    assert fast_syms[:top_n] == expected
    assert full_syms_top == expected

    # 指標一致確認 (return_6d / entry_price)
    for sym in expected:
        f_payload = fast_by_date[fast_latest][sym]
        full_payload = full_by_date[fast_latest][sym]
        assert float(f_payload["return_6d"]) == float(full_payload["return_6d"])  # type: ignore[index]
        assert float(f_payload["entry_price"]) == float(full_payload["entry_price"])  # type: ignore[index]
