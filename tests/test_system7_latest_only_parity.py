from __future__ import annotations

import pandas as pd

from core.system7 import generate_candidates_system7


def make_spy_df(dates: pd.DatetimeIndex, setup_flags: list[bool]) -> pd.DataFrame:
    assert len(dates) == len(setup_flags)
    return pd.DataFrame(
        {
            "Open": [400 + i for i in range(len(dates))],
            "High": [401 + i for i in range(len(dates))],
            "Low": [399 - i * 0.1 for i in range(len(dates))],
            "Close": [400.5 + i for i in range(len(dates))],
            # 必須指標 (lowercase + uppercase依存): atr50, min_50, max_70 -> fast/full で参照
            "atr50": [5.0] * len(dates),
            "min_50": [398.0] * len(dates),
            "max_70": [430.0] * len(dates),
            "setup": setup_flags,
        },
        index=dates,
    )


def test_system7_latest_only_parity_when_setup_today():
    # 5 営業日, 最終日が setup
    dates = pd.date_range("2024-07-01", periods=5, freq="B")
    df = make_spy_df(dates, [False, False, False, False, True])
    prepared = {"SPY": df}

    fast_map, fast_df = generate_candidates_system7(prepared, latest_only=True)
    full_map, full_df = generate_candidates_system7(prepared, latest_only=False)

    # fast は DataFrame を返し、full は None (仕様通り) だが、マッピングで比較可能
    assert fast_df is not None
    assert full_df is None
    assert len(fast_map) == 1
    # full 側も同じ entry_date キーが 1 件
    assert len(full_map) == 1
    fast_entry_date = next(iter(fast_map.keys()))
    full_entry_date = next(iter(full_map.keys()))
    assert fast_entry_date == full_entry_date
    assert "SPY" in fast_map[fast_entry_date]
    assert "SPY" in full_map[full_entry_date]

    f_payload = fast_map[fast_entry_date]["SPY"]
    full_payload = full_map[full_entry_date]["SPY"]
    # entry_price は一致。ATR50 は fast path 側で atr50/ATR50 を拾うが
    # full path 正規化では欠落することがあるため存在する場合のみ比較。
    assert f_payload.get("entry_price") == full_payload.get("entry_price")
    # ATR50 は full path 側で欠落または None の場合があるため存在かつ非 None の時のみ厳密比較
    if ("ATR50" in full_payload) and full_payload.get("ATR50") is not None:
        assert f_payload.get("ATR50") == full_payload.get("ATR50")


def test_system7_latest_only_parity_when_no_setup_today():
    dates = pd.date_range("2024-07-01", periods=5, freq="B")
    # setup が 3 日目のみ。最終日は False。
    df = make_spy_df(dates, [False, False, True, False, False])
    prepared = {"SPY": df}
    fast_map, fast_df = generate_candidates_system7(prepared, latest_only=True)
    full_map, _ = generate_candidates_system7(prepared, latest_only=False)

    # latest_only では最終日 setup False なので 0 件
    assert fast_map == {}
    assert fast_df is None
    # full_map は過去 setup 日 (エントリー日は翌営業日) に基づき >=1 件
    assert len(full_map) >= 1
