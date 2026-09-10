from __future__ import annotations

import pandas as pd

from common.alpaca_trading import ExitReasonCode, PositionSnapshot
from common.system5_live_exit import (
    SYSTEM5_TARGET_NEXT_OPEN,
    build_system5_exit_orders,
    entry_atr10_from_history,
    evaluate_system5_state,
)


def _history(*, hit_day: str | None = None, later_atr: float = 9.0) -> pd.DataFrame:
    dates = pd.bdate_range("2026-08-28", "2026-09-10")
    rows = []
    for d in dates:
        high = 100.0
        if hit_day and str(d.date()) == hit_day:
            high = 106.0
        rows.append(
            {
                "Date": str(d.date()),
                "Open": 100.0,
                "High": high,
                "Low": 95.0,
                # Entry is 2026-09-01.  The pre-entry 2026-08-31 ATR is 5;
                # all later ATRs are intentionally different to catch drift.
                "atr10": 5.0 if str(d.date()) == "2026-08-31" else later_atr,
            }
        )
    return pd.DataFrame(rows)


def _snap() -> PositionSnapshot:
    return PositionSnapshot(
        symbol="TEST",
        qty=10.0,
        side="long",
        avg_entry_price=100.0,
        market_value=1000.0,
        system="system5",
        entry_date="2026-09-01",
    )


def test_entry_atr_is_frozen_from_bar_before_entry():
    df = _history(later_atr=99.0)
    assert entry_atr10_from_history(df, "2026-09-01") == 5.0
    state = evaluate_system5_state(_snap(), today="2026-09-08", history=df)
    assert state.entry_atr10 == 5.0
    assert state.target_price == 105.0


def test_target_touch_does_not_exit_same_session():
    df = _history(hit_day="2026-09-02")
    exits = build_system5_exit_orders(_snap(), today="2026-09-02", history=df)
    assert not [e for e in exits if e.reason == SYSTEM5_TARGET_NEXT_OPEN]


def test_target_touch_exits_market_at_next_session_open():
    df = _history(hit_day="2026-09-02")
    exits = build_system5_exit_orders(_snap(), today="2026-09-03", history=df)
    target = [e for e in exits if e.reason == SYSTEM5_TARGET_NEXT_OPEN]
    assert len(target) == 1
    assert target[0].order_type == "market"
    assert target[0].limit_price == 105.0  # audit trigger only, not a limit order


def test_sixth_session_is_still_observation_day_not_timeout_open():
    # 9/1 entry -> post-entry sessions: 9/2,3,4,8,9,10 because 9/7 is Labor Day.
    df = _history()
    state = evaluate_system5_state(_snap(), today="2026-09-10", history=df)
    assert state.holding_days == 6
    assert state.timeout_exit_due is False
    exits = build_system5_exit_orders(_snap(), today="2026-09-10", history=df)
    assert not [e for e in exits if e.reason == ExitReasonCode.TIME]


def test_timeout_is_next_open_after_six_observation_sessions():
    df = _history()
    exits = build_system5_exit_orders(_snap(), today="2026-09-11", history=df)
    time_rows = [e for e in exits if e.reason == ExitReasonCode.TIME]
    assert len(time_rows) == 1
    assert time_rows[0].holding_days == 7
    assert time_rows[0].order_type == "market"


def test_target_on_sixth_session_wins_at_seventh_open():
    df = _history(hit_day="2026-09-10")
    # Add the seventh session so today can be 9/11; target scan uses completed 9/10.
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "Date": "2026-09-11",
                        "Open": 100.0,
                        "High": 100.0,
                        "Low": 95.0,
                        "atr10": 99.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    exits = build_system5_exit_orders(_snap(), today="2026-09-11", history=df)
    assert len(exits) == 1
    assert exits[0].reason == SYSTEM5_TARGET_NEXT_OPEN


def test_stop_uses_frozen_entry_atr_not_latest_atr(monkeypatch):
    # PROTECT_USE_OCO must not alter System5's canonical target timing.
    monkeypatch.setenv("PROTECT_USE_OCO", "1")
    exits = build_system5_exit_orders(
        _snap(), today="2026-09-03", history=_history(later_atr=40.0)
    )
    stops = [e for e in exits if e.reason == ExitReasonCode.PROTECT_STOP]
    assert len(stops) == 1
    assert stops[0].order_type == "stop"
    assert stops[0].stop_price == 85.0  # 100 - 3*5, never 100 - 3*40
    assert all(e.order_type != "oco" for e in exits)


def test_existing_stop_is_idempotent():
    snap = _snap()
    coid = "protect-system5-TEST-20260901-protect-stop"
    exits = build_system5_exit_orders(
        snap,
        today="2026-09-03",
        history=_history(),
        existing_protect_coids={coid},
    )
    assert exits == []
