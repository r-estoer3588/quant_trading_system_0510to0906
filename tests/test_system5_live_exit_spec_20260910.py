"""Regression tests for the System5 live adapter.

These tests pin *existing* Bensdorp/System5 semantics.  They must fail if live
execution drifts toward an immediate target limit/OCO, a day-6 time exit, or a
moving ATR target/stop.  No strategy parameter is introduced here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from common.alpaca_trading import ExitReasonCode, PositionSnapshot
from common.system5_live_exit import (
    SYSTEM5_TARGET_NEXT_OPEN,
    build_system5_live_exits,
    resolve_system5_context,
)
from common.trade_management import SYSTEM_TRADE_RULES


def _snap(symbol: str = "S5X", *, entry_date: str = "2026-08-10", qty: float = 10):
    return PositionSnapshot(
        symbol=symbol,
        qty=qty,
        side="long",
        avg_entry_price=100.0,
        system="system5",
        entry_date=entry_date,
        market_value=1000.0,
    )


def _write_history(
    root: Path,
    symbol: str = "S5X",
    *,
    target_hit_date: str | None = None,
    pre_entry_atr: float = 2.0,
    later_atr: float = 9.0,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    dates = [
        "2026-08-07",  # pre-entry ATR row
        "2026-08-10",  # entry day (target/stop observation starts after this)
        "2026-08-11",  # post-entry day 1
        "2026-08-12",  # 2
        "2026-08-13",  # 3
        "2026-08-14",  # 4
        "2026-08-17",  # 5
        "2026-08-18",  # 6
        "2026-08-19",  # 7 -- fallback exit is at this open
    ]
    highs = [100.0] * len(dates)
    if target_hit_date is not None:
        highs[dates.index(target_hit_date)] = 102.5
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": [100.0] * len(dates),
            "High": highs,
            "Low": [99.0] * len(dates),
            "Close": [100.0] * len(dates),
            # The large later ATR proves the adapter freezes the pre-entry value.
            "atr10": [pre_entry_atr] + [later_atr] * (len(dates) - 1),
        }
    )
    path = root / f"{symbol}.csv"
    df.to_csv(path, index=False)
    return path


def test_repo_strategy_values_remain_unchanged():
    rules = SYSTEM_TRADE_RULES["system5"]
    assert rules.stop_atr_period == 10
    assert rules.stop_atr_multiplier == pytest.approx(3.0)
    assert rules.profit_target_type == "atr"
    assert rules.profit_target_value == pytest.approx(1.0)
    assert rules.profit_target_atr_period == 10
    assert rules.max_holding_days == 6


def test_day6_is_observation_day_not_time_exit(tmp_path: Path):
    rolling = tmp_path / "rolling"
    _write_history(rolling)
    snap = _snap()

    day6 = build_system5_live_exits([snap], today="2026-08-18", rolling_dir=rolling)
    assert not [x for x in day6 if x.reason == ExitReasonCode.TIME]

    day7 = build_system5_live_exits([snap], today="2026-08-19", rolling_dir=rolling)
    timed = [x for x in day7 if x.reason == ExitReasonCode.TIME]
    assert len(timed) == 1
    assert timed[0].order_type == "market"
    assert timed[0].holding_days == 7
    # Metadata still says the strategy observes six days; live adapter translates
    # that to the following open rather than changing the strategy value to seven.
    assert timed[0].max_holding_days == 6


def test_target_touch_closes_at_following_session_not_same_session(tmp_path: Path):
    rolling = tmp_path / "rolling"
    _write_history(rolling, target_hit_date="2026-08-13")
    snap = _snap()

    same_day = build_system5_live_exits(
        [snap], today="2026-08-13", rolling_dir=rolling
    )
    assert not [x for x in same_day if x.reason == SYSTEM5_TARGET_NEXT_OPEN]

    following_open = build_system5_live_exits(
        [snap], today="2026-08-14", rolling_dir=rolling
    )
    exits = [x for x in following_open if x.reason == SYSTEM5_TARGET_NEXT_OPEN]
    assert len(exits) == 1
    assert exits[0].order_type == "market"
    assert exits[0].side == "sell"


def test_target_and_stop_use_pre_entry_atr_not_latest_atr(tmp_path: Path):
    rolling = tmp_path / "rolling"
    _write_history(
        rolling,
        target_hit_date="2026-08-11",
        pre_entry_atr=2.0,
        later_atr=9.0,
    )
    snap = _snap()

    ctx = resolve_system5_context(snap, today="2026-08-12", rolling_dir=rolling)
    assert ctx.entry_atr10 == pytest.approx(2.0)
    assert ctx.entry_atr_source == "rolling_pre_entry"
    assert ctx.target_price == pytest.approx(102.0)
    assert ctx.target_hit_date == "2026-08-11"

    # Before a target is due, missing native stop is proposed at entry - 3*entryATR.
    stops = build_system5_live_exits(
        [snap], today="2026-08-11", rolling_dir=rolling
    )
    stop = next(x for x in stops if x.reason == ExitReasonCode.PROTECT_STOP)
    assert stop.stop_price == pytest.approx(94.0)
    assert stop.order_type == "stop"
    assert stop.time_in_force == "gtc"


def test_existing_legacy_oco_is_migrated_to_stop_only(tmp_path: Path):
    rolling = tmp_path / "rolling"
    _write_history(rolling)
    snap = _snap()
    old_oco = "protect-system5-S5X-20260810-protect-oco"

    exits = build_system5_live_exits(
        [snap],
        today="2026-08-11",
        rolling_dir=rolling,
        existing_protect_coids={old_oco},
    )
    assert len(exits) == 1
    stop = exits[0]
    assert stop.reason == ExitReasonCode.PROTECT_STOP
    assert stop.order_type == "stop"
    assert stop.stop_price == pytest.approx(94.0)
    assert stop.cancel_client_order_ids == [old_oco]
    assert all(x.order_type not in ("limit", "oco") for x in exits)


def test_existing_stop_stays_put_without_daily_recreation(tmp_path: Path):
    rolling = tmp_path / "rolling"
    _write_history(rolling)
    snap = _snap()
    stop_coid = "protect-system5-S5X-20260810-protect-stop"
    coverage: list[dict] = []

    exits = build_system5_live_exits(
        [snap],
        today="2026-08-11",
        rolling_dir=rolling,
        existing_protect_coids={stop_coid},
        protection_coverage_out=coverage,
    )
    assert exits == []
    assert coverage[0]["resident_order"] is True
    assert coverage[0]["detail"] == "existing_stop"


def test_missing_history_never_manufactures_profit_target_from_latest_atr(tmp_path: Path):
    snap = _snap(symbol="NODATA")
    diagnostics: list[dict] = []

    exits = build_system5_live_exits(
        [snap],
        today="2026-08-12",
        rolling_dir=tmp_path / "missing",
        latest_atr_by_symbol={"NODATA": {10: 3.0}},
        diagnostics_out=diagnostics,
    )
    # Downside safety can fall back to the existing latest-ATR behavior.
    stop = next(x for x in exits if x.reason == ExitReasonCode.PROTECT_STOP)
    assert stop.stop_price == pytest.approx(91.0)
    # But latest ATR is never used to invent an S5 target trigger.
    assert not [x for x in exits if x.reason == SYSTEM5_TARGET_NEXT_OPEN]
    assert diagnostics[0]["target_price"] is None
    assert str(diagnostics[0]["data_error"]).startswith("rolling_missing:")
