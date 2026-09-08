"""Regression tests for the 2026-09-09 cache freshness WARN.

Two independent defects were visible in the morning SelfMonitor:

1. ``check_data_advance`` counted Mon-Fri weekdays with ``pd.bdate_range`` and
   therefore counted Labor Day (2026-09-07) as a market session.  SPY 09-04 vs
   NYSE latest 09-08 is one missing NYSE session, not two.
2. ``cache_daily_polygon --auto-latest`` could have a real fetch range, receive
   no usable day (or fail to advance the SPY reference), and still return 0.
   That made daily_pipeline report a green cache step while data stayed stale.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import scripts.cache_daily_polygon as cdp
from scripts.self_monitor_check import check_data_advance


def _settings(tmp_path: Path) -> SimpleNamespace:
    full_dir = tmp_path / "data_cache" / "full_backup"
    full_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        cache=SimpleNamespace(full_dir=str(full_dir), round_decimals=None)
    )


def _write_spy(settings: SimpleNamespace, last_date: str) -> None:
    full_dir = Path(settings.cache.full_dir)
    full_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Date": [last_date], "Close": [100.0]}).to_csv(
        full_dir / "SPY.csv", index=False
    )


def _patch_auto_range(monkeypatch, settings: SimpleNamespace) -> None:
    monkeypatch.setattr(
        "config.settings.get_settings", lambda create_dirs=True: settings
    )
    monkeypatch.setattr(
        cdp,
        "resolve_auto_range",
        lambda _settings, **kwargs: (date(2026, 9, 8), date(2026, 9, 8)),
    )


def test_data_fresh_labor_day_counts_real_nyse_sessions(
    tmp_path: Path, monkeypatch
) -> None:
    """09-04 -> 09-08 crosses Labor Day, so the true NYSE lag is exactly one."""
    dc = tmp_path / "data_cache"
    full_dir = dc / "full_backup"
    full_dir.mkdir(parents=True)
    pd.DataFrame({"Date": ["2026-09-04"], "Close": [100.0]}).to_csv(
        full_dir / "SPY.csv", index=False
    )
    monkeypatch.setattr(
        "common.utils_spy.get_latest_nyse_trading_day",
        lambda ts=None: pd.Timestamp("2026-09-08"),
    )

    result = check_data_advance(dc)

    assert result.status == "ok", result.detail
    assert result.data["latest_nyse"] == "2026-09-08"
    assert result.data["lag_business_days"] == 1
    assert "lag 1 営業日" in result.detail


def test_auto_latest_range_but_zero_fetched_days_is_nonzero(
    tmp_path: Path, monkeypatch
) -> None:
    """A real auto range with an empty Polygon response must not be green."""
    settings = _settings(tmp_path)
    _write_spy(settings, "2026-09-04")
    _patch_auto_range(monkeypatch, settings)
    monkeypatch.setattr(
        cdp,
        "run_backfill",
        lambda *args, **kwargs: {"days": 0, "symbols": 0, "written": 0, "failed": 0},
    )

    rc = cdp.main(["--auto-latest", "--sleep", "0"])

    assert rc == 2


def test_auto_latest_fetch_without_spy_progress_is_nonzero(
    tmp_path: Path, monkeypatch
) -> None:
    """Even a non-empty fetch is not success when the freshness reference did not move."""
    settings = _settings(tmp_path)
    _write_spy(settings, "2026-09-04")
    _patch_auto_range(monkeypatch, settings)
    monkeypatch.setattr(
        cdp,
        "run_backfill",
        lambda *args, **kwargs: {
            "days": 1,
            "symbols": 5000,
            "written": 5000,
            "failed": 0,
        },
    )

    rc = cdp.main(["--auto-latest", "--sleep", "0"])

    assert rc == 2


def test_auto_latest_fetch_with_spy_progress_stays_zero(
    tmp_path: Path, monkeypatch
) -> None:
    """The normal path remains green after the reference cache actually advances."""
    settings = _settings(tmp_path)
    _write_spy(settings, "2026-09-04")
    _patch_auto_range(monkeypatch, settings)

    def fake_backfill(*args, **kwargs):
        _write_spy(settings, "2026-09-08")
        return {"days": 1, "symbols": 5000, "written": 5000, "failed": 0}

    monkeypatch.setattr(cdp, "run_backfill", fake_backfill)

    rc = cdp.main(["--auto-latest", "--sleep", "0"])

    assert rc == 0
