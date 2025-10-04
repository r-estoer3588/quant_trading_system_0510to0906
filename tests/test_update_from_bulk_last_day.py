from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from common.cache_manager import CacheManager
from scripts.update_from_bulk_last_day import (
    CacheUpdateInterrupted,
    append_to_cache,
    run_bulk_update,
)


class DummyRolling(SimpleNamespace):
    base_lookback_days = 3
    buffer_days = 1
    prune_chunk_days = 2
    meta_file = "_meta.json"


def _build_cache_manager(tmp_path) -> CacheManager:
    cache = SimpleNamespace(
        full_dir=tmp_path / "full",
        rolling_dir=tmp_path / "rolling",
        rolling=DummyRolling(),
        file_format="csv",
    )
    settings = SimpleNamespace(cache=cache, DATA_CACHE_DIR=tmp_path)
    return CacheManager(settings)


def _base_rows() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
            "adjusted_close": [10.4, 11.4, 12.4],
            "volume": [1000, 1100, 1200],
            "sma25": [1.0, 1.1, 1.2],
        }
    )


def _format_dates(df: pd.DataFrame) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.to_datetime(df["date"])]


def test_append_to_cache_integrates_with_cache_manager(tmp_path):
    cm = _build_cache_manager(tmp_path)
    cm.upsert_both("AAA", _base_rows())

    bulk = pd.DataFrame(
        {
            "code": ["AAA", "BBB"],
            "date": ["2024-01-04", "2024-01-04"],
            "open": [13.0, 20.0],
            "high": [14.0, 21.0],
            "low": [12.0, 19.0],
            "close": [13.5, 20.5],
            "adjusted_close": [13.4, 20.4],
            "volume": [1300, 2000],
        }
    )

    total, updated = append_to_cache(bulk, cm)
    assert total == 2
    assert updated == 2

    aaa_full = cm.read("AAA", "full")
    assert _format_dates(aaa_full) == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
    ]
    assert "sma25" in aaa_full.columns
    assert aaa_full["sma25"].isna().all()

    aaa_rolling = cm.read("AAA", "rolling")
    assert _format_dates(aaa_rolling) == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
    ]

    bbb_full = cm.read("BBB", "full")
    assert _format_dates(bbb_full) == ["2024-01-04"]
    bbb_rolling = cm.read("BBB", "rolling")
    assert _format_dates(bbb_rolling) == ["2024-01-04"]

    aaa_base_path = tmp_path / "base" / "AAA.csv"
    bbb_base_path = tmp_path / "base" / "BBB.csv"
    assert aaa_base_path.exists()
    assert bbb_base_path.exists()
    aaa_base = pd.read_csv(aaa_base_path)
    assert "Date" in aaa_base.columns
    close_col = "Close" if "Close" in aaa_base.columns else "close"
    assert pytest.approx(float(aaa_base.iloc[-1][close_col])) == 13.4

    # Running again with the same payload should not create duplicates
    append_to_cache(bulk, cm)
    aaa_full_again = cm.read("AAA", "full")
    assert len(aaa_full_again) == 4

    # Next trading day should extend full cache but keep rolling window length
    bulk_next = pd.DataFrame(
        {
            "code": ["AAA"],
            "date": ["2024-01-05"],
            "open": [14.0],
            "high": [15.0],
            "low": [13.0],
            "close": [14.5],
            "adjusted_close": [14.4],
            "volume": [1400],
        }
    )
    append_to_cache(bulk_next, cm)

    aaa_full_final = cm.read("AAA", "full")
    assert _format_dates(aaa_full_final) == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
    ]

    aaa_rolling_final = cm.read("AAA", "rolling")
    assert _format_dates(aaa_rolling_final) == [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
    ]


def test_append_to_cache_reports_progress(tmp_path):
    cm = _build_cache_manager(tmp_path)
    bulk = pd.DataFrame(
        {
            "code": ["AAA", "BBB"],
            "date": ["2024-01-04", "2024-01-04"],
            "open": [13.0, 20.0],
            "high": [14.0, 21.0],
            "low": [12.0, 19.0],
            "close": [13.5, 20.5],
            "adjusted_close": [13.4, 20.4],
            "volume": [1300, 2000],
        }
    )

    calls: list[tuple[int, int, int]] = []

    def _progress(processed: int, total: int, updated: int) -> None:
        calls.append((processed, total, updated))

    total, updated = append_to_cache(
        bulk,
        cm,
        progress_callback=_progress,
        progress_step=1,
    )

    assert total == 2
    assert updated == 2
    assert calls
    assert calls[0] == (0, 2, 0)
    assert calls[1] == (1, 2, 1)
    assert calls[-1] == (2, 2, 2)


def test_append_to_cache_ignores_rows_without_prices(tmp_path):
    cm = _build_cache_manager(tmp_path)
    payload = pd.DataFrame(
        {
            "code": ["AAA", "AAA"],
            "date": ["2024-01-04", "2024-01-05"],
            "open": [np.nan, np.nan],
            "high": [np.nan, np.nan],
            "low": [np.nan, np.nan],
            "close": [np.nan, np.nan],
            "adjusted_close": [np.nan, np.nan],
            "volume": [np.nan, np.nan],
        }
    )

    total, updated = append_to_cache(payload, cm)

    assert total == 0
    assert updated == 0
    assert not (tmp_path / "full" / "AAA.csv").exists()


def test_append_to_cache_skips_symbol_with_nan_prices(tmp_path):
    cm = _build_cache_manager(tmp_path)
    payload = pd.DataFrame(
        {
            "code": ["BAD", "GOOD"],
            "date": ["2024-01-04", "2024-01-04"],
            "open": [np.nan, 10.0],
            "high": [np.nan, 11.0],
            "low": [np.nan, 9.0],
            "close": [np.nan, 10.5],
            "adjusted_close": [np.nan, 10.4],
            "volume": [np.nan, 1000],
        }
    )

    total, updated = append_to_cache(payload, cm)

    assert total == 1
    assert updated == 1
    assert not (tmp_path / "full" / "BAD.csv").exists()
    assert (tmp_path / "full" / "GOOD.csv").exists()


def test_append_to_cache_progress_step_auto_large(tmp_path):
    cm = _build_cache_manager(tmp_path)
    symbols = [f"SYM{i:04d}" for i in range(150)]
    payload = pd.DataFrame(
        {
            "code": symbols,
            "date": ["2024-01-04"] * len(symbols),
            "open": np.linspace(10.0, 10.0 + len(symbols) - 1, len(symbols)),
            "high": np.linspace(10.5, 10.5 + len(symbols) - 1, len(symbols)),
            "low": np.linspace(9.5, 9.5 + len(symbols) - 1, len(symbols)),
            "close": np.linspace(10.2, 10.2 + len(symbols) - 1, len(symbols)),
            "adjusted_close": np.linspace(10.1, 10.1 + len(symbols) - 1, len(symbols)),
            "volume": np.linspace(1000, 1000 + len(symbols) - 1, len(symbols)).astype(
                int
            ),
        }
    )

    calls: list[tuple[int, int, int]] = []

    def _progress(processed: int, total: int, updated: int) -> None:
        calls.append((processed, total, updated))

    total, updated = append_to_cache(
        payload,
        cm,
        progress_callback=_progress,
    )

    assert total == len(symbols)
    assert updated == len(symbols)
    processed = [p for p, _, _ in calls if p > 0]
    assert processed
    diffs = [b - a for a, b in zip(processed, processed[1:], strict=False)]
    if diffs:
        assert max(diffs) <= 20


def test_append_to_cache_keyboard_interrupt(tmp_path, monkeypatch):
    cm = _build_cache_manager(tmp_path)
    bulk = pd.DataFrame(
        {
            "code": ["AAA", "BBB"],
            "date": ["2024-01-04", "2024-01-04"],
            "open": [13.0, 20.0],
            "high": [14.0, 21.0],
            "low": [12.0, 19.0],
            "close": [13.5, 20.5],
            "adjusted_close": [13.4, 20.4],
            "volume": [1300, 2000],
        }
    )

    call_count = {"count": 0}

    def _interrupting_add_indicators(*args, **kwargs):
        call_count["count"] += 1
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "scripts.update_from_bulk_last_day.add_indicators",
        _interrupting_add_indicators,
    )

    with pytest.raises(CacheUpdateInterrupted) as excinfo:
        append_to_cache(bulk, cm, progress_step=1)

    err = excinfo.value
    assert err.processed == 1
    assert err.updated == 0
    # 中断までに一度 add_indicators を試行していることを確認
    assert call_count["count"] == 1


def test_run_bulk_update_uses_provided_payload(tmp_path):
    cm = _build_cache_manager(tmp_path)
    cm.upsert_both("AAA", _base_rows())

    payload = pd.DataFrame(
        {
            "code": ["AAA", "BBB"],
            "date": ["2024-01-04", "2024-01-04"],
            "open": [13.0, 21.0],
            "high": [14.0, 22.0],
            "low": [12.0, 20.0],
            "close": [13.5, 21.5],
            "adjusted_close": [13.4, 21.4],
            "volume": [1300, 2100],
        }
    )

    stats = run_bulk_update(
        cm,
        data=payload,
        universe=["AAA", "BBB"],
        fetch_universe=False,
    )

    assert stats.has_payload is True
    assert stats.filtered_rows == 2
    assert stats.processed_symbols == 2
    assert stats.updated_symbols == 2
    assert stats.progress_step_used == 1
    aaa_full = cm.read("AAA", "full")
    assert _format_dates(aaa_full)[-1] == "2024-01-04"
    bbb_base_path = tmp_path / "base" / "BBB.csv"
    assert bbb_base_path.exists()
