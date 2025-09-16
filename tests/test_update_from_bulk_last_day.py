from types import SimpleNamespace

import pandas as pd

from common.cache_manager import CacheManager
from scripts.update_from_bulk_last_day import append_to_cache


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
    settings = SimpleNamespace(cache=cache)
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
    assert aaa_full.loc[0, "sma25"] == 1.0
    assert pd.isna(aaa_full.iloc[-1]["sma25"])

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
