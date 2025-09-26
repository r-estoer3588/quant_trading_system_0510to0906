from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from common.cache_manager import CacheManager
from common.symbols_manifest import save_symbol_manifest
from scripts.build_rolling_with_indicators import extract_rolling_from_full


class DummyRolling(SimpleNamespace):
    base_lookback_days = 300
    buffer_days = 30
    prune_chunk_days = 30
    meta_file = "_meta.json"
    max_symbols = None


def _build_cache_manager(tmp_path) -> CacheManager:
    rolling = DummyRolling()

    cache = SimpleNamespace(
        full_dir=tmp_path / "full",
        rolling_dir=tmp_path / "rolling",
        rolling=rolling,
        file_format="csv",
    )
    settings = SimpleNamespace(cache=cache)
    return CacheManager(settings)


def _sample_full_df(days: int = 400) -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=days, freq="B")
    base = pd.Series(range(days), dtype="float")
    df = pd.DataFrame(
        {
            "date": dates,
            "open": 100 + base,
            "high": 101 + base,
            "low": 99 + base,
            "close": 100.5 + base,
            "adjusted_close": 100.4 + base,
            "volume": 1_000_000 + base * 10,
        }
    )
    return df


def test_extract_rolling_with_indicators(tmp_path):
    cm = _build_cache_manager(tmp_path)
    full_df = _sample_full_df(days=400)
    cm.write_atomic(full_df, "AAA", "full")

    stats = extract_rolling_from_full(cm)

    assert stats.total_symbols == 1
    assert stats.updated_symbols == 1
    assert not stats.errors

    rolling_df = cm.read("AAA", "rolling")
    assert rolling_df is not None
    assert len(rolling_df) == 330

    start_expected = pd.Timestamp(full_df["date"].iloc[-330])
    first_date = pd.to_datetime(rolling_df.iloc[0]["date"])
    assert first_date == start_expected

    indicator_cols = {
        "sma25",
        "sma50",
        "sma100",
        "sma150",
        "sma200",
        "roc200",
        "atr10",
        "atr20",
        "atr40",
        "atr50",
        "rsi3",
        "rsi4",
        "adx7",
        "dollarvolume20",
        "dollarvolume50",
        "avgvolume50",
        "atr_ratio",
        "atr_pct",
        "return_3d",
        "return_6d",
        "uptwodays",
        "twodayup",
        "hv50",
        "min_50",
        "max_70",
    }
    missing = indicator_cols - set(rolling_df.columns)
    assert not missing

    last_row = rolling_df.iloc[-1]
    assert pd.notna(last_row["sma200"])
    assert pd.notna(last_row["atr20"])


def test_extract_subset_and_target_days(tmp_path):
    cm = _build_cache_manager(tmp_path)
    df = _sample_full_df(days=50)
    cm.write_atomic(df, "XYZ", "full")

    stats = extract_rolling_from_full(cm, symbols=["XYZ"], target_days=40)

    assert stats.total_symbols == 1
    assert stats.updated_symbols == 1
    rolling_df = cm.read("XYZ", "rolling")
    assert len(rolling_df) == 40


def test_extract_uses_symbol_manifest(tmp_path):
    cm = _build_cache_manager(tmp_path)
    df = _sample_full_df(days=120)
    cm.write_atomic(df, "AAA", "full")
    cm.write_atomic(df, "BBB", "full")
    cm.write_atomic(df, "CCC", "full")

    save_symbol_manifest(["AAA", "CCC", "ZZZ"], cm.full_dir)

    stats = extract_rolling_from_full(cm)

    assert stats.total_symbols == 2
    assert stats.updated_symbols == 2
    assert stats.errors == {}
    assert stats.skipped_no_data == 0
    assert cm.read("AAA", "rolling") is not None
    assert cm.read("BBB", "rolling") is None
    assert cm.read("CCC", "rolling") is not None
    assert cm.read("ZZZ", "rolling") is None


def test_extract_manifest_missing_symbols_falls_back(tmp_path):
    cm = _build_cache_manager(tmp_path)
    df = _sample_full_df(days=120)
    cm.write_atomic(df, "AAA", "full")

    save_symbol_manifest(["ZZZ"], cm.full_dir)

    stats = extract_rolling_from_full(cm)

    assert stats.total_symbols == 1
    assert stats.updated_symbols == 1
    assert stats.errors == {}
    assert stats.skipped_no_data == 0
    assert cm.read("AAA", "rolling") is not None
