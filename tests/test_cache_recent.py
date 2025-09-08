from datetime import datetime
import importlib
import os
import sys
import types

import pandas as pd


def test_cache_single_creates_recent_from_existing_if_missing(tmp_path):
    dummy_cm = types.ModuleType("common.cache_manager")

    class DummyCacheManager:  # pragma: no cover - simple stub
        def __init__(self, *args, **kwargs):
            pass

    dummy_cm.CacheManager = DummyCacheManager
    sys.modules["common.cache_manager"] = dummy_cm
    sys.modules.pop("scripts.cache_daily_data", None)
    os.environ["SKIP_CACHE_MIGRATION"] = "1"
    cache_daily_data = importlib.import_module("scripts.cache_daily_data")
    cache_single = cache_daily_data.cache_single
    symbol = "AAA"
    output_dir = tmp_path / "cache"
    recent_dir = tmp_path / "recent"
    output_dir.mkdir()
    # create existing cache file with today's date
    dates = pd.date_range(end=datetime.today(), periods=5)
    df = pd.DataFrame({"date": dates, "close": range(5)})
    filepath = output_dir / f"{symbol}.csv"
    df.to_csv(filepath)
    # ensure recent file does not exist
    recent_path = recent_dir / f"{symbol}.csv"
    assert not recent_path.exists()

    msg, used_api, success = cache_single(symbol, output_dir, recent_dir, recent_days=2)

    assert success is True
    assert used_api is False
    assert msg == f"{symbol}: already cached"
    assert recent_path.exists()
    expected_df = pd.read_csv(filepath).tail(2).reset_index(drop=True)
    recent_df = pd.read_csv(recent_path).reset_index(drop=True)
    pd.testing.assert_frame_equal(recent_df, expected_df)
