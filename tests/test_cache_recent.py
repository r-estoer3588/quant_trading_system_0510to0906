import importlib
import os
import sys
import types
from datetime import datetime

import pandas as pd


def test_cache_single_creates_base_from_existing_if_missing(tmp_path):
    dummy_cm = types.ModuleType("common.cache_manager")

    class DummyCacheManager:  # pragma: no cover - simple stub
        def __init__(self, *args, **kwargs):
            pass

    def dummy_compute_base(df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        if "date" in frame.columns:
            frame["Date"] = pd.to_datetime(frame["date"])  # normalize
        else:
            frame["Date"] = pd.to_datetime(frame.index)
        frame = frame.rename(columns={"close": "Close"})
        cols = ["Date"] + [c for c in frame.columns if c != "Date"]
        return frame.loc[:, cols].set_index("Date")

    dummy_cm.CacheManager = DummyCacheManager
    dummy_cm.compute_base_indicators = dummy_compute_base
    sys.modules["common.cache_manager"] = dummy_cm
    sys.modules.pop("scripts.cache_daily_data", None)
    os.environ["SKIP_CACHE_MIGRATION"] = "1"
    cache_daily_data = importlib.import_module("scripts.cache_daily_data")
    cache_single = cache_daily_data.cache_single
    symbol = "AAA"
    output_dir = tmp_path / "cache"
    base_dir = tmp_path / "base"
    output_dir.mkdir()
    # create existing cache file with today's date
    dates = pd.date_range(end=datetime.today(), periods=5)
    df = pd.DataFrame({"date": dates, "close": range(5)})
    filepath = output_dir / f"{symbol}.csv"
    df.to_csv(filepath, index=False)
    base_path = base_dir / f"{symbol}.csv"
    assert not base_path.exists()

    msg, used_api, success = cache_single(symbol, output_dir, base_dir)

    assert success is True
    assert used_api is False
    assert msg == f"{symbol}: already cached"
    assert base_path.exists()
    base_df = pd.read_csv(base_path)
    assert "Date" in base_df.columns
    assert base_df["Close"].tolist() == list(df["close"])  # type: ignore[index]
