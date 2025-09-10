from types import SimpleNamespace

import pandas as pd

from common.cache_manager import CacheManager


class DummyRolling(SimpleNamespace):
    base_lookback_days = 200
    buffer_days = 40
    prune_chunk_days = 30
    meta_file = "_meta.json"


def _build_cm(tmp_path):
    cache = SimpleNamespace(
        full_dir=tmp_path,
        rolling_dir=tmp_path,
        rolling=DummyRolling(),
        file_format="csv",
    )
    settings = SimpleNamespace(cache=cache)
    return CacheManager(settings)


def test_read_handles_uppercase_date(tmp_path):
    cm = _build_cm(tmp_path)
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=2),
            "Open": [1, 2],
            "High": [1, 2],
            "Low": [1, 2],
            "Close": [1, 2],
            "Volume": [100, 200],
        }
    )
    df.to_csv(tmp_path / "AAA.csv", index=False)
    out = cm.read("AAA", "full")
    assert out is not None
    assert list(out.columns)[:6] == ["date", "open", "high", "low", "close", "volume"]
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
