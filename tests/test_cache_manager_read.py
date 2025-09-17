from types import SimpleNamespace

import pandas as pd
import pytest

from common.cache_manager import CacheManager


class DummyRolling(SimpleNamespace):
    base_lookback_days = 200
    buffer_days = 100
    prune_chunk_days = 30
    meta_file = "_meta.json"


def _build_cm(tmp_path, file_format: str = "csv"):
    cache = SimpleNamespace(
        full_dir=tmp_path,
        rolling_dir=tmp_path,
        rolling=DummyRolling(),
        file_format=file_format,
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


def test_read_supports_feather(tmp_path):
    pytest.importorskip("pyarrow")
    cm = _build_cm(tmp_path, file_format="feather")
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2),
            "open": [1.0, 2.0],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.2, 2.2],
            "volume": [100, 200],
        }
    )
    df.to_feather(tmp_path / "AAA.feather")
    out = cm.read("AAA", "rolling")
    assert out is not None
    assert "close" in out.columns
    assert len(out) == 2


def test_write_atomic_feather(tmp_path):
    pytest.importorskip("pyarrow")
    cm = _build_cm(tmp_path, file_format="feather")
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-02-01", periods=3),
            "open": [1.0, 2.0, 3.0],
            "high": [1.5, 2.5, 3.5],
            "low": [0.5, 1.5, 2.5],
            "close": [1.2, 2.2, 3.2],
            "volume": [100, 200, 300],
        }
    )
    cm.write_atomic(df, "BBB", "rolling")
    saved = pd.read_feather(tmp_path / "BBB.feather")
    assert list(saved.columns)[:5] == ["date", "open", "high", "low", "close"]
    assert len(saved) == len(df)
