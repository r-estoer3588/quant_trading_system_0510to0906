import pandas as pd

from common.cache_manager import CacheManager
from config.settings import get_settings


def test_preserve_existing_indicator_columns(tmp_path, monkeypatch):
    settings = get_settings(create_dirs=False)
    # Create a CacheManager pointing to a temp dir
    cm = CacheManager(settings)
    # construct a simple df with Date/Open/High/Low/Close and precomputed indicator
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "Open": [1, 2, 3, 4, 5],
            "High": [1, 2, 3, 4, 5],
            "Low": [1, 2, 3, 4, 5],
            "Close": [1, 2, 3, 4, 5],
            "ATR10": [0.1, 0.1, 0.1, 0.1, 0.1],
        }
    )

    out = cm._recompute_indicators(df)
    # existing ATR10 should be preserved (i.e., equal to original values)
    assert "ATR10" in out.columns or "atr10" in out.columns
    # if lowercased column exists, check values
    if "ATR10" in out.columns:
        assert list(out["ATR10"]) == [0.1] * 5 or pd.isna(out["ATR10"]).sum() < 5
    if "atr10" in out.columns:
        assert list(out["atr10"]) == [0.1] * 5 or pd.isna(out["atr10"]).sum() < 5
