import pandas as pd
from common.cache_manager import load_base_cache, base_cache_path
import os


def test_load_base_cache_prefers_precomputed(tmp_path, monkeypatch):
    # prepare a CSV that already has indicator-like columns
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2021-01-01", periods=3),
            "Open": [1, 2, 3],
            "High": [1, 2, 3],
            "Low": [1, 2, 3],
            "Close": [1, 2, 3],
            "ATR10": [0.1, 0.2, 0.15],
        }
    )
    symbol = "TEST_PREF"
    path = base_cache_path(symbol)
    # ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    # load with prefer_precomputed_indicators=True should return the CSV content
    out = load_base_cache(symbol, rebuild_if_missing=True, prefer_precomputed_indicators=True)
    assert out is not None
    assert "ATR10" in out.columns or "atr10" in out.columns

    # load with prefer_precomputed_indicators=False should compute indicators (still returns frame)
    out2 = load_base_cache(symbol, rebuild_if_missing=True, prefer_precomputed_indicators=False)
    assert out2 is not None
    assert "ATR10" in out2.columns or "atr10" in out2.columns

    # cleanup
    try:
        os.remove(path)
    except Exception:
        pass
