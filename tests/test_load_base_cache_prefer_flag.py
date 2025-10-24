import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from common.cache_manager import _read_legacy_cache, base_cache_path, load_base_cache


def test_load_base_cache_prefers_precomputed(tmp_path, monkeypatch):
    # prepare a CSV that already has indicator-like columns
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2021-01-01", periods=3),
            "Open": [1, 2, 3],
            "High": [1, 2, 3],
            "Low": [1, 2, 3],
            "Close": [1, 2, 3],
            "atr10": [0.1, 0.2, 0.15],
        }
    )
    symbol = "TEST_PREF"
    path = base_cache_path(symbol)
    # ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    # load with prefer_precomputed_indicators=True should return the CSV content
    out = load_base_cache(
        symbol, rebuild_if_missing=True, prefer_precomputed_indicators=True
    )
    assert out is not None
    assert "atr10" in out.columns

    # load with prefer_precomputed_indicators=False should compute indicators (still returns frame)
    out2 = load_base_cache(
        symbol, rebuild_if_missing=True, prefer_precomputed_indicators=False
    )
    assert out2 is not None
    assert "atr10" in out2.columns

    # cleanup
    try:
        os.remove(path)
    except Exception:
        pass


def test_load_base_cache_csv_read_error():
    """CSVファイル読み取りエラー時の処理をテスト"""
    with patch("common.cache_manager.base_cache_path") as mock_path_func:
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path_func.return_value = mock_path

        # CSV読み取りエラーをモック
        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.side_effect = pd.errors.ParserError("Invalid CSV")

            result = load_base_cache("ERROR_SYMBOL", rebuild_if_missing=False)
            assert result is None


def test_load_base_cache_freshness_with_none_values():
    """allowed_recent_datesにNoneやNaN値が含まれる場合のテスト"""
    # 既存データを模擬
    with patch("pandas.read_csv") as mock_read_csv:
        df = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=5),
                "Close": [100, 101, 102, 103, 104],
            }
        )
        mock_read_csv.return_value = df

        with patch("pathlib.Path.exists", return_value=True):
            # None、NaN、有効な日付が混在するallowed_dates
            allowed_dates = [
                None,
                pd.NaT,
                "2024-01-05",
                np.nan,
                pd.Timestamp("2024-01-05"),
            ]

            result = load_base_cache("FRESH_TEST", allowed_recent_dates=allowed_dates)
            # 有効な日付のみが考慮される
            assert result is not None
            assert len(result) == 5


def test_read_legacy_cache_exception():
    """_read_legacy_cache での例外処理をテスト"""
    # ファイルパスの構築をモック
    with patch("common.cache_manager.Path") as mock_path_cls:
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path_cls.return_value = mock_path

        # pandas.read_csvで予期しない例外
        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.side_effect = RuntimeError("Unexpected error")

            result = _read_legacy_cache("EXCEPTION_SYMBOL")
            assert result is None
