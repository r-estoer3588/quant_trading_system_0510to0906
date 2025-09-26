"""
Tests for core.system1 module targeting key functions for improved coverage
Focus on utility functions that can be tested independently
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from common.testing import set_test_determinism

# Import functions directly to avoid dependency issues
try:
    from core.system1 import (
        _rename_ohlcv,
        _normalize_index,
        _prepare_source_frame,
        _compute_indicators_frame,
        REQUIRED_COLUMNS,
    )

    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


class TestSystem1Utilities:
    """Test system1 utility functions for improved coverage"""

    def setup_method(self):
        set_test_determinism()
        if not IMPORTS_AVAILABLE:
            pytest.skip("core.system1 imports not available")

    def test_rename_ohlcv_lowercase_to_uppercase(self):
        """Test _rename_ohlcv function with lowercase columns"""
        df = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [103, 104, 105],
                "volume": [1000, 1100, 1200],
            }
        )

        result = _rename_ohlcv(df)

        assert "Open" in result.columns
        assert "High" in result.columns
        assert "Low" in result.columns
        assert "Close" in result.columns
        assert "Volume" in result.columns

        # Verify data integrity
        assert result["Open"].iloc[0] == 100
        assert result["High"].iloc[1] == 106

    def test_rename_ohlcv_already_uppercase(self):
        """Test _rename_ohlcv with already uppercase columns"""
        df = pd.DataFrame(
            {
                "Open": [100, 101],
                "High": [105, 106],
                "Low": [95, 96],
                "Close": [103, 104],
                "Volume": [1000, 1100],
            }
        )

        result = _rename_ohlcv(df)

        # Should remain unchanged
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert result["Open"].iloc[0] == 100

    def test_rename_ohlcv_mixed_case(self):
        """Test _rename_ohlcv with mixed case scenarios"""
        df = pd.DataFrame(
            {
                "open": [100],
                "High": [105],  # Already uppercase
                "low": [95],
                "Close": [103],  # Already uppercase
                "volume": [1000],
            }
        )

        result = _rename_ohlcv(df)

        assert "Open" in result.columns  # Renamed from 'open'
        assert "High" in result.columns  # Unchanged
        assert "Low" in result.columns  # Renamed from 'low'
        assert "Close" in result.columns  # Unchanged
        assert "Volume" in result.columns  # Renamed from 'volume'


class TestNormalizeIndex:
    """Test _normalize_index function"""

    def setup_method(self):
        set_test_determinism()
        if not IMPORTS_AVAILABLE:
            pytest.skip("core.system1 imports not available")

    def test_normalize_index_with_date_column(self):
        """Test _normalize_index with Date column"""
        df = pd.DataFrame(
            {"Date": ["2023-01-01", "2023-01-02", "2023-01-03"], "Close": [100, 101, 102]}
        )

        result = _normalize_index(df)

        assert result.index.name == "Date"
        assert len(result) == 3
        assert pd.api.types.is_datetime64_any_dtype(result.index)

    def test_normalize_index_with_lowercase_date(self):
        """Test _normalize_index with lowercase date column"""
        df = pd.DataFrame({"date": ["2023-01-01", "2023-01-02"], "Close": [100, 101]})

        result = _normalize_index(df)

        assert result.index.name == "Date"
        assert len(result) == 2

    def test_normalize_index_from_existing_index(self):
        """Test _normalize_index using existing datetime index"""
        dates = pd.date_range("2023-01-01", periods=3, freq="D")
        df = pd.DataFrame({"Close": [100, 101, 102]}, index=dates)

        result = _normalize_index(df)

        assert result.index.name == "Date"
        assert len(result) == 3

    def test_normalize_index_removes_duplicates(self):
        """Test _normalize_index removes duplicate dates"""
        df = pd.DataFrame(
            {
                "Date": ["2023-01-01", "2023-01-01", "2023-01-02"],
                "Close": [100, 101, 102],  # Duplicate date with different values
            }
        )

        result = _normalize_index(df)

        # Should keep last occurrence
        assert len(result) == 2
        assert result.index.name == "Date"


class TestPrepareSourceFrame:
    """Test _prepare_source_frame function"""

    def setup_method(self):
        set_test_determinism()
        if not IMPORTS_AVAILABLE:
            pytest.skip("core.system1 imports not available")

    def test_prepare_source_frame_valid_data(self):
        """Test _prepare_source_frame with valid OHLCV data"""
        df = pd.DataFrame(
            {
                "Date": ["2023-01-01", "2023-01-02"],
                "Open": [100, 101],
                "High": [105, 106],
                "Low": [95, 96],
                "Close": [103, 104],
                "Volume": [1000, 1100],
            }
        )

        result = _prepare_source_frame(df)

        assert all(col in result.columns for col in REQUIRED_COLUMNS)
        assert len(result) == 2
        assert result.index.name == "Date"

    def test_prepare_source_frame_empty_input(self):
        """Test _prepare_source_frame with empty DataFrame"""
        df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty_frame"):
            _prepare_source_frame(df)

    def test_prepare_source_frame_missing_columns(self):
        """Test _prepare_source_frame with missing required columns"""
        df = pd.DataFrame(
            {
                "Date": ["2023-01-01"],
                "Open": [100],
                # Missing High, Low, Close, Volume
            }
        )

        with pytest.raises(ValueError, match="missing_cols"):
            _prepare_source_frame(df)

    def test_prepare_source_frame_with_na_values(self):
        """Test _prepare_source_frame removes rows with NA in essential columns"""
        df = pd.DataFrame(
            {
                "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "Open": [100, 101, 102],
                "High": [105, np.nan, 107],  # NaN in High
                "Low": [95, 96, 97],
                "Close": [103, 104, np.nan],  # NaN in Close
                "Volume": [1000, 1100, 1200],
            }
        )

        result = _prepare_source_frame(df)

        # Should remove rows with NaN in High/Low/Close
        assert len(result) == 1  # Only first row should remain
        assert result.iloc[0]["Open"] == 100


class TestComputeIndicatorsFrame:
    """Test _compute_indicators_frame function"""

    def setup_method(self):
        set_test_determinism()
        if not IMPORTS_AVAILABLE:
            pytest.skip("core.system1 imports not available")

    def test_compute_indicators_basic(self):
        """Test _compute_indicators_frame computes expected indicators"""
        # Create DataFrame with enough data for indicators
        dates = pd.date_range("2023-01-01", periods=250, freq="D")
        df = pd.DataFrame(
            {
                "Open": range(100, 350),
                "High": range(105, 355),
                "Low": range(95, 345),
                "Close": range(103, 353),
                "Volume": range(1000, 1250),
            },
            index=dates,
        )

        result = _compute_indicators_frame(df)

        # Check that indicators are computed
        assert "SMA25" in result.columns
        assert "SMA50" in result.columns
        assert "ROC200" in result.columns
        assert "ATR20" in result.columns
        assert "DollarVolume20" in result.columns
        assert "filter" in result.columns

        # Verify SMA25 calculation for recent data
        recent_sma25 = result["Close"].iloc[-25:].mean()
        computed_sma25 = result["SMA25"].iloc[-1]
        assert abs(recent_sma25 - computed_sma25) < 0.01

    def test_compute_indicators_filter_logic(self):
        """Test _compute_indicators_frame filter column logic"""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        df = pd.DataFrame(
            {
                "Open": [10] * 50,  # Price > 5
                "High": [12] * 50,
                "Low": [8] * 50,  # Low >= 5
                "Close": [10] * 50,
                "Volume": [10_000_000] * 50,  # High volume
            },
            index=dates,
        )

        result = _compute_indicators_frame(df)

        # DollarVolume20 should be 10 * 10M = 100M > 50M threshold
        assert "filter" in result.columns
        # After warm-up period, filter should be True
        assert result["filter"].iloc[-1]

    def test_compute_indicators_roc200_calculation(self):
        """Test ROC200 calculation specifically"""
        dates = pd.date_range("2023-01-01", periods=250, freq="D")
        # Create prices with known ROC200
        close_prices = [100] * 200 + [120] * 50  # 20% increase after 200 days
        df = pd.DataFrame(
            {
                "Open": close_prices,
                "High": [p + 5 for p in close_prices],
                "Low": [p - 5 for p in close_prices],
                "Close": close_prices,
                "Volume": [1000] * 250,
            },
            index=dates,
        )

        result = _compute_indicators_frame(df)

        # ROC200 at end should be approximately 20%
        roc200_final = result["ROC200"].iloc[-1]
        expected_roc = ((120 - 100) / 100) * 100  # 20%
        assert abs(roc200_final - expected_roc) < 0.01


if __name__ == "__main__":
    pytest.main([__file__])
