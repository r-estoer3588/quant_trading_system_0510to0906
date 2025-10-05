"""Tests for common.indicators_common module to improve coverage."""

from unittest.mock import patch

import pandas as pd

from common.indicators_common import _add_indicators_optimized, add_indicators, add_indicators_batch


class TestIndicatorsCommonBasicFunctions:
    """Test basic indicator calculation functions."""

    def test_add_indicators_with_valid_data(self):
        """Test add_indicators with valid OHLCV data."""
        df = pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104],
                "High": [105, 106, 107, 108, 109],
                "Low": [95, 96, 97, 98, 99],
                "Close": [104, 103, 105, 106, 107],
                "Volume": [1000, 1100, 1200, 1300, 1400],
            },
            index=pd.date_range("2023-01-01", periods=5),
        )

        result = add_indicators(df)

        # Check that indicators are added
        assert "atr20" in result.columns
        assert "sma25" in result.columns
        assert "sma50" in result.columns
        # Check that data is not empty
        assert len(result) == 5

    def test_add_indicators_with_empty_data(self):
        """Test add_indicators with empty DataFrame."""
        df = pd.DataFrame()

        result = add_indicators(df)

        assert result.empty

    def test_add_indicators_with_none_input(self):
        """Test add_indicators with None input."""
        result = add_indicators(None)

        assert result is None

    def test_add_indicators_with_precomputed_indicators(self):
        """Test add_indicators skips precomputed indicators."""
        df = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [105, 106, 107],
                "Low": [95, 96, 97],
                "Close": [104, 103, 105],
                "Volume": [1000, 1100, 1200],
                "atr20": [2.5, 2.6, 2.7],  # Precomputed
                "sma25": [102, 103, 104],  # Precomputed
            }
        )

        result = add_indicators(df)

        # Should not overwrite precomputed values
        pd.testing.assert_series_equal(result["atr20"], df["atr20"])
        pd.testing.assert_series_equal(result["sma25"], df["sma25"])

    def test_add_indicators_with_zero_close_values(self):
        """Test add_indicators handles zero close values."""
        df = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [105, 106, 107],
                "Low": [95, 96, 97],
                "Close": [104, 0, 105],  # Zero value
                "Volume": [1000, 1100, 1200],
            }
        )

        result = add_indicators(df)

        # Should handle zero values gracefully
        assert len(result) == 3
        assert "Close" in result.columns

    def test_add_indicators_optimized_basic(self):
        """Test _add_indicators_optimized internal function."""
        df = pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104],
                "High": [105, 106, 107, 108, 109],
                "Low": [95, 96, 97, 98, 99],
                "Close": [104, 103, 105, 106, 107],
                "Volume": [1000, 1100, 1200, 1300, 1400],
            }
        )

        result = _add_indicators_optimized(df)

        # Should return DataFrame with indicators
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5


class TestIndicatorsBatchFunctions:
    """Test batch indicator processing functions."""

    def test_add_indicators_batch_basic(self):
        """Test add_indicators_batch function."""
        data_dict = {
            "SYMBOL1": pd.DataFrame(
                {
                    "Open": [100, 101, 102],
                    "High": [105, 106, 107],
                    "Low": [95, 96, 97],
                    "Close": [104, 103, 105],
                    "Volume": [1000, 1100, 1200],
                }
            ),
            "SYMBOL2": pd.DataFrame(
                {
                    "Open": [200, 201, 202],
                    "High": [205, 206, 207],
                    "Low": [195, 196, 197],
                    "Close": [204, 203, 205],
                    "Volume": [2000, 2100, 2200],
                }
            ),
        }

        result = add_indicators_batch(data_dict)

        assert isinstance(result, dict)
        assert "SYMBOL1" in result
        assert "SYMBOL2" in result
        # Check that indicators were added
        for _symbol, df in result.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0

    def test_add_indicators_batch_with_empty_dict(self):
        """Test add_indicators_batch with empty dictionary."""
        result = add_indicators_batch({})

        assert result == {}

    def test_add_indicators_batch_with_invalid_data(self):
        """Test add_indicators_batch with invalid data."""
        data_dict = {
            "VALID": pd.DataFrame(
                {
                    "Close": [100, 101, 102],
                    "High": [105, 106, 107],
                    "Low": [95, 96, 97],
                    "Volume": [1000, 1100, 1200],
                }
            ),
            "EMPTY": pd.DataFrame(),
            "NONE": None,
        }

        result = add_indicators_batch(data_dict)

        # Should handle invalid data gracefully
        assert isinstance(result, dict)


class TestIndicatorsEdgeCases:
    """Test edge cases and error handling."""

    def test_add_indicators_missing_columns(self):
        """Test add_indicators with missing required columns."""
        df = pd.DataFrame({"SomeOtherColumn": [1, 2, 3]})

        result = add_indicators(df)

        # Should handle missing columns gracefully
        assert "atr20" in result.columns
        assert "sma25" in result.columns
        # Values should be NaN due to missing data
        assert result["atr20"].isna().all()
        assert result["sma25"].isna().all()

    def test_indicators_with_insufficient_data(self):
        """Test indicators with insufficient data points."""
        df = pd.DataFrame(
            {
                "Open": [100],
                "High": [105],
                "Low": [95],
                "Close": [104],
                "Volume": [1000],
            }
        )

        result = add_indicators(df)

        # Should handle insufficient data
        assert len(result) == 1
        # Most indicators should be NaN due to insufficient data
        assert pd.isna(result["sma25"].iloc[0])
        assert pd.isna(result["atr20"].iloc[0])

    @patch("common.indicators_common.SMAIndicator")
    def test_add_indicators_with_calculation_error(self, mock_sma):
        """Test add_indicators handles calculation errors."""
        # Mock SMA to raise an exception
        mock_sma.side_effect = Exception("Calculation error")

        df = pd.DataFrame(
            {
                "Close": [100, 101, 102],
                "High": [105, 106, 107],
                "Low": [95, 96, 97],
                "Volume": [1000, 1100, 1200],
            }
        )

        result = add_indicators(df)

        # Should handle errors gracefully
        assert "sma25" in result.columns
        # Values should be NaN due to calculation error
        assert result["sma25"].isna().all()
