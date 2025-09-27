"""
Tests for common.dataframe_utils module to improve coverage
Focus on DataFrame processing utility functions
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from common.dataframe_utils import (
    DataFrameNormalizationError,
    ensure_numeric_columns,
    normalize_column_names,
    normalize_date_column,
    prepare_dataframe_for_cache,
    round_dataframe,
    standardize_ohlcv_columns,
    validate_required_columns,
)
from common.testing import set_test_determinism


class TestDataFrameNormalizationError:
    """Test custom exception class"""

    def setup_method(self):
        set_test_determinism()

    def test_exception_creation(self):
        """Test DataFrameNormalizationError can be raised"""
        with pytest.raises(DataFrameNormalizationError):
            raise DataFrameNormalizationError("Test error")

    def test_exception_with_message(self):
        """Test DataFrameNormalizationError with custom message"""
        error_msg = "DataFrame processing failed"
        with pytest.raises(DataFrameNormalizationError) as exc_info:
            raise DataFrameNormalizationError(error_msg)

        assert str(exc_info.value) == error_msg


class TestNormalizeColumnNames:
    """Test normalize_column_names function"""

    def setup_method(self):
        set_test_determinism()

    def test_normalize_column_names_with_none(self):
        """Test normalize_column_names with None input"""
        result = normalize_column_names(None)
        assert result is None

    def test_normalize_column_names_with_empty_df(self):
        """Test normalize_column_names with empty DataFrame"""
        empty_df = pd.DataFrame()
        result = normalize_column_names(empty_df)
        assert result.equals(empty_df)

    def test_normalize_column_names_basic(self):
        """Test normalize_column_names with basic input"""
        df = pd.DataFrame({"Symbol": [1, 2], "Date": [3, 4], "Open": [5, 6]})
        result = normalize_column_names(df)

        expected_columns = ["symbol", "date", "open"]
        assert list(result.columns) == expected_columns
        assert result["symbol"].tolist() == [1, 2]

    def test_normalize_column_names_with_duplicates(self):
        """Test normalize_column_names with duplicate columns"""
        df = pd.DataFrame({"Symbol": [1, 2], "SYMBOL": [3, 4], "Date": [5, 6]})
        result = normalize_column_names(df)

        # Should keep the first occurrence and remove duplicates
        assert "symbol" in result.columns
        assert len([col for col in result.columns if col == "symbol"]) == 1
        assert result["symbol"].tolist() == [1, 2]  # First column data

    def test_normalize_column_names_with_numbers(self):
        """Test normalize_column_names with numeric column names"""
        df = pd.DataFrame({1: [1, 2], "Symbol": [3, 4], 2.5: [5, 6]})
        result = normalize_column_names(df)

        expected_columns = ["1", "symbol", "2.5"]
        assert list(result.columns) == expected_columns


class TestNormalizeDateColumn:
    """Test normalize_date_column function"""

    def setup_method(self):
        set_test_determinism()

    def test_normalize_date_column_with_none(self):
        """Test normalize_date_column with None input"""
        result = normalize_date_column(None)
        assert result is None

    def test_normalize_date_column_with_empty_df(self):
        """Test normalize_date_column with empty DataFrame"""
        empty_df = pd.DataFrame()
        result = normalize_date_column(empty_df)
        assert result.equals(empty_df)

    def test_normalize_date_column_no_date_col(self):
        """Test normalize_date_column with no date column"""
        df = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "price": [100, 200]})
        result = normalize_date_column(df)
        assert result.equals(df)

    def test_normalize_date_column_basic(self):
        """Test normalize_date_column with valid dates"""
        df = pd.DataFrame(
            {"date": ["2023-01-01", "2023-01-03", "2023-01-02"], "price": [100, 300, 200]}
        )
        result = normalize_date_column(df)

        # Should be sorted by date
        expected_dates = ["2023-01-01", "2023-01-02", "2023-01-03"]
        result_dates = result["date"].dt.strftime("%Y-%m-%d").tolist()
        assert result_dates == expected_dates
        assert result["price"].tolist() == [100, 200, 300]

    def test_normalize_date_column_with_duplicates(self):
        """Test normalize_date_column with duplicate dates"""
        df = pd.DataFrame(
            {"date": ["2023-01-01", "2023-01-01", "2023-01-02"], "price": [100, 150, 200]}
        )
        result = normalize_date_column(df)

        # Should remove duplicates and keep first occurrence
        assert len(result) == 2
        assert result["price"].tolist() == [100, 200]

    def test_normalize_date_column_with_nan(self):
        """Test normalize_date_column with NaN dates"""
        df = pd.DataFrame({"date": ["2023-01-01", None, "2023-01-02"], "price": [100, 150, 200]})
        result = normalize_date_column(df)

        # Should remove rows with NaN dates
        assert len(result) == 2
        assert result["price"].tolist() == [100, 200]

    @patch("common.dataframe_utils.logger")
    def test_normalize_date_column_conversion_error(self, mock_logger):
        """Test normalize_date_column with date conversion error"""
        # Create DataFrame with problematic date column
        df = pd.DataFrame(
            {"date": [1, 2, 3], "price": [100, 200, 300]}  # Non-date data that might cause issues
        )

        # Mock pd.to_datetime to raise an exception
        with patch("pandas.to_datetime", side_effect=Exception("Conversion error")):
            result = normalize_date_column(df)

            # Should return original DataFrame and log warning
            assert result.equals(df)
            mock_logger.warning.assert_called()


class TestEnsureNumericColumns:
    """Test ensure_numeric_columns function"""

    def setup_method(self):
        set_test_determinism()

    def test_ensure_numeric_columns_with_none(self):
        """Test ensure_numeric_columns with None input"""
        result = ensure_numeric_columns(None, ["price"])
        assert result is None

    def test_ensure_numeric_columns_with_empty_df(self):
        """Test ensure_numeric_columns with empty DataFrame"""
        empty_df = pd.DataFrame()
        result = ensure_numeric_columns(empty_df, ["price"])
        assert result.equals(empty_df)

    def test_ensure_numeric_columns_basic(self):
        """Test ensure_numeric_columns with valid data"""
        df = pd.DataFrame(
            {"symbol": ["AAPL", "MSFT"], "price": ["100.5", "200.75"], "volume": ["1000", "2000"]}
        )
        result = ensure_numeric_columns(df, ["price", "volume"])

        assert pd.api.types.is_numeric_dtype(result["price"])
        assert pd.api.types.is_numeric_dtype(result["volume"])
        assert result["price"].iloc[0] == 100.5
        assert result["volume"].iloc[0] == 1000

    def test_ensure_numeric_columns_missing_column(self):
        """Test ensure_numeric_columns with missing column"""
        df = pd.DataFrame({"symbol": ["AAPL", "MSFT"]})
        result = ensure_numeric_columns(df, ["price", "volume"])

        # Should not fail and return original DataFrame
        assert result.equals(df)

    def test_ensure_numeric_columns_invalid_data(self):
        """Test ensure_numeric_columns with invalid numeric data"""
        df = pd.DataFrame({"price": ["invalid", "200.75"], "volume": ["abc", "2000"]})
        result = ensure_numeric_columns(df, ["price", "volume"])

        # Invalid values should become NaN
        assert pd.isna(result["price"].iloc[0])
        assert result["price"].iloc[1] == 200.75
        assert pd.isna(result["volume"].iloc[0])
        assert result["volume"].iloc[1] == 2000

    @patch("common.dataframe_utils.logger")
    def test_ensure_numeric_columns_conversion_error(self, mock_logger):
        """Test ensure_numeric_columns with conversion error"""
        df = pd.DataFrame({"price": [100, 200]})

        # Mock pd.to_numeric to raise an exception
        with patch("pandas.to_numeric", side_effect=Exception("Conversion error")):
            ensure_numeric_columns(df, ["price"])

            # Should log warning
            mock_logger.warning.assert_called()


class TestStandardizeOhlcvColumns:
    """Test standardize_ohlcv_columns function"""

    def setup_method(self):
        set_test_determinism()

    def test_standardize_ohlcv_columns_with_none(self):
        """Test standardize_ohlcv_columns with None input"""
        result = standardize_ohlcv_columns(None)
        assert result is None

    def test_standardize_ohlcv_columns_with_empty_df(self):
        """Test standardize_ohlcv_columns with empty DataFrame"""
        empty_df = pd.DataFrame()
        result = standardize_ohlcv_columns(empty_df)
        assert result.equals(empty_df)

    def test_standardize_ohlcv_columns_basic(self):
        """Test standardize_ohlcv_columns with basic OHLCV data"""
        df = pd.DataFrame(
            {
                "date": ["2023-01-01"],
                "open": [100],
                "high": [110],
                "low": [90],
                "adjusted_close": [105],
                "volume": [1000],
            }
        )
        result = standardize_ohlcv_columns(df)

        expected_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        assert list(result.columns) == expected_columns

    def test_standardize_ohlcv_columns_alternative_names(self):
        """Test standardize_ohlcv_columns with alternative column names"""
        df = pd.DataFrame({"date": ["2023-01-01"], "adj_close": [105], "vol": [1000]})
        result = standardize_ohlcv_columns(df)

        assert "Date" in result.columns
        assert "Close" in result.columns
        assert "Volume" in result.columns

    def test_standardize_ohlcv_columns_no_matching_columns(self):
        """Test standardize_ohlcv_columns with no matching columns"""
        df = pd.DataFrame({"symbol": ["AAPL"], "sector": ["Tech"]})
        result = standardize_ohlcv_columns(df)

        # Should return copy of original DataFrame
        assert result.columns.tolist() == df.columns.tolist()
        # Should have same values but be different object
        assert result["symbol"].iloc[0] == "AAPL"
        assert result is not df  # Different object


class TestValidateRequiredColumns:
    """Test validate_required_columns function"""

    def setup_method(self):
        set_test_determinism()

    def test_validate_required_columns_with_none(self):
        """Test validate_required_columns with None input"""
        is_valid, missing = validate_required_columns(None, {"date", "close"})
        assert not is_valid
        assert missing == {"date", "close"}

    def test_validate_required_columns_with_empty_df(self):
        """Test validate_required_columns with empty DataFrame"""
        empty_df = pd.DataFrame()
        is_valid, missing = validate_required_columns(empty_df, {"date", "close"})
        assert not is_valid
        assert missing == {"date", "close"}

    def test_validate_required_columns_all_present(self):
        """Test validate_required_columns with all required columns present"""
        df = pd.DataFrame({"date": [1], "open": [2], "close": [3], "volume": [4]})
        is_valid, missing = validate_required_columns(df, {"date", "close"})
        assert is_valid
        assert missing == set()

    def test_validate_required_columns_some_missing(self):
        """Test validate_required_columns with some required columns missing"""
        df = pd.DataFrame({"date": [1], "open": [2]})
        is_valid, missing = validate_required_columns(df, {"date", "close", "volume"})
        assert not is_valid
        assert missing == {"close", "volume"}


class TestRoundDataFrame:
    """Test round_dataframe function"""

    def setup_method(self):
        set_test_determinism()

    def test_round_dataframe_with_none(self):
        """Test round_dataframe with None DataFrame"""
        result = round_dataframe(None, 2)
        assert result is None

    def test_round_dataframe_with_none_decimals(self):
        """Test round_dataframe with None decimals"""
        df = pd.DataFrame({"price": [100.123]})
        result = round_dataframe(df, None)
        assert result.equals(df)

    def test_round_dataframe_with_invalid_decimals(self):
        """Test round_dataframe with invalid decimals"""
        df = pd.DataFrame({"price": [100.123]})
        result = round_dataframe(df, "invalid")
        assert result.equals(df)

    def test_round_dataframe_price_columns(self):
        """Test round_dataframe with price columns"""
        df = pd.DataFrame({"open": [100.12345], "close": [101.98765], "high": [102.55555]})
        result = round_dataframe(df, 4)  # General decimals, but prices should round to 2

        assert result["open"].iloc[0] == 100.12
        assert result["close"].iloc[0] == 101.99
        assert result["high"].iloc[0] == 102.56

    def test_round_dataframe_volume_columns(self):
        """Test round_dataframe with volume columns"""
        df = pd.DataFrame({"volume": [1000.789], "avgvolume50": [2000.456]})
        result = round_dataframe(df, 2)

        # Volume should be rounded to 0 decimals and converted to Int64
        assert result["volume"].iloc[0] == 1001
        assert result["avgvolume50"].iloc[0] == 2000

    def test_round_dataframe_percentage_columns(self):
        """Test round_dataframe with percentage columns"""
        df = pd.DataFrame({"roc200": [0.123456], "return_3d": [0.098765]})
        result = round_dataframe(df, 2)  # General decimals, but pct should round to 4

        assert result["roc200"].iloc[0] == 0.1235
        assert result["return_3d"].iloc[0] == 0.0988

    def test_round_dataframe_generic_numeric_columns(self):
        """Test round_dataframe with generic numeric columns"""
        df = pd.DataFrame({"custom_metric": [1.23456789], "another_value": [9.87654321]})
        result = round_dataframe(df, 3)

        assert result["custom_metric"].iloc[0] == 1.235
        assert result["another_value"].iloc[0] == 9.877


class TestPrepareDataFrameForCache:
    """Test prepare_dataframe_for_cache function"""

    def setup_method(self):
        set_test_determinism()

    def test_prepare_dataframe_for_cache_with_none(self):
        """Test prepare_dataframe_for_cache with None input"""
        result = prepare_dataframe_for_cache(None)
        assert result is None

    def test_prepare_dataframe_for_cache_with_empty_df(self):
        """Test prepare_dataframe_for_cache with empty DataFrame"""
        empty_df = pd.DataFrame()
        result = prepare_dataframe_for_cache(empty_df)
        assert result.equals(empty_df)

    def test_prepare_dataframe_for_cache_basic(self):
        """Test prepare_dataframe_for_cache with basic data"""
        df = pd.DataFrame(
            {
                "Date": ["2023-01-02", "2023-01-01"],
                "Open": ["100.5", "99.0"],
                "Close": ["101.0", "100.5"],
                "Volume": ["1000", "2000"],
            }
        )
        result = prepare_dataframe_for_cache(df)

        # Should normalize column names and sort by date
        assert "date" in result.columns
        assert "open" in result.columns
        assert pd.api.types.is_numeric_dtype(result["open"])
        assert pd.api.types.is_numeric_dtype(result["volume"])

        # Should be sorted by date
        dates = pd.to_datetime(result["date"])
        assert dates.iloc[0] < dates.iloc[1]


if __name__ == "__main__":
    pytest.main([__file__])
