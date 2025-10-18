"""Tests for core.system1 module to improve coverage."""

from unittest.mock import patch

import pandas as pd
import pytest

from core.system1 import (
    _compute_indicators,
    _compute_indicators_frame,
    _normalize_index,
    _prepare_source_frame,
    _rename_ohlcv,
    generate_candidates_system1,
    generate_roc200_ranking_system1,
    get_total_days_system1,
    prepare_data_vectorized_system1,
)


class TestSystem1HelperFunctions:
    """Test System1 helper functions."""

    def test_rename_ohlcv_with_lowercase_columns(self):
        """Test _rename_ohlcv with lowercase column names."""
        df = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [104, 103, 105],
                "volume": [1000, 1100, 1200],
            }
        )

        result = _rename_ohlcv(df)

        expected_columns = {"Open", "High", "Low", "Close", "Volume"}
        assert set(result.columns) == expected_columns

    def test_rename_ohlcv_already_uppercase(self):
        """Test _rename_ohlcv with already uppercase columns."""
        df = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [105, 106, 107],
                "Low": [95, 96, 97],
                "Close": [104, 103, 105],
                "Volume": [1000, 1100, 1200],
            }
        )

        result = _rename_ohlcv(df)

        # Should not change anything
        pd.testing.assert_frame_equal(result, df)

    def test_rename_ohlcv_mixed_case(self):
        """Test _rename_ohlcv with mixed case columns."""
        df = pd.DataFrame(
            {
                "open": [100],
                "High": [105],  # Already uppercase
                "low": [95],
                "Close": [104],  # Already uppercase
                "volume": [1000],
            }
        )

        result = _rename_ohlcv(df)

        # Should rename only the lowercase ones
        assert "Open" in result.columns
        assert "High" in result.columns
        assert "Low" in result.columns
        assert "Close" in result.columns
        assert "Volume" in result.columns

    def test_normalize_index_with_date_column(self):
        """Test _normalize_index with Date column."""
        df = pd.DataFrame(
            {
                "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "Close": [100, 101, 102],
            }
        )

        result = _normalize_index(df)

        # Check that index is datetime
        assert hasattr(result.index, "normalize")
        assert len(result) == 3

    def test_normalize_index_without_date_column(self):
        """Test _normalize_index without Date column."""
        df = pd.DataFrame({"Close": [100, 101, 102], "Volume": [1000, 1100, 1200]})

        result = _normalize_index(df)

        # Should return the same DataFrame
        pd.testing.assert_frame_equal(result, df)

    def test_prepare_source_frame_basic(self):
        """Test _prepare_source_frame function."""
        df = pd.DataFrame(
            {
                "High": [105, 106, 107],
                "Low": [95, 96, 97],
                "Close": [104, 103, 105],
                "Volume": [1000, 1100, 1200],
            }
        )

        result = _prepare_source_frame(df)

        # Should drop rows with NaN in High, Low, Close
        assert len(result) <= len(df)
        required_cols = ["High", "Low", "Close"]
        for col in required_cols:
            if col in result.columns:
                assert not result[col].isna().any()

    def test_compute_indicators_frame_with_required_indicators(self):
        """Test _compute_indicators_frame with all required indicators."""
        df = pd.DataFrame(
            {
                "High": [105, 106, 107],
                "Low": [6, 7, 8],  # Above 5 threshold
                "Close": [104, 103, 105],
                "sma25": [100, 101, 102],
                "sma50": [98, 99, 100],
                "roc200": [0.1, 0.2, 0.15],
                "atr20": [2.5, 2.6, 2.7],
                "dollarvolume20": [60_000_000, 70_000_000, 80_000_000],
            }
        )

        result = _compute_indicators_frame(df)

        # Check that filter and setup columns are added
        assert "filter" in result.columns
        assert "setup" in result.columns
        # Filter should be True (Low >= 5 and dollarvolume20 > 50M)
        assert result["filter"].any()
        # Setup should depend on sma25 > sma50
        expected_setup = result["filter"] & (result["sma25"] > result["sma50"])
        pd.testing.assert_series_equal(result["setup"], expected_setup)

    def test_compute_indicators_frame_missing_indicators(self):
        """Test _compute_indicators_frame with missing indicators."""
        df = pd.DataFrame(
            {
                "High": [105, 106, 107],
                "Low": [6, 7, 8],
                "Close": [104, 103, 105],
                "sma25": [100, 101, 102],
                # Missing other required indicators
            }
        )

        with pytest.raises(ValueError, match="missing precomputed indicators"):
            _compute_indicators_frame(df)


class TestSystem1MainFunctions:
    """Test System1 main processing functions."""

    @patch("core.system1.get_cached_data")
    def test_compute_indicators_success(self, mock_get_data):
        """Test _compute_indicators with valid data."""
        mock_df = pd.DataFrame(
            {
                "Close": [100, 101, 102],
                "Low": [6, 7, 8],
                "sma25": [100, 101, 102],
                "sma50": [98, 99, 100],
                "roc200": [0.1, 0.2, 0.15],
                "atr20": [2.5, 2.6, 2.7],
                "dollarvolume20": [60_000_000, 70_000_000, 80_000_000],
            }
        )
        mock_get_data.return_value = mock_df

        symbol, result = _compute_indicators("TEST")

        assert symbol == "TEST"
        assert result is not None
        assert "filter" in result.columns
        assert "setup" in result.columns

    @patch("core.system1.get_cached_data")
    def test_compute_indicators_no_data(self, mock_get_data):
        """Test _compute_indicators with no data."""
        mock_get_data.return_value = None

        symbol, result = _compute_indicators("TEST")

        assert symbol == "TEST"
        assert result is None

    @patch("core.system1.get_cached_data")
    def test_compute_indicators_empty_data(self, mock_get_data):
        """Test _compute_indicators with empty DataFrame."""
        mock_get_data.return_value = pd.DataFrame()

        symbol, result = _compute_indicators("TEST")

        assert symbol == "TEST"
        assert result is None

    @patch("core.system1.get_cached_data")
    def test_compute_indicators_missing_indicators(self, mock_get_data):
        """Test _compute_indicators with missing indicators."""
        mock_df = pd.DataFrame(
            {
                "Close": [100, 101, 102],
                "Low": [6, 7, 8],
                # Missing required indicators
            }
        )
        mock_get_data.return_value = mock_df

        symbol, result = _compute_indicators("TEST")

        assert symbol == "TEST"
        assert result is None

    def test_generate_candidates_system1_with_valid_data(self):
        """Test generate_candidates_system1 with valid setup data."""
        data_dict = {
            "TEST1": pd.DataFrame(
                {
                    "Close": [100, 101, 102, 103, 104],
                    "setup": [True, True, False, True, True],
                    "roc200": [0.15, 0.25, 0.05, 0.20, 0.10],
                },
                index=pd.date_range("2023-01-01", periods=5),
            )
        }

        # Test with top_n=3
        result = generate_candidates_system1(data_dict, target_date="2023-01-05", top_n=3)

        assert isinstance(result, dict)

    def test_get_total_days_system1_basic(self):
        """Test get_total_days_system1 function."""
        data_dict = {
            "TEST1": pd.DataFrame(index=pd.date_range("2023-01-01", periods=10)),
            "TEST2": pd.DataFrame(index=pd.date_range("2023-01-01", periods=15)),
        }

        result = get_total_days_system1(data_dict)

        # Should return the maximum number of days
        assert result == 15

    def test_generate_roc200_ranking_system1_basic(self):
        """Test generate_roc200_ranking_system1 function."""
        data_dict = {
            "TEST1": pd.DataFrame(
                {
                    "Close": [100, 101, 102],
                    "setup": [True, True, False],
                    "roc200": [0.2, 0.15, 0.1],
                },
                index=pd.date_range("2023-01-01", periods=3),
            ),
            "TEST2": pd.DataFrame(
                {
                    "Close": [200, 201, 202],
                    "setup": [True, False, True],
                    "roc200": [0.3, 0.25, 0.35],
                },
                index=pd.date_range("2023-01-01", periods=3),
            ),
        }

        result = generate_roc200_ranking_system1(data_dict, date="2023-01-03", top_n=5)

        assert isinstance(result, list)


class TestSystem1IntegrationFunctions:
    """Test System1 integration and signal generation functions."""

    @patch("core.system1.process_symbols_batch")
    def test_prepare_data_vectorized_system1(self, mock_process_batch):
        """Test prepare_data_vectorized_system1 integration."""
        # Mock the batch processing
        mock_result = {
            "TEST1": pd.DataFrame(
                {
                    "Close": [100, 101, 102],
                    "setup": [True, True, False],
                    "roc200": [0.2, 0.15, 0.1],
                },
                index=pd.date_range("2023-01-01", periods=3),
            )
        }
        mock_process_batch.return_value = mock_result

        # Provide proper data dict instead of symbols list
        raw_data = {
            "TEST1": pd.DataFrame(
                {
                    "Close": [100, 101, 102],
                    "Low": [6, 7, 8],
                    "sma25": [100, 101, 102],
                    "sma50": [98, 99, 100],
                    "roc200": [0.1, 0.2, 0.15],
                    "atr20": [2.5, 2.6, 2.7],
                    "dollarvolume20": [60_000_000, 70_000_000, 80_000_000],
                }
            )
        }
        result = prepare_data_vectorized_system1(raw_data)

        assert isinstance(result, dict)
        mock_process_batch.assert_called()
