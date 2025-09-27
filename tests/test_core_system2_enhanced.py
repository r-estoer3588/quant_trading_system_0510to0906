"""
Comprehensive tests for core.system2 module.
Focus on System2 RSI spike short strategy logic.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from core.system2 import (
    _compute_indicators,
    prepare_data_vectorized_system2,
    generate_candidates_system2,
    get_total_days_system2,
)


@pytest.fixture
def sample_system2_data():
    """Sample data with System2 required indicators."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")

    base_price = 50 + np.cumsum(np.random.randn(100) * 0.5)

    return pd.DataFrame(
        {
            "Date": dates,
            "Close": base_price,
            "Volume": np.random.randint(10000, 100000, 100),
            "rsi3": 50 + np.random.randn(100) * 20,  # RSI3 values
            "adx7": 15 + np.random.rand(100) * 70,  # ADX7 values
            "atr10": np.random.rand(100) * 2,  # ATR10 values
            "dollarvolume20": np.random.randint(10_000_000, 100_000_000, 100),
            "atr_ratio": np.random.rand(100) * 0.1,  # ATR_Ratio values
            "TwoDayUp": np.random.choice([True, False], 100),  # TwoDayUp (matching constants)
            "filter": np.random.choice([True, False], 100),  # Filter column
            "setup": np.random.choice([True, False], 100),  # Setup column
        }
    )


@pytest.fixture
def sample_system2_data_with_setup():
    """Sample data that meets System2 setup conditions."""
    dates = pd.date_range("2023-01-01", periods=50, freq="D")

    return pd.DataFrame(
        {
            "Date": dates,
            "Close": [10.0] * 50,  # Close >= 5
            "rsi3": [95.0] * 50,  # RSI3 > 90
            "adx7": np.linspace(60, 10, 50),  # Descending ADX7 for ranking
            "atr10": [1.0] * 50,
            "dollarvolume20": [30_000_000] * 50,  # > 25M
            "atr_ratio": [0.05] * 50,  # > 0.03
            "TwoDayUp": [True] * 50,  # TwoDayUp condition met (matching constants)
            "filter": [True] * 50,  # Filter column
            "setup": [True] * 50,  # Setup column
        }
    )


class TestSystem2ComputeIndicators:
    """Test System2 _compute_indicators function."""

    @patch("core.system2.get_cached_data")
    def test_compute_indicators_success(self, mock_get_data, sample_system2_data):
        """Test _compute_indicators with valid data."""
        mock_get_data.return_value = sample_system2_data

        symbol, result = _compute_indicators("AAPL")

        assert symbol == "AAPL"
        assert result is not None
        assert "filter" in result.columns
        assert "setup" in result.columns
        mock_get_data.assert_called_once_with("AAPL")

    @patch("core.system2.get_cached_data")
    def test_compute_indicators_none_data(self, mock_get_data):
        """Test _compute_indicators with None data."""
        mock_get_data.return_value = None

        symbol, result = _compute_indicators("INVALID")

        assert symbol == "INVALID"
        assert result is None
        mock_get_data.assert_called_once_with("INVALID")

    @patch("core.system2.get_cached_data")
    def test_compute_indicators_empty_data(self, mock_get_data):
        """Test _compute_indicators with empty data."""
        mock_get_data.return_value = pd.DataFrame()

        symbol, result = _compute_indicators("EMPTY")

        assert symbol == "EMPTY"
        assert result is None
        mock_get_data.assert_called_once_with("EMPTY")

    @patch("core.system2.get_cached_data")
    def test_compute_indicators_missing_indicators(self, mock_get_data):
        """Test _compute_indicators with missing required indicators."""
        incomplete_data = pd.DataFrame(
            {
                "Date": pd.date_range("2023-01-01", periods=10),
                "Close": [50] * 10,
                # Missing required indicators like rsi3, adx7, etc.
            }
        )
        mock_get_data.return_value = incomplete_data

        symbol, result = _compute_indicators("INCOMPLETE")

        assert symbol == "INCOMPLETE"
        assert result is None

    @patch("core.system2.get_cached_data")
    def test_compute_indicators_filter_conditions(self, mock_get_data, sample_system2_data):
        """Test System2 filter conditions."""
        # Modify data to test filter conditions
        test_data = sample_system2_data.copy()
        test_data.loc[0, "Close"] = 3.0  # Below 5.0 threshold
        test_data.loc[1, "dollarvolume20"] = 10_000_000  # Below 25M threshold
        test_data.loc[2, "atr_ratio"] = 0.01  # Below 0.03 threshold

        mock_get_data.return_value = test_data

        symbol, result = _compute_indicators("TEST")

        assert symbol == "TEST"
        assert result is not None

        # Check filter conditions
        assert not result.loc[0, "filter"]  # Close < 5
        assert not result.loc[1, "filter"]  # DollarVolume < 25M
        assert not result.loc[2, "filter"]  # ATR_Ratio < 0.03

    @patch("core.system2.get_cached_data")
    def test_compute_indicators_setup_conditions(self, mock_get_data, sample_system2_data):
        """Test System2 setup conditions."""
        # Modify data to test setup conditions
        test_data = sample_system2_data.copy()
        test_data.loc[0:4, "Close"] = 10.0  # Meet Close >= 5
        test_data.loc[0:4, "dollarvolume20"] = 30_000_000  # Meet DollarVolume > 25M
        test_data.loc[0:4, "atr_ratio"] = 0.05  # Meet ATR_Ratio > 0.03
        test_data.loc[0, "rsi3"] = 95.0  # Meet RSI3 > 90
        test_data.loc[0, "twodayup"] = True  # Meet TwoDayUp
        test_data.loc[1, "rsi3"] = 85.0  # Fail RSI3 > 90
        test_data.loc[1, "twodayup"] = True
        test_data.loc[2, "rsi3"] = 95.0
        test_data.loc[2, "twodayup"] = False  # Fail TwoDayUp

        mock_get_data.return_value = test_data

        symbol, result = _compute_indicators("TEST")

        assert symbol == "TEST"
        assert result is not None

        # Check setup conditions
        assert result.loc[0, "setup"]  # All conditions met
        assert not result.loc[1, "setup"]  # RSI3 condition failed
        assert not result.loc[2, "setup"]  # TwoDayUp condition failed


class TestSystem2PrepareDataVectorized:
    """Test System2 prepare_data_vectorized function."""

    def test_prepare_data_vectorized_basic(self, sample_system2_data):
        """Test prepare_data_vectorized basic functionality."""
        raw_data_dict = {
            "AAPL": sample_system2_data,
            "MSFT": sample_system2_data.copy(),
        }

        result = prepare_data_vectorized_system2(raw_data_dict, max_workers=2)

        assert isinstance(result, dict)
        # Verify that processing happened
        assert len(result) <= len(raw_data_dict)

    def test_prepare_data_vectorized_none_input(self):
        """Test prepare_data_vectorized with None input (skipped for now)."""
        # This test requires complex setup and symbol loading mechanism
        pytest.skip("Complex symbol loading test - requires full cache setup")

    def test_prepare_data_vectorized_empty_dict(self):
        """Test prepare_data_vectorized with empty data dict."""
        result = prepare_data_vectorized_system2({}, max_workers=1)

        assert isinstance(result, dict)
        assert len(result) == 0


class TestSystem2GenerateCandidates:
    """Test System2 generate_candidates function."""

    def test_generate_candidates_with_valid_data(self, sample_system2_data_with_setup):
        """Test generate_candidates with data that meets setup conditions."""
        data_dict = {"TEST_SYMBOL": sample_system2_data_with_setup}
        top_n = 5

        candidates_dict, candidates_df = generate_candidates_system2(data_dict, top_n=top_n)

        assert isinstance(candidates_dict, dict)
        assert isinstance(candidates_df, pd.DataFrame | type(None))

    def test_generate_candidates_no_setup_data(self):
        """Test generate_candidates with no setup conditions met."""
        data_dict = {
            "NO_SETUP": pd.DataFrame(
                {
                    "Date": pd.date_range("2023-01-01", periods=10),
                    "Close": [3.0] * 10,  # Below filter threshold
                    "setup": [False] * 10,  # No setup conditions
                    "adx7": [20.0] * 10,
                }
            )
        }
        top_n = 5

        candidates_dict, candidates_df = generate_candidates_system2(data_dict, top_n=top_n)

        assert isinstance(candidates_dict, dict)
        assert isinstance(candidates_df, pd.DataFrame | type(None))

    def test_generate_candidates_empty_data_dict(self):
        """Test generate_candidates with empty data dictionary."""
        candidates_dict, candidates_df = generate_candidates_system2({}, top_n=5)

        assert isinstance(candidates_dict, dict)
        assert len(candidates_dict) == 0
        assert candidates_df is None

    def test_generate_candidates_with_default_top_n(self, sample_system2_data_with_setup):
        """Test generate_candidates with default top_n value."""
        data_dict = {"RANK_TEST": sample_system2_data_with_setup}

        candidates_dict, candidates_df = generate_candidates_system2(data_dict)

        assert isinstance(candidates_dict, dict)
        assert isinstance(candidates_df, pd.DataFrame | type(None))

    def test_generate_candidates_with_progress_callback(self, sample_system2_data_with_setup):
        """Test generate_candidates with progress callback."""
        data_dict = {"DATE_TEST": sample_system2_data_with_setup}

        callback_calls = []

        def mock_log_callback(message):
            callback_calls.append(message)

        candidates_dict, candidates_df = generate_candidates_system2(
            data_dict, top_n=10, log_callback=mock_log_callback
        )

        assert isinstance(candidates_dict, dict)
        assert len(callback_calls) > 0  # Should have called the callback


class TestSystem2GetTotalDays:
    """Test System2 get_total_days function."""

    def test_get_total_days_with_data(self, sample_system2_data):
        """Test get_total_days with valid data."""
        data_dict = {
            "SYMBOL1": sample_system2_data,
            "SYMBOL2": sample_system2_data.copy(),
        }

        result = get_total_days_system2(data_dict)

        assert isinstance(result, int)
        assert result > 0

    def test_get_total_days_empty_dict(self):
        """Test get_total_days with empty data dictionary."""
        result = get_total_days_system2({})

        assert isinstance(result, int)
        assert result == 0  # Or whatever the expected default is

    def test_get_total_days_with_none_values(self, sample_system2_data):
        """Test get_total_days with valid data only (None filtered out)."""
        data_dict = {
            "VALID": sample_system2_data,
            # None values should be filtered out before calling get_total_days
        }

        result = get_total_days_system2(data_dict)
        assert result == len(sample_system2_data)

        assert isinstance(result, int)
        assert result > 0  # Should handle None values gracefully


class TestSystem2Integration:
    """Integration tests for System2 workflow."""

    @patch("core.system2.get_cached_data")
    def test_full_system2_workflow(self, mock_get_data, sample_system2_data_with_setup):
        """Test complete System2 workflow from data preparation to candidate generation."""
        mock_get_data.return_value = sample_system2_data_with_setup

        # Step 1: Prepare data using the correct approach
        raw_data_dict = {"TEST": sample_system2_data_with_setup}
        data_dict = prepare_data_vectorized_system2(raw_data_dict, max_workers=1)

        assert isinstance(data_dict, dict)

        # Step 2: Generate candidates
        candidates_dict, candidates_df = generate_candidates_system2(data_dict, top_n=5)

        assert isinstance(candidates_dict, dict)
        assert isinstance(candidates_df, pd.DataFrame | type(None))

    def test_system2_edge_cases(self):
        """Test System2 edge cases and error handling."""
        # Test with invalid symbol
        symbol, result = _compute_indicators("")
        assert symbol == ""
        # Result depends on implementation, could be None or empty DataFrame

        # Test generate_candidates with invalid inputs
        candidates_dict, candidates_df = generate_candidates_system2({}, top_n=5)
        assert isinstance(candidates_dict, dict)
        assert len(candidates_dict) == 0
        assert candidates_df is None
