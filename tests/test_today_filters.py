"""Tests for common.today_filters module."""

from __future__ import annotations

import pandas as pd
import pytest

from common.today_filters import (
    _calc_dollar_volume_from_series,
    _last_scalar,
    filter_system1,
    filter_system2,
    filter_system3,
    filter_system4,
    filter_system5,
    filter_system6,
)


class TestTodayFilters:
    """Test suite for today filters functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample market data for testing."""
        return pd.DataFrame(
            {
                "Date": pd.date_range("2023-01-01", periods=50),
                "Open": [100.0 + i * 0.1 for i in range(50)],
                "High": [102.0 + i * 0.1 for i in range(50)],
                "Low": [98.0 + i * 0.1 for i in range(50)],
                "Close": [101.0 + i * 0.1 for i in range(50)],
                "Volume": [10000 + i * 100 for i in range(50)],
                "Dollar_Volume": [1000000 + i * 10000 for i in range(50)],
            }
        )

    @pytest.fixture
    def sample_symbol_data(self, sample_data):
        """Create sample symbol data mapping."""
        # Create properly differentiated data for each symbol
        aapl_data = sample_data.copy()

        msft_data = sample_data.copy()
        # Only modify numeric columns, not Date column
        numeric_cols = msft_data.select_dtypes(include=["number"]).columns
        msft_data[numeric_cols] = msft_data[numeric_cols] * 1.1

        googl_data = sample_data.copy()
        googl_data[numeric_cols] = googl_data[numeric_cols] * 0.9

        spy_data = sample_data.copy()

        return {
            "AAPL": aapl_data,
            "MSFT": msft_data,
            "GOOGL": googl_data,
            "SPY": spy_data,
        }

    def test_last_scalar_valid_data(self):
        """Test _last_scalar with valid data."""
        series = pd.Series([1, 2, 3, 4, 5])
        result = _last_scalar(series)
        assert result == 5

    def test_last_scalar_empty_data(self):
        """Test _last_scalar with empty data."""
        empty_series = pd.Series(dtype=float)
        result = _last_scalar(empty_series)
        assert result is None

    def test_calc_dollar_volume_from_series(self):
        """Test dollar volume calculation."""
        close_series = pd.Series([100, 101, 102, 103, 104])
        volume_series = pd.Series([1000, 1100, 1200, 1300, 1400])

        result = _calc_dollar_volume_from_series(close_series, volume_series, window=3)
        assert isinstance(result, float | type(None))
        if result is not None:
            assert result > 0

    def test_filter_system1(self, sample_symbol_data):
        """Test system1 filtering logic."""
        symbols = list(sample_symbol_data.keys())
        stats = {}

        result = filter_system1(symbols, sample_symbol_data, stats)

        assert isinstance(result, list)
        assert len(stats) > 0
        assert "total" in stats
        # Check for actual stat keys returned by the function
        expected_keys = ["dv_pass", "price_pass", "total"]
        assert all(key in stats for key in expected_keys)

    def test_filter_system2(self, sample_symbol_data):
        """Test system2 filtering logic."""
        symbols = list(sample_symbol_data.keys())
        stats = {}

        result = filter_system2(symbols, sample_symbol_data, stats)

        assert isinstance(result, list)
        assert len(stats) > 0
        assert "total" in stats
        # Check for actual stat keys returned by the function
        expected_keys = ["atr_pass", "dv_pass", "price_pass", "total"]
        assert all(key in stats for key in expected_keys)

    def test_filter_system3(self, sample_symbol_data):
        """Test system3 filtering logic."""
        symbols = list(sample_symbol_data.keys())
        stats = {}

        result = filter_system3(symbols, sample_symbol_data, stats)

        assert isinstance(result, list)
        assert len(stats) > 0
        assert "total" in stats
        # Check for actual stat keys returned by the function
        expected_keys = ["atr_pass", "avgvol_pass", "low_pass", "total"]
        assert all(key in stats for key in expected_keys)

    def test_filter_system4(self, sample_symbol_data):
        """Test system4 filtering logic."""
        symbols = list(sample_symbol_data.keys())
        stats = {}

        result = filter_system4(symbols, sample_symbol_data, stats)

        assert isinstance(result, list)
        assert len(stats) > 0
        assert "total" in stats
        # Check for actual stat keys returned by the function
        expected_keys = ["dv_pass", "hv_pass", "total"]
        assert all(key in stats for key in expected_keys)

    def test_filter_system5(self, sample_symbol_data):
        """Test system5 filtering logic."""
        symbols = list(sample_symbol_data.keys())
        stats = {}

        result = filter_system5(symbols, sample_symbol_data, stats)

        assert isinstance(result, list)
        assert len(stats) > 0
        assert "total" in stats
        # Check for actual stat keys returned by the function
        expected_keys = ["atr_pass", "avgvol_pass", "dv_pass", "total"]
        assert all(key in stats for key in expected_keys)

    def test_filter_system6(self, sample_symbol_data):
        """Test system6 filtering logic."""
        symbols = list(sample_symbol_data.keys())
        stats = {}

        result = filter_system6(symbols, sample_symbol_data, stats)

        assert isinstance(result, list)
        assert len(stats) > 0
        assert "total" in stats
        # Check for actual stat keys returned by the function
        expected_keys = ["dv_pass", "low_pass", "total"]
        assert all(key in stats for key in expected_keys)

    def test_filter_with_empty_symbols(self):
        """Test filtering with empty symbol list."""
        for filter_func in [
            filter_system1,
            filter_system2,
            filter_system3,
            filter_system4,
            filter_system5,
            filter_system6,
        ]:
            result = filter_func([], {})
            assert isinstance(result, list)
            assert len(result) == 0

    def test_filter_with_missing_data(self, sample_symbol_data):
        """Test filtering when some symbols have missing data."""
        symbols = ["AAPL", "MISSING_SYMBOL"]
        data = {"AAPL": sample_symbol_data["AAPL"]}  # Missing data for MISSING_SYMBOL

        result = filter_system1(symbols, data)
        assert isinstance(result, list)
        # Should handle missing data gracefully

    def test_stats_tracking(self, sample_symbol_data):
        """Test that stats are properly tracked across filters."""
        symbols = list(sample_symbol_data.keys())

        for filter_func in [
            filter_system1,
            filter_system2,
            filter_system3,
            filter_system4,
            filter_system5,
            filter_system6,
        ]:
            stats = {}
            filter_func(symbols, sample_symbol_data, stats)

            # Check that stats were populated
            assert isinstance(stats, dict)
            assert len(stats) > 0

            # Check required stat keys
            assert "total" in stats
            # Each filter has different specific stats, but all should have total

            # Stats should be non-negative integers
            for _key, value in stats.items():
                assert isinstance(value, int)
                assert value >= 0

    @pytest.mark.parametrize(
        "system_filter",
        [
            filter_system1,
            filter_system2,
            filter_system3,
            filter_system4,
            filter_system5,
            filter_system6,
        ],
    )
    def test_all_filters_parametrized(self, system_filter, sample_symbol_data):
        """Test all filter functions with same basic requirements."""
        symbols = list(sample_symbol_data.keys())
        stats = {}

        result = system_filter(symbols, sample_symbol_data, stats)

        # Basic assertions that should apply to all filters
        assert isinstance(result, list)
        assert all(isinstance(symbol, str) for symbol in result)
        assert len(stats) > 0
        assert isinstance(stats, dict)

    def test_filter_edge_cases(self):
        """Test filters with edge case data."""
        # Create minimal data
        minimal_data = pd.DataFrame(
            {
                "Date": [pd.Timestamp("2023-01-01")],
                "Close": [100.0],
                "Volume": [1000],
                "High": [101.0],
                "Low": [99.0],
                "Open": [100.0],
            }
        )

        symbols = ["TEST"]
        data = {"TEST": minimal_data}

        # All filters should handle minimal data without crashing
        for filter_func in [
            filter_system1,
            filter_system2,
            filter_system3,
            filter_system4,
            filter_system5,
            filter_system6,
        ]:
            try:
                result = filter_func(symbols, data)
                assert isinstance(result, list)
            except Exception as e:
                pytest.fail(
                    f"Filter {filter_func.__name__} failed with minimal data: {e}"
                )
