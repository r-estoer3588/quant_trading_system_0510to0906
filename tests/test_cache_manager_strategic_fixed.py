"""
Strategic expansion tests for cache_manager.py - targeting 80% coverage
Building on 58% baseline by focusing on uncovered methods
"""

from __future__ import annotations

import pandas as pd
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path
import pytest

from common.cache_manager import (
    CacheManager,
    get_indicator_column_flexible,
    save_base_cache,
    load_base_cache,
    round_dataframe,
    make_csv_formatters,
    report_rolling_issue,
    _RollingIssueAggregator,
)
from common.testing import set_test_determinism


class TestCacheManagerUncoveredMethods:
    """Test uncovered CacheManager methods to boost coverage from 58% to 80%"""

    def setup_method(self):
        set_test_determinism()
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create mock settings with all necessary attributes
        self.mock_settings = Mock()
        self.mock_settings.cache = Mock()
        self.mock_settings.cache.full_dir = str(self.temp_dir / "full")
        self.mock_settings.cache.rolling_dir = str(self.temp_dir / "rolling")
        self.mock_settings.cache.rolling = Mock()
        self.mock_settings.cache.rolling.meta_file = "meta.json"
        self.mock_settings.cache.file_format = "csv"

        with patch("common.cache_manager.get_settings", return_value=self.mock_settings):
            self.manager = CacheManager(self.mock_settings)

    def teardown_method(self):
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_warn_once_functionality(self):
        """Test _warn_once method to prevent duplicate warnings"""
        with patch("common.cache_manager.logger.warning") as mock_warning:
            # First call should trigger warning
            self.manager._warn_once("AAPL", "test", "error", "Test message 1")
            assert mock_warning.call_count == 1

            # Second call with same parameters should not trigger warning
            self.manager._warn_once("AAPL", "test", "error", "Test message 1")
            assert mock_warning.call_count == 1  # Still 1, not 2

            # Different parameters should trigger new warning
            self.manager._warn_once("MSFT", "test", "error", "Test message 2")
            assert mock_warning.call_count == 2

    def test_recompute_indicators_comprehensive(self):
        """Test _recompute_indicators method for various data scenarios"""
        # Create test data with OHLC columns
        test_data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=10),
                "open": [100 + i for i in range(10)],
                "high": [105 + i for i in range(10)],
                "low": [95 + i for i in range(10)],
                "close": [102 + i for i in range(10)],
                "volume": [1000 * (i + 1) for i in range(10)],
            }
        )

        with patch("common.cache_manager.add_indicators") as mock_add_indicators:
            # Mock the add_indicators function to return enriched data
            mock_enriched = test_data.copy()
            mock_enriched["SMA25"] = [100] * 10  # Mock indicator
            mock_add_indicators.return_value = mock_enriched

            result = self.manager._recompute_indicators(test_data)
            assert result is not None
            assert len(result) == 10
            mock_add_indicators.assert_called_once()

    def test_detect_path_method(self):
        """Test _detect_path method for file format detection"""
        # Create temporary files with different extensions
        base_dir = self.temp_dir / "test_detect"
        base_dir.mkdir(parents=True, exist_ok=True)

        csv_file = base_dir / "AAPL.csv"
        parquet_file = base_dir / "MSFT.parquet"

        csv_file.touch()
        parquet_file.touch()

        # Test CSV detection
        detected_csv = self.manager._detect_path(base_dir, "AAPL")
        assert detected_csv == csv_file

        # Test parquet detection
        detected_parquet = self.manager._detect_path(base_dir, "MSFT")
        assert detected_parquet == parquet_file

        # Test default fallback
        detected_new = self.manager._detect_path(base_dir, "GOOGL")
        assert detected_new.suffix == ".csv"

    def test_read_with_fallback_method(self):
        """Test _read_with_fallback method for file reading with error handling"""
        # Create test CSV file
        test_file = self.temp_dir / "test_read.csv"
        test_data = pd.DataFrame({"date": ["2023-01-01", "2023-01-02"], "close": [100, 101]})
        test_data.to_csv(test_file, index=False)

        # Test successful reading
        result = self.manager._read_with_fallback(test_file, "TEST", "profile")
        assert result is not None
        assert len(result) == 2

        # Test non-existent file
        missing_file = self.temp_dir / "missing.csv"
        result = self.manager._read_with_fallback(missing_file, "MISSING", "profile")
        assert result is None


class TestUtilityFunctionsCoverage:
    """Test utility functions to improve overall coverage"""

    def test_round_dataframe_functionality(self):
        """Test round_dataframe with various column types and scenarios"""
        test_df = pd.DataFrame(
            {
                "open": [100.12345, 101.67890],
                "close": [100.56789, 101.23456],
                "volume": [1000.5, 2000.7],
                "rsi14": [45.6789, 67.1234],
                "roc200": [0.123456, 0.234567],
            }
        )

        # Test normal rounding
        rounded = round_dataframe(test_df, decimals=2)
        assert rounded is not None

        # Test with None input
        result = round_dataframe(None, decimals=2)
        assert result is None

        # Test with None decimals
        result = round_dataframe(test_df, decimals=None)
        assert result is test_df

    def test_make_csv_formatters(self):
        """Test make_csv_formatters function for CSV output formatting"""
        test_df = pd.DataFrame(
            {"open": [100.123, 101.456], "volume": [1000, 2000], "rsi14": [45.67, 67.12]}
        )

        # Test formatter creation
        formatters = make_csv_formatters(test_df)
        assert isinstance(formatters, dict)

        # Test with custom decimal point
        formatters = make_csv_formatters(test_df, dec_point=",")
        assert isinstance(formatters, dict)

    def test_get_indicator_column_flexible(self):
        """Test get_indicator_column_flexible function"""
        test_df = pd.DataFrame({"SMA25": [100, 101], "sma25": [100, 101], "other_col": [1, 2]})

        # Test column retrieval
        result = get_indicator_column_flexible(test_df, "sma25")
        assert result is not None
        assert len(result) == 2

    def test_rolling_issue_aggregator(self):
        """Test _RollingIssueAggregator for issue reporting"""
        # Test singleton pattern
        agg1 = _RollingIssueAggregator()
        agg2 = _RollingIssueAggregator()
        assert agg1 is agg2

        # Test issue reporting
        with patch("common.cache_manager.logger"):
            report_rolling_issue("test_category", "TEST_SYMBOL", "test message")
            # Should call logger methods


class TestCacheManagerAdvancedFeatures:
    """Test advanced CacheManager features and edge cases"""

    def setup_method(self):
        set_test_determinism()
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create comprehensive mock settings
        self.mock_settings = Mock()
        self.mock_settings.cache = Mock()
        self.mock_settings.cache.full_dir = str(self.temp_dir / "full")
        self.mock_settings.cache.rolling_dir = str(self.temp_dir / "rolling")
        self.mock_settings.cache.rolling = Mock()
        self.mock_settings.cache.rolling.meta_file = "meta.json"
        self.mock_settings.cache.file_format = "csv"
        self.mock_settings.cache.round_decimals = 4
        self.mock_settings.cache.csv = Mock()
        self.mock_settings.cache.csv.decimal_point = "."
        self.mock_settings.cache.csv.field_sep = ","

        with patch("common.cache_manager.get_settings", return_value=self.mock_settings):
            self.manager = CacheManager(self.mock_settings)

    def teardown_method(self):
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_manager_initialization(self):
        """Test CacheManager initialization and directory creation"""
        assert self.manager.settings == self.mock_settings
        assert self.manager.full_dir.exists()
        assert self.manager.rolling_dir.exists()
        assert self.manager.file_format == "csv"

    def test_ui_prefix_and_logging_setup(self):
        """Test UI prefix and logging configuration"""
        assert hasattr(self.manager, "_ui_prefix")
        assert self.manager._ui_prefix == "[CacheManager]"
        assert hasattr(self.manager, "_warned")

    def test_save_and_load_base_cache_integration(self):
        """Test save_base_cache and load_base_cache integration"""
        test_data = pd.DataFrame(
            {
                "Date": pd.date_range("2023-01-01", periods=5),
                "Open": [100, 101, 102, 103, 104],
                "Close": [101, 102, 103, 104, 105],
            }
        ).set_index("Date")

        with patch("common.cache_manager.get_settings", return_value=self.mock_settings):
            # Test save
            with patch("common.cache_manager.base_cache_path") as mock_path:
                mock_file_path = self.temp_dir / "TEST.csv"
                mock_path.return_value = mock_file_path

                save_base_cache("TEST", test_data, self.mock_settings)
                assert mock_path.called

        # Test load with error handling
        with patch("common.cache_manager.base_cache_path") as mock_path:
            mock_file_path = self.temp_dir / "MISSING.csv"  # Non-existent file
            mock_path.return_value = mock_file_path

            load_base_cache("MISSING")
            # Should handle missing file gracefully

    def test_error_handling_comprehensive(self):
        """Test comprehensive error handling scenarios"""
        # Test recompute_indicators with invalid data
        invalid_data = pd.DataFrame({"invalid_col": [1, 2, 3]})
        result = self.manager._recompute_indicators(invalid_data)
        assert result is invalid_data  # Should return original data

        # Test recompute_indicators with empty data
        empty_data = pd.DataFrame()
        result = self.manager._recompute_indicators(empty_data)
        assert result is empty_data

        # Test recompute_indicators with None input
        result = self.manager._recompute_indicators(None)
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__])
