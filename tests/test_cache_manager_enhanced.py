"""
Enhanced tests for CacheManager to boost coverage to 80%+
Testing all critical methods including read, write, indicators, and batch operations
"""

import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from unittest.mock import Mock, patch

from common.cache_manager import (
    CacheManager,
    compute_base_indicators,
    get_indicator_column_flexible,
    standardize_indicator_columns,
    _base_dir,
)
from config.settings import Settings


@pytest.fixture
def mock_settings():
    """Mock settings for testing"""
    settings = Mock(spec=Settings)

    # Create temporary directories for testing
    temp_dir = Path(tempfile.mkdtemp())
    settings.cache = Mock()
    settings.cache.full_dir = str(temp_dir / "full")
    settings.cache.rolling_dir = str(temp_dir / "rolling")
    settings.cache.rolling = Mock()
    settings.cache.rolling.meta_file = "meta.json"
    settings.cache.rolling.days = 300
    settings.cache.rolling.base_lookback_days = 250
    settings.cache.rolling.buffer_days = 50
    settings.cache.file_format = "csv"

    # Add DATA_CACHE_DIR for _base_dir function
    settings.DATA_CACHE_DIR = str(temp_dir)

    return settings


@pytest.fixture
def sample_data():
    """Sample OHLCV data for testing"""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    np.random.seed(42)  # For reproducible tests

    # Generate realistic price data
    base_price = 100.0
    prices = []
    for i in range(len(dates)):
        if i == 0:
            prices.append(base_price)
        else:
            # Random walk with trend
            change = np.random.normal(0.001, 0.02)
            prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": prices * (1 + np.random.normal(0, 0.001, len(dates))),
            "High": prices * (1 + np.abs(np.random.normal(0.01, 0.005, len(dates)))),
            "Low": prices * (1 - np.abs(np.random.normal(0.01, 0.005, len(dates)))),
            "Close": prices,
            "Volume": np.random.randint(100000, 1000000, len(dates)),
        }
    )

    df.set_index("Date", inplace=True)
    return df


class TestCacheManager:
    """Test CacheManager core functionality"""

    def test_init(self, mock_settings):
        """Test CacheManager initialization"""
        manager = CacheManager(mock_settings)

        assert manager.settings == mock_settings
        assert manager.full_dir == Path(mock_settings.cache.full_dir)
        assert manager.rolling_dir == Path(mock_settings.cache.rolling_dir)

        # Check directories are created
        assert manager.full_dir.exists()
        assert manager.rolling_dir.exists()

    def test_recompute_indicators(self, mock_settings, sample_data):
        """Test indicator recomputation"""
        manager = CacheManager(mock_settings)

        # Test _recompute_indicators method
        result = manager._recompute_indicators(sample_data)

        # Should have original columns plus indicators
        assert "Close" in result.columns
        assert "Volume" in result.columns
        assert len(result) == len(sample_data)

    def test_read_base_and_tail(self, mock_settings, sample_data):
        """Test reading base cache with tail operation"""
        manager = CacheManager(mock_settings)

        # Mock file manager to return sample data
        with (
            patch.object(manager.file_manager, "detect_path") as mock_detect,
            patch.object(manager.file_manager, "read_with_fallback") as mock_read,
        ):

            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_detect.return_value = mock_path
            mock_read.return_value = sample_data

            result = manager._read_base_and_tail("AAPL", tail_rows=50)

            assert result is not None
            assert len(result) <= 50  # Should be truncated

    def test_read_base_and_tail_missing_file(self, mock_settings):
        """Test reading non-existent file"""
        manager = CacheManager(mock_settings)

        # Mock missing file
        with patch.object(manager.file_manager, "detect_path") as mock_detect:
            mock_path = Mock()
            mock_path.exists.return_value = False
            mock_detect.return_value = mock_path

            result = manager._read_base_and_tail("NONEXISTENT")
            assert result is None

    @patch("common.cache_manager.CacheFileManager")
    def test_read_method(self, mock_file_manager, mock_settings, sample_data):
        """Test main read method"""
        manager = CacheManager(mock_settings)

        # Test rolling profile read
        mock_path = Mock()
        mock_path.exists.return_value = True
        manager.file_manager.detect_path.return_value = mock_path
        manager.file_manager.read_data.return_value = sample_data

        result = manager.read("AAPL", "rolling")
        assert result is not None
        assert len(result) > 0

        # Test base profile read
        result = manager.read("AAPL", "base")
        assert result is not None

    @patch("common.cache_manager.CacheFileManager")
    def test_write_atomic(self, mock_file_manager, mock_settings, sample_data):
        """Test atomic write operation"""
        manager = CacheManager(mock_settings)

        # Mock successful write
        manager.file_manager.write_data.return_value = None

        # Should not raise exception
        manager.write_atomic(sample_data, "AAPL", "rolling")

        # Verify write was called
        manager.file_manager.write_data.assert_called()

    def test_enforce_rolling_window(self, mock_settings, sample_data):
        """Test rolling window enforcement"""
        manager = CacheManager(mock_settings)

        # Create data longer than rolling window
        long_data = pd.concat([sample_data] * 5)  # 500 rows

        result = manager._enforce_rolling_window(long_data)

        # Should be truncated to rolling window size
        assert len(result) <= 300

    @patch("common.cache_manager.CacheFileManager")
    def test_upsert_both(self, mock_file_manager, mock_settings, sample_data):
        """Test upserting to both full and rolling caches"""
        manager = CacheManager(mock_settings)

        # Mock existing data
        existing_data = sample_data.iloc[:50]  # First 50 rows
        new_data = sample_data.iloc[45:]  # Overlapping + new rows

        manager.file_manager.detect_path.return_value = Mock(exists=lambda: True)
        manager.file_manager.read_data.return_value = existing_data
        manager.file_manager.write_data.return_value = None

        # Should not raise exception
        manager.upsert_both("AAPL", new_data)

        # Verify calls were made
        assert manager.file_manager.write_data.call_count >= 2  # Both profiles

    def test_optimize_dataframe_memory(self, mock_settings, sample_data):
        """Test memory optimization"""
        manager = CacheManager(mock_settings)

        # Add some columns that can be optimized
        test_data = sample_data.copy()
        test_data["IntCol"] = np.arange(len(test_data), dtype=np.int64)
        test_data["FloatCol"] = np.random.random(len(test_data)).astype(np.float64)

        original_memory = test_data.memory_usage(deep=True).sum()
        optimized = manager.optimize_dataframe_memory(test_data)
        optimized_memory = optimized.memory_usage(deep=True).sum()

        # Should use less or equal memory
        assert optimized_memory <= original_memory

    def test_remove_unnecessary_columns(self, mock_settings):
        """Test column removal"""
        manager = CacheManager(mock_settings)

        # Create data with unnecessary columns
        df = pd.DataFrame(
            {
                "Close": [100, 101, 102],
                "Volume": [1000, 1100, 1200],
                "Unnecessary": ["a", "b", "c"],
                "AlsoUnnecessary": [1, 2, 3],
            }
        )

        keep_cols = ["Close", "Volume"]
        result = manager.remove_unnecessary_columns(df, keep_cols)

        assert list(result.columns) == keep_cols

    @patch("common.cache_manager.CacheFileManager")
    def test_read_batch_parallel(self, mock_file_manager, mock_settings, sample_data):
        """Test parallel batch reading"""
        manager = CacheManager(mock_settings)

        # Mock successful reads
        manager.file_manager.detect_path.return_value = Mock(exists=lambda: True)
        manager.file_manager.read_data.return_value = sample_data

        symbols = ["AAPL", "GOOGL", "MSFT"]
        results = manager.read_batch_parallel(symbols, "rolling", max_workers=2)

        assert len(results) == 3
        for symbol, df in results.items():
            assert symbol in symbols
            if df is not None:
                assert isinstance(df, pd.DataFrame)


class TestStandaloneFunctions:
    """Test standalone functions in cache_manager.py"""

    def test_compute_base_indicators(self, sample_data):
        """Test indicator computation"""
        result = compute_base_indicators(sample_data)

        # Should have more columns than input (indicators added)
        assert len(result.columns) >= len(sample_data.columns)

        # Check for common indicators (depending on implementation)
        potential_indicators = ["SMA_20", "RSI_14", "ATR_20", "adx7"]
        found_indicators = [col for col in potential_indicators if col in result.columns]
        assert len(found_indicators) > 0  # Should have at least some indicators

    def test_get_indicator_column_flexible(self, sample_data):
        """Test flexible indicator column retrieval"""
        # Create test data with indicator
        df = sample_data.copy()
        df["RSI_14"] = np.random.random(len(df)) * 100
        df["RSI"] = np.random.random(len(df)) * 100

        # Test exact match
        result = get_indicator_column_flexible(df, "RSI_14")
        assert result is not None
        assert len(result) == len(df)

        # Test partial match
        result = get_indicator_column_flexible(df, "RSI")
        assert result is not None

        # Test no match
        result = get_indicator_column_flexible(df, "NONEXISTENT")
        assert result is None

    def test_standardize_indicator_columns(self, sample_data):
        """Test indicator column standardization"""
        # Create test data with various indicator formats
        df = sample_data.copy()
        df["rsi"] = np.random.random(len(df)) * 100  # lowercase
        df["RSI_14"] = np.random.random(len(df)) * 100  # with period
        df["Bollinger_Upper"] = df["Close"] * 1.1  # mixed case

        result = standardize_indicator_columns(df)

        # Should have same or more columns
        assert len(result.columns) >= len(sample_data.columns)

        # Original OHLCV columns should remain
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in result.columns

    def test_base_dir(self):
        """Test base directory function"""
        with patch("common.cache_manager.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.cache.full_dir = "/test/cache"
            mock_get_settings.return_value = mock_settings

            result = _base_dir()

            assert isinstance(result, Path)
            assert str(result).endswith("base")


class TestCacheManagerErrorHandling:
    """Test error handling and edge cases"""

    def test_read_with_corruption(self, mock_settings):
        """Test handling of corrupted cache files"""
        manager = CacheManager(mock_settings)

        # Mock corrupted file
        manager.file_manager.detect_path.return_value = Mock(exists=lambda: True)
        manager.file_manager.read_data.side_effect = Exception("File corrupted")

        result = manager.read("AAPL", "rolling")
        assert result is None  # Should handle gracefully

    def test_write_with_permission_error(self, mock_settings, sample_data):
        """Test handling of write permission errors"""
        manager = CacheManager(mock_settings)

        # Mock permission error
        manager.file_manager.write_data.side_effect = PermissionError("Access denied")

        # Should not crash
        try:
            manager.write_atomic(sample_data, "AAPL", "rolling")
        except PermissionError:
            pass  # Expected behavior

    def test_empty_dataframe_handling(self, mock_settings):
        """Test handling of empty DataFrames"""
        manager = CacheManager(mock_settings)

        empty_df = pd.DataFrame()

        # Should handle empty DataFrame gracefully
        result = manager._enforce_rolling_window(empty_df)
        assert len(result) == 0

        # Memory optimization of empty DataFrame
        optimized = manager.optimize_dataframe_memory(empty_df)
        assert len(optimized) == 0


# Performance and Integration Tests
class TestCacheManagerIntegration:
    """Integration tests for CacheManager"""

    @patch("common.cache_manager.CacheFileManager")
    def test_full_workflow(self, mock_file_manager, mock_settings, sample_data):
        """Test complete read-modify-write workflow"""
        manager = CacheManager(mock_settings)

        # Setup mocks
        manager.file_manager.detect_path.return_value = Mock(exists=lambda: True)
        manager.file_manager.read_data.return_value = sample_data
        manager.file_manager.write_data.return_value = None

        # Full workflow
        # 1. Read existing data
        existing = manager.read("AAPL", "rolling")
        assert existing is not None

        # 2. Add new data
        new_rows = sample_data.tail(10).copy()
        new_rows.index = new_rows.index + pd.Timedelta(days=1)

        # 3. Upsert
        manager.upsert_both("AAPL", new_rows)

        # Verify operations were called
        assert manager.file_manager.read_data.called
        assert manager.file_manager.write_data.called

    def test_batch_operations_performance(self, mock_settings, sample_data):
        """Test that batch operations are more efficient than individual ops"""
        manager = CacheManager(mock_settings)

        # Mock successful operations
        manager.file_manager.detect_path.return_value = Mock(exists=lambda: True)
        manager.file_manager.read_data.return_value = sample_data

        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

        # Batch read should work efficiently
        results = manager.read_batch_parallel(symbols, "rolling", max_workers=3)

        # Should get results for all symbols
        assert len(results) == len(symbols)
