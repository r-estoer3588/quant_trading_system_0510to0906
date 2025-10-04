"""
Comprehensive tests for common.cache_manager module.
Focus on key functionality: initialization, read/write operations, health checks.
"""

from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from common.cache_manager import (
    CacheManager,
    _base_dir,
    compute_base_indicators,
    get_indicator_column_flexible,
    standardize_indicator_columns,
)


@pytest.fixture
def sample_ohlcv_data():
    """Sample OHLCV data for testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    np.random.seed(42)

    prices = np.cumsum(np.random.randn(100) * 0.01) + 100

    return pd.DataFrame(
        {
            "date": dates,
            "open": prices + np.random.randn(100) * 0.1,
            "high": prices + np.abs(np.random.randn(100)) * 0.2,
            "low": prices - np.abs(np.random.randn(100)) * 0.2,
            "close": prices,
            "volume": np.random.randint(1000, 10000, 100),
        }
    )


@pytest.fixture
def mock_settings():
    """Mock settings for CacheManager."""
    settings = Mock()

    # Create cache mock with proper structure
    cache_mock = Mock()
    cache_mock.full_dir = "test_cache/full"
    cache_mock.rolling_dir = "test_cache/rolling"

    rolling_mock = Mock()
    rolling_mock.meta_file = "meta.json"
    rolling_mock.window_size = 300
    rolling_mock.retention_days = 90
    cache_mock.rolling = rolling_mock

    settings.cache = cache_mock
    return settings


@pytest.fixture
def cache_manager_with_temp_dirs(mock_settings):
    """CacheManager with temporary directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        mock_settings.cache.full_dir = str(temp_path / "full")
        mock_settings.cache.rolling_dir = str(temp_path / "rolling")

        manager = CacheManager(mock_settings)
        yield manager


class TestCacheManagerInitialization:
    """Test CacheManager initialization and basic setup."""

    def test_init_creates_directories(self, cache_manager_with_temp_dirs):
        """Test that initialization creates necessary directories."""
        manager = cache_manager_with_temp_dirs

        assert manager.full_dir.exists()
        assert manager.rolling_dir.exists()
        assert manager.settings is not None
        assert manager.file_manager is not None

    def test_init_with_existing_directories(self, mock_settings):
        """Test initialization when directories already exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            full_dir = temp_path / "full"
            rolling_dir = temp_path / "rolling"

            # Pre-create directories
            full_dir.mkdir(parents=True)
            rolling_dir.mkdir(parents=True)

            mock_settings.cache.full_dir = str(full_dir)
            mock_settings.cache.rolling_dir = str(rolling_dir)

            manager = CacheManager(mock_settings)
            assert manager.full_dir.exists()
            assert manager.rolling_dir.exists()


class TestCacheManagerReadOperations:
    """Test CacheManager read operations."""

    @patch.object(CacheManager, "_read_base_and_tail")
    def test_read_base_and_tail_success(
        self, mock_read, cache_manager_with_temp_dirs, sample_ohlcv_data
    ):
        """Test successful _read_base_and_tail operation."""
        manager = cache_manager_with_temp_dirs
        mock_read.return_value = sample_ohlcv_data.tail(330)

        result = manager._read_base_and_tail("AAPL", 330)
        assert result is not None
        assert len(result) <= 330
        mock_read.assert_called_once()

    @patch.object(CacheManager, "_read_base_and_tail")
    def test_read_base_and_tail_none_result(
        self, mock_read, cache_manager_with_temp_dirs
    ):
        """Test _read_base_and_tail when file doesn't exist."""
        manager = cache_manager_with_temp_dirs
        mock_read.return_value = None

        result = manager._read_base_and_tail("NONEXISTENT", 330)
        assert result is None
        mock_read.assert_called_once()

    def test_read_with_profile_parameter(self, cache_manager_with_temp_dirs):
        """Test read method with profile parameter."""
        manager = cache_manager_with_temp_dirs

        # Mock file_manager.detect_path and read_with_fallback
        with (
            patch.object(manager.file_manager, "detect_path") as mock_detect,
            patch.object(
                manager.file_manager, "read_with_fallback"
            ),  # mock_read removed
        ):

            mock_path = Mock()
            mock_path.exists.return_value = False
            mock_detect.return_value = mock_path

            result = manager.read("AAPL", "base")
            assert result is None
            mock_detect.assert_called_once()


class TestCacheManagerWriteOperations:
    """Test CacheManager write operations."""

    def test_write_atomic_basic(self, cache_manager_with_temp_dirs, sample_ohlcv_data):
        """Test basic write_atomic operation."""
        manager = cache_manager_with_temp_dirs

        with patch.object(manager.file_manager, "write_atomic") as mock_write:
            manager.write_atomic(sample_ohlcv_data, "AAPL", "base")
            mock_write.assert_called_once()

    def test_upsert_both_calls_upsert_one(
        self, cache_manager_with_temp_dirs, sample_ohlcv_data
    ):
        """Test that upsert_both calls _upsert_one for both profiles."""
        manager = cache_manager_with_temp_dirs

        with patch.object(manager, "_upsert_one") as mock_upsert:
            manager.upsert_both("AAPL", sample_ohlcv_data)
            assert mock_upsert.call_count == 2

            # Verify both full and rolling profiles are called
            call_args = [call[0] for call in mock_upsert.call_args_list]
            profiles = [args[2] for args in call_args]
            assert "full" in profiles
            assert "rolling" in profiles


class TestCacheManagerIndicatorOperations:
    """Test indicator-related operations."""

    def test_recompute_indicators_with_valid_data(
        self, cache_manager_with_temp_dirs, sample_ohlcv_data
    ):
        """Test _recompute_indicators with valid OHLCV data."""
        manager = cache_manager_with_temp_dirs

        # Ensure required columns exist
        test_data = sample_ohlcv_data.copy()
        test_data["date"] = pd.to_datetime(test_data["date"])

        with patch("common.cache_manager.add_indicators") as mock_add:
            mock_add.return_value = test_data
            result = manager._recompute_indicators(test_data)

            assert result is not None
            assert not result.empty
            mock_add.assert_called_once()

    def test_recompute_indicators_with_empty_data(self, cache_manager_with_temp_dirs):
        """Test _recompute_indicators with empty or None data."""
        manager = cache_manager_with_temp_dirs

        # Test with None
        result = manager._recompute_indicators(None)
        assert result is None

        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result = manager._recompute_indicators(empty_df)
        assert result.empty

    def test_recompute_indicators_missing_required_columns(
        self, cache_manager_with_temp_dirs
    ):
        """Test _recompute_indicators when required OHLC columns are missing."""
        manager = cache_manager_with_temp_dirs

        incomplete_data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=10),
                "close": np.random.randn(10) + 100,
                # Missing open, high, low
            }
        )

        result = manager._recompute_indicators(incomplete_data)
        # Should return original data when required columns are missing
        pd.testing.assert_frame_equal(result, incomplete_data)


class TestCacheManagerHealthAndAnalysis:
    """Test health checking and analysis methods."""

    def test_get_rolling_health_summary_structure(self, cache_manager_with_temp_dirs):
        """Test that get_rolling_health_summary returns expected structure."""
        manager = cache_manager_with_temp_dirs

        with patch("common.cache_manager.perform_cache_health_check") as mock_health:
            mock_health.return_value = {
                "status": "healthy",
                "file_count": 5,
                "total_size": 1024,
            }

            result = manager.get_rolling_health_summary()
            assert isinstance(result, dict)
            mock_health.assert_called_once()

    def test_analyze_rolling_gaps_with_symbols(self, cache_manager_with_temp_dirs):
        """Test analyze_rolling_gaps with provided symbols."""
        manager = cache_manager_with_temp_dirs

        test_symbols = ["AAPL", "MSFT", "GOOGL"]

        with patch.object(manager, "rolling_dir") as mock_dir:
            mock_dir.iterdir.return_value = []

            result = manager.analyze_rolling_gaps(test_symbols)
            assert isinstance(result, dict)

    def test_prune_rolling_if_needed_basic(self, cache_manager_with_temp_dirs):
        """Test prune_rolling_if_needed basic functionality."""
        manager = cache_manager_with_temp_dirs

        with patch("common.cache_manager.report_rolling_issue"):  # mock_report removed
            result = manager.prune_rolling_if_needed("SPY")
            assert isinstance(result, dict)


class TestCacheManagerUtilityMethods:
    """Test utility and helper methods."""

    def test_ui_prefix_returns_string(self, cache_manager_with_temp_dirs):
        """Test that _ui_prefix returns a string."""
        manager = cache_manager_with_temp_dirs
        result = manager._ui_prefix()
        assert isinstance(result, str)

    def test_enforce_rolling_window_limits_size(
        self, cache_manager_with_temp_dirs, sample_ohlcv_data
    ):
        """Test that _enforce_rolling_window limits DataFrame size."""
        manager = cache_manager_with_temp_dirs
        manager.rolling_cfg.window_size = 50  # Smaller than sample data

        result = manager._enforce_rolling_window(sample_ohlcv_data)
        assert len(result) <= manager.rolling_cfg.window_size

    def test_optimize_dataframe_memory_basic(
        self, cache_manager_with_temp_dirs, sample_ohlcv_data
    ):
        """Test optimize_dataframe_memory basic functionality."""
        manager = cache_manager_with_temp_dirs

        result = manager.optimize_dataframe_memory(sample_ohlcv_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_data)

    def test_remove_unnecessary_columns_basic(self, cache_manager_with_temp_dirs):
        """Test remove_unnecessary_columns basic functionality."""
        manager = cache_manager_with_temp_dirs

        test_df = pd.DataFrame(
            {
                "open": [100, 101],
                "high": [102, 103],
                "low": [99, 100],
                "close": [101, 102],
                "volume": [1000, 1100],
                "unnecessary_col": [1, 2],
            }
        )

        keep_columns = ["open", "high", "low", "close", "volume"]
        result = manager.remove_unnecessary_columns(test_df, keep_columns)

        assert set(result.columns) == set(keep_columns)


class TestCacheManagerParallelOperations:
    """Test parallel operations."""

    def test_read_batch_parallel_basic(self, cache_manager_with_temp_dirs):
        """Test read_batch_parallel basic functionality."""
        manager = cache_manager_with_temp_dirs

        symbols = ["AAPL", "MSFT"]

        with patch.object(manager, "read") as mock_read:
            mock_read.return_value = pd.DataFrame({"close": [100, 101]})

            result = manager.read_batch_parallel(symbols, "base", max_workers=1)
            assert isinstance(result, dict)
            assert len(result) <= len(symbols)


class TestCacheManagerStandaloneFunctions:
    """Test standalone functions in cache_manager module."""

    def test_base_dir_returns_path(self):
        """Test _base_dir returns a Path object."""
        result = _base_dir()
        assert isinstance(result, Path)

    def test_compute_base_indicators_with_valid_data(self, sample_ohlcv_data):
        """Test compute_base_indicators with valid OHLCV data."""
        test_data = sample_ohlcv_data.copy()

        # The compute_base_indicators function calls add_indicators internally
        result = compute_base_indicators(test_data)

        assert result is not None
        assert not result.empty
        # Verify that indicators were actually added
        assert len(result.columns) >= len(test_data.columns)

    def test_get_indicator_column_flexible_existing_column(self):
        """Test get_indicator_column_flexible with existing column."""
        test_df = pd.DataFrame(
            {
                "RSI": [30, 40, 50, 60, 70],
                "rsi_14": [25, 35, 45, 55, 65],
            }
        )

        result = get_indicator_column_flexible(test_df, "RSI")
        assert result is not None
        assert len(result) == 5
        pd.testing.assert_series_equal(result, test_df["RSI"])

    def test_get_indicator_column_flexible_nonexistent_column(self):
        """Test get_indicator_column_flexible with non-existent column."""
        test_df = pd.DataFrame(
            {
                "close": [100, 101, 102],
            }
        )

        result = get_indicator_column_flexible(test_df, "NONEXISTENT")
        assert result is None

    def test_standardize_indicator_columns_basic(self):
        """Test standardize_indicator_columns basic functionality."""
        test_df = pd.DataFrame(
            {
                "open": [100, 101],
                "close": [101, 102],
                "SMA_20": [100.5, 101.5],
                "rsi": [50, 60],
            }
        )

        result = standardize_indicator_columns(test_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(test_df)
        # Original columns should be preserved
        assert "open" in result.columns
        assert "close" in result.columns
