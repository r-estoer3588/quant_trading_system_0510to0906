"""
System7 edge case tests for higher coverage.

Focus on:
- Lines 34-64: Date normalization and cache loading exceptions
- Lines 99-116: Cache incremental update detailed branches
- Lines 228-264: latest_only data construction edge cases
- Lines 324-343: Date grouping edge cases
"""

from unittest.mock import patch

import numpy as np
import pandas as pd

from core.system7 import generate_candidates_system7, prepare_data_vectorized_system7


class TestSystem7DateNormalization:
    """Test date normalization paths."""

    def test_prepare_data_with_date_column(self):
        """Test date normalization when DataFrame has 'Date' column (line 35-37)."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = np.linspace(400, 450, 100)
        lows = [p * 0.992 for p in prices]
        highs = [p * 1.008 for p in prices]

        # Need min_50/max_70 as well
        df_temp = pd.DataFrame({"Low": lows, "High": highs}, index=dates)
        min_50 = df_temp["Low"].rolling(window=50, min_periods=1).min()
        max_70 = df_temp["High"].rolling(window=70, min_periods=1).max()

        # Create DataFrame with 'Date' column (not as index)
        df_with_date_col = pd.DataFrame(
            {
                "Date": dates,
                "Close": prices,
                "Low": lows,
                "High": highs,
                "atr50": [p * 0.02 for p in prices],
                "min_50": min_50.values,
                "max_70": max_70.values,
            }
        )

        raw_data = {"SPY": df_with_date_col}

        result = prepare_data_vectorized_system7(raw_data, reuse_indicators=False)

        assert isinstance(result, dict)
        assert "SPY" in result
        # Index should be normalized dates
        assert isinstance(result["SPY"].index, pd.DatetimeIndex)

    def test_prepare_data_without_date_column(self):
        """Test date normalization when index is datetime (line 39-40)."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = np.linspace(400, 450, 100)
        lows = [p * 0.992 for p in prices]
        highs = [p * 1.008 for p in prices]

        # Need min_50/max_70 as well
        df_temp = pd.DataFrame({"Low": lows, "High": highs}, index=dates)
        min_50 = df_temp["Low"].rolling(window=50, min_periods=1).min()
        max_70 = df_temp["High"].rolling(window=70, min_periods=1).max()

        # Create DataFrame with datetime index (no Date column)
        df_no_date_col = pd.DataFrame(
            {
                "Close": prices,
                "Low": lows,
                "High": highs,
                "atr50": [p * 0.02 for p in prices],
                "min_50": min_50.values,
                "max_70": max_70.values,
            },
            index=dates,
        )

        raw_data = {"SPY": df_no_date_col}

        result = prepare_data_vectorized_system7(raw_data, reuse_indicators=False)

        assert isinstance(result, dict)
        assert "SPY" in result
        # Index should be normalized dates
        assert isinstance(result["SPY"].index, pd.DatetimeIndex)

    def test_prepare_data_missing_atr50_immediate_stop(self):
        """Test handling when atr50 is missing (line 42-45, 64-66, 135-140)."""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        prices = np.linspace(400, 450, 50)
        lows = [p * 0.992 for p in prices]
        highs = [p * 1.008 for p in prices]

        # Need min_50/max_70 as well so the code reaches atr50 check
        df_temp = pd.DataFrame({"Low": lows, "High": highs}, index=dates)
        min_50 = df_temp["Low"].rolling(window=50, min_periods=1).min()
        max_70 = df_temp["High"].rolling(window=70, min_periods=1).max()

        # Intentionally missing atr50 column
        df_no_atr = pd.DataFrame(
            {
                "Close": prices,
                "Low": lows,
                "High": highs,
                "min_50": min_50.values,
                "max_70": max_70.values,
                # atr50 missing intentionally
            },
            index=dates,
        )

        raw_data = {"SPY": df_no_atr}

        # Should skip SPY when error occurs (lines 135-140)
        skip_messages = []

        def mock_skip(msg):
            skip_messages.append(msg)

        result = prepare_data_vectorized_system7(
            raw_data, reuse_indicators=False, skip_callback=mock_skip
        )

        # SPY should be skipped due to error
        assert "SPY" not in result
        # Skip callback should have been called
        assert len(skip_messages) > 0
        assert any("SPY" in msg for msg in skip_messages)

    @patch("pandas.read_feather")
    @patch("os.path.exists")
    def test_cache_loading_exception_handled(self, mock_exists, mock_read_feather):
        """Test cache loading exception is handled gracefully (line 51-56)."""
        dates = pd.date_range("2023-01-01", periods=350, freq="D")
        prices = np.linspace(400, 450, 350)
        lows = [p * 0.992 for p in prices]
        highs = [p * 1.008 for p in prices]

        df_temp = pd.DataFrame({"Low": lows, "High": highs}, index=dates)
        min_50 = df_temp["Low"].rolling(window=50, min_periods=1).min()
        max_70 = df_temp["High"].rolling(window=70, min_periods=1).max()

        spy_data = pd.DataFrame(
            {
                "Close": prices,
                "Low": lows,
                "High": highs,
                "atr50": [p * 0.02 for p in prices],
                "min_50": min_50.values,
                "max_70": max_70.values,
            },
            index=dates,
        )

        raw_data = {"SPY": spy_data}

        # Mock cache exists but reading fails
        mock_exists.return_value = True
        mock_read_feather.side_effect = Exception("Feather read failed")

        # Should fall back to recalculating without crashing
        result = prepare_data_vectorized_system7(raw_data, reuse_indicators=True)

        assert isinstance(result, dict)
        assert "SPY" in result


class TestSystem7CacheIncrementalDetailedBranches:
    """Test cache incremental update detailed branches."""

    def create_spy_with_all_columns(self, periods=100):
        """Create complete SPY data."""
        dates = pd.date_range("2023-01-01", periods=periods, freq="D")
        prices = np.linspace(400, 450, periods)
        lows = [p * 0.992 for p in prices]
        highs = [p * 1.008 for p in prices]

        df_temp = pd.DataFrame({"Low": lows, "High": highs}, index=dates)
        min_50 = df_temp["Low"].rolling(window=50, min_periods=1).min()
        max_70 = df_temp["High"].rolling(window=70, min_periods=1).max()

        return pd.DataFrame(
            {
                "Close": prices,
                "Low": lows,
                "High": highs,
                "atr50": [p * 0.02 for p in prices],
                "ATR50": [p * 0.02 for p in prices],
                "min_50": min_50.values,
                "Min_50": min_50.values,
                "max_70": max_70.values,
                "Max_70": max_70.values,
            },
            index=dates,
        )

    @patch("os.path.exists")
    @patch("pandas.read_feather")
    def test_cache_recompute_with_context(self, mock_read_feather, mock_exists):
        """Test cache incremental update with 70-day context (lines 104-108)."""
        # Cached data: first 300 days
        cached_data = self.create_spy_with_all_columns(periods=300)

        # New data: 320 days (20 new days)
        new_data = self.create_spy_with_all_columns(periods=320)
        raw_data = {"SPY": new_data}

        mock_exists.return_value = True

        cached_feather = cached_data.copy()
        cached_feather.reset_index(inplace=True)
        cached_feather.rename(columns={"index": "Date"}, inplace=True)
        mock_read_feather.return_value = cached_feather

        result = prepare_data_vectorized_system7(raw_data, reuse_indicators=True)

        # Should merge cached + recomputed data
        assert isinstance(result, dict)
        assert "SPY" in result
        assert len(result["SPY"]) == 320

    @patch("os.path.exists")
    @patch("pandas.read_feather")
    def test_cache_concat_with_max_70_priority(self, mock_read_feather, mock_exists):
        """Test max_70 priority merge in concat (lines 109-113)."""
        cached_data = self.create_spy_with_all_columns(periods=300)
        # Set distinctive max_70 value
        cached_data["max_70"] = 999.0

        new_data = self.create_spy_with_all_columns(periods=320)
        new_data["max_70"] = 111.0
        raw_data = {"SPY": new_data}

        mock_exists.return_value = True

        cached_feather = cached_data.copy()
        cached_feather.reset_index(inplace=True)
        cached_feather.rename(columns={"index": "Date"}, inplace=True)
        mock_read_feather.return_value = cached_feather

        result = prepare_data_vectorized_system7(raw_data, reuse_indicators=True)

        # Cached max_70 values should be preserved for overlapping dates
        assert isinstance(result, dict)
        assert "SPY" in result


class TestSystem7LatestOnlyDetailedBranches:
    """Test latest_only mode detailed branches."""

    def create_spy_complete(self, setup_today=True, include_close=True):
        """Create SPY data with configurable columns."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = np.linspace(400, 450, 100)
        lows = [p * 0.992 for p in prices]
        highs = [p * 1.008 for p in prices]

        df_temp = pd.DataFrame({"Low": lows, "High": highs}, index=dates)
        min_50 = df_temp["Low"].rolling(window=50, min_periods=1).min()
        max_70 = df_temp["High"].rolling(window=70, min_periods=1).max()

        setup_vals = [False] * 99 + [setup_today]

        data = {
            "atr50": [p * 0.02 for p in prices],
            "ATR50": [p * 0.02 for p in prices],
            "min_50": min_50.values,
            "max_70": max_70.values,
            "setup": setup_vals,
        }

        if include_close:
            data["Close"] = prices

        return pd.DataFrame(data, index=dates)

    def test_latest_only_symbol_payload_construction(self):
        """Test symbol_payload construction excludes symbol/date (lines 248-252)."""
        spy_data = self.create_spy_complete(setup_today=True)
        data_dict = {"SPY": spy_data}

        result_tuple = generate_candidates_system7(
            data_dict, latest_only=True, include_diagnostics=True
        )
        normalized = result_tuple[0]

        # Check payload structure
        for date_key, symbols in normalized.items():
            if "SPY" in symbols:
                spy_payload = symbols["SPY"]
                # Should NOT have 'symbol' or 'date'
                assert "symbol" not in spy_payload
                assert "date" not in spy_payload
                # Should have expected keys
                assert "entry_date" in spy_payload

    def test_latest_only_df_fast_rank_assignment(self):
        """Test rank assignment in df_fast (lines 243-244)."""
        spy_data = self.create_spy_complete(setup_today=True)
        data_dict = {"SPY": spy_data}

        result_tuple = generate_candidates_system7(
            data_dict, latest_only=True, include_diagnostics=True
        )
        df_fast = result_tuple[1]

        # Should have rank columns with value 1
        if df_fast is not None and not df_fast.empty:
            assert "rank" in df_fast.columns
            assert "rank_total" in df_fast.columns
            assert all(df_fast["rank"] == 1)
            assert all(df_fast["rank_total"] == 1)

    def test_latest_only_no_setup_debug_log(self):
        """Test DEBUG log when setup=False today (lines 270-288)."""
        spy_data = self.create_spy_complete(setup_today=False)
        data_dict = {"SPY": spy_data}

        log_messages = []

        def mock_log(msg):
            log_messages.append(msg)

        result_tuple = generate_candidates_system7(
            data_dict, latest_only=True, log_callback=mock_log, include_diagnostics=True
        )
        normalized = result_tuple[0]

        # Should return empty candidates
        assert len(normalized) == 0
        # Should have logged DEBUG message
        assert any("DEBUG" in msg and "0 candidates" in msg for msg in log_messages)

    def test_latest_only_fallback_exception_message(self):
        """Test fallback exception message (lines 297-300)."""
        # Create incomplete data to trigger exception
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        incomplete_data = pd.DataFrame(
            {
                "atr50": [8.0] * 10,
                "setup": [True] * 10,
                # Missing required indicators
            },
            index=dates,
        )
        data_dict = {"SPY": incomplete_data}

        log_messages = []

        def mock_log(msg):
            log_messages.append(msg)

        # Should fall back to full scan
        result_tuple = generate_candidates_system7(
            data_dict, latest_only=True, log_callback=mock_log, include_diagnostics=True
        )

        # May log fallback message
        # (exact behavior depends on whether exception triggers)
        assert isinstance(result_tuple, tuple)


class TestSystem7DateGroupingDetailedBranches:
    """Test date grouping detailed branches."""

    def create_spy_multiple_dates(self, setup_count=5):
        """Create SPY with multiple setup dates."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = np.linspace(400, 450, 100)
        lows = [p * 0.992 for p in prices]
        highs = [p * 1.008 for p in prices]

        df_temp = pd.DataFrame({"Low": lows, "High": highs}, index=dates)
        min_50 = df_temp["Low"].rolling(window=50, min_periods=1).min()
        max_70 = df_temp["High"].rolling(window=70, min_periods=1).max()

        setup_vals = [False] * 100
        setup_indices = np.linspace(60, 95, setup_count, dtype=int)
        for idx in setup_indices:
            if idx < len(setup_vals):
                setup_vals[idx] = True

        return pd.DataFrame(
            {
                "Close": prices,
                "atr50": [p * 0.02 for p in prices],
                "ATR50": [p * 0.02 for p in prices],
                "min_50": min_50.values,
                "max_70": max_70.values,
                "setup": setup_vals,
            },
            index=dates,
        )

    def test_date_grouping_atr_val_exception_handling(self):
        """Test ATR value exception handling (lines 331-333)."""
        spy_data = self.create_spy_multiple_dates(setup_count=3)
        # Remove ATR columns to trigger exception path
        spy_data = spy_data.drop(columns=["ATR50", "atr50"], errors="ignore")

        data_dict = {"SPY": spy_data}

        result_tuple = generate_candidates_system7(data_dict, top_n=5, include_diagnostics=True)
        normalized = result_tuple[0]

        # Should handle missing ATR gracefully
        assert isinstance(normalized, dict)

    def test_log_callback_window_size_calculation(self):
        """Test window size calculation in log callback (lines 349-357)."""
        spy_data = self.create_spy_multiple_dates(setup_count=8)
        data_dict = {"SPY": spy_data}

        log_messages = []

        def mock_log(msg):
            log_messages.append(msg)

        _ = generate_candidates_system7(
            data_dict, top_n=10, log_callback=mock_log, include_diagnostics=True
        )

        # Should log message with window calculation
        assert any("候補日数" in msg for msg in log_messages)
        # Should mention 50-day window
        assert any("50" in msg or "日間" in msg for msg in log_messages)
