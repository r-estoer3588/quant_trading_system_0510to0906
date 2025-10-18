"""
Test error handling and edge cases for System7.

This module tests error conditions, missing data, and edge cases
to improve coverage of error handling paths.
"""

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from core.system7 import generate_candidates_system7, prepare_data_vectorized_system7


@pytest.fixture(autouse=True)
def clean_system7_cache():
    """Remove System7 indicator cache before each test."""
    cache_path = "data_cache/indicators_system7_cache/SPY.feather"
    if os.path.exists(cache_path):
        os.remove(cache_path)
    yield
    if os.path.exists(cache_path):
        os.remove(cache_path)


class TestSystem7ErrorHandling:
    """Test error handling in System7."""

    def test_prepare_missing_spy_data(self):
        """Test prepare_data handles missing SPY data (line 33)."""
        # Empty dict - no SPY key
        raw_dict = {}

        result = prepare_data_vectorized_system7(raw_dict, reuse_indicators=False)

        # Should return empty dict
        assert result == {} or "SPY" not in result

    def test_prepare_spy_none(self):
        """Test prepare_data handles SPY=None (line 33-34)."""
        raw_dict = {"SPY": None}

        # Should raise ValueError or return empty
        try:
            result = prepare_data_vectorized_system7(raw_dict, reuse_indicators=False)
            # If it doesn't raise, should return empty or skip SPY
            assert result == {} or "SPY" not in result
        except ValueError as e:
            # Expected error
            assert "SPY data missing" in str(e)

    def test_prepare_missing_atr50(self):
        """Test prepare_data detects missing atr50 indicator (line 43-47)."""
        dates = pd.date_range("2025-01-01", periods=10, freq="D")

        # DataFrame without atr50 (required precomputed indicator)
        spy_df = pd.DataFrame(
            {
                "Close": np.linspace(400, 410, 10),
                "Low": np.linspace(398, 408, 10),
                "High": np.linspace(402, 412, 10),
                # Missing: "atr50"
            },
            index=dates,
        )

        raw_dict = {"SPY": spy_df}

        # Capture skip messages
        skip_messages = []

        def capture_skip(msg):
            skip_messages.append(msg)

        # Should call skip_callback with error about missing atr50
        result = prepare_data_vectorized_system7(raw_dict, reuse_indicators=False, skip_callback=capture_skip)

        # Should return empty dict (SPY skipped)
        assert result == {} or "SPY" not in result

        # Should have skip message about atr50
        assert len(skip_messages) > 0
        assert any("atr50" in msg for msg in skip_messages)

    @patch("core.system7.resolve_signal_entry_date")
    def test_generate_empty_prepared_dict(self, mock_resolve):
        """Test generate_candidates with empty prepared dict (line 197)."""
        mock_resolve.return_value = pd.Timestamp("2025-04-11")

        # Empty prepared dict
        data_dict = {}

        result_tuple = generate_candidates_system7(
            data_dict,
            latest_only=True,
            include_diagnostics=True,
        )

        normalized, df_fast, diagnostics = result_tuple

        # Should return empty results
        assert len(normalized) == 0
        assert df_fast is None

    @patch("core.system7.resolve_signal_entry_date")
    def test_generate_spy_none(self, mock_resolve):
        """Test generate_candidates with SPY=None (line 201)."""
        mock_resolve.return_value = pd.Timestamp("2025-04-11")

        # Dict with SPY=None
        data_dict = {"SPY": None}

        result_tuple = generate_candidates_system7(
            data_dict,
            latest_only=True,
            include_diagnostics=True,
        )

        normalized, df_fast, diagnostics = result_tuple

        # Should return empty results
        assert len(normalized) == 0
        assert df_fast is None

    @patch("core.system7.resolve_signal_entry_date")
    def test_generate_spy_empty_dataframe(self, mock_resolve):
        """Test generate_candidates with empty DataFrame (line 201)."""
        mock_resolve.return_value = pd.Timestamp("2025-04-11")

        # Empty DataFrame
        empty_df = pd.DataFrame()
        data_dict = {"SPY": empty_df}

        result_tuple = generate_candidates_system7(
            data_dict,
            latest_only=True,
            include_diagnostics=True,
        )

        normalized, df_fast, diagnostics = result_tuple

        # Should return empty results
        assert len(normalized) == 0
        assert df_fast is None

    @patch("core.system7.resolve_signal_entry_date")
    def test_latest_only_predicate_exception(self, mock_resolve):
        """Test latest_only handles predicate exception (line 214-215)."""
        mock_resolve.return_value = pd.Timestamp("2025-04-11")

        dates = pd.date_range("2025-01-01", periods=10, freq="D")

        # DataFrame with invalid data that might cause predicate to fail
        spy_df = pd.DataFrame(
            {
                "Close": [np.nan] * 10,  # All NaN - might cause issues
                "Low": [np.nan] * 10,
                "High": [np.nan] * 10,
                "atr50": [np.nan] * 10,
                "ATR50": [np.nan] * 10,
                "min_50": [np.nan] * 10,
                "Min_50": [np.nan] * 10,
                "max_70": [np.nan] * 10,
                "Max_70": [np.nan] * 10,
                "setup": [False] * 10,
            },
            index=dates,
        )

        data_dict = {"SPY": spy_df}

        # Should handle predicate exception gracefully
        result_tuple = generate_candidates_system7(
            data_dict,
            latest_only=True,
            include_diagnostics=True,
        )

        normalized, df_fast, diagnostics = result_tuple

        # Should return empty results (no setup due to NaN data)
        assert len(normalized) == 0

    @patch("core.system7.resolve_signal_entry_date")
    def test_latest_only_atr50_fallback(self, mock_resolve):
        """Test latest_only ATR50 fallback logic (line 230)."""
        mock_resolve.return_value = pd.Timestamp("2025-04-11")

        dates = pd.date_range("2025-01-01", periods=100, freq="D")

        # Create data with only lowercase atr50 (no ATR50)
        prices = np.linspace(400, 450, 100)
        lows = prices - 2
        highs = prices + 2

        # Setup condition: last day Low <= min_50
        lows[-1] = 380.0  # Below min_50

        spy_df = pd.DataFrame(
            {
                "Close": prices,
                "Low": lows,
                "High": highs,
                "atr50": [p * 0.02 for p in prices],
                # No ATR50 (uppercase) - should fall back to atr50
                "min_50": [390.0] * 100,
                "Min_50": [390.0] * 100,
                "max_70": [450.0] * 100,
                "Max_70": [450.0] * 100,
                "setup": [False] * 99 + [True],
            },
            index=dates,
        )

        raw_dict = {"SPY": spy_df}
        data_dict = prepare_data_vectorized_system7(raw_dict, reuse_indicators=False)

        result_tuple = generate_candidates_system7(
            data_dict,
            latest_only=True,
            include_diagnostics=True,
        )

        normalized, df_fast, diagnostics = result_tuple

        # Should have result using lowercase atr50
        assert len(normalized) >= 1
        assert diagnostics.get("ranking_source") == "latest_only"

        # Verify ATR50 was populated from atr50 fallback
        for date_key in normalized:
            spy_payload = normalized[date_key].get("SPY")
            if spy_payload:
                assert "ATR50" in spy_payload
                assert spy_payload["ATR50"] is not None
