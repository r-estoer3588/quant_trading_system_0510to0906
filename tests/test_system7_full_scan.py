"""
Test full scan mode (latest_only=False) for System7.

This module tests the comprehensive historical scanning path
that processes all setup signals, not just the most recent one.
"""

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from core.system7 import generate_candidates_system7


@pytest.fixture(autouse=True)
def clean_system7_cache():
    """Remove System7 indicator cache before each test."""
    cache_path = "data_cache/indicators_system7_cache/SPY.feather"
    if os.path.exists(cache_path):
        os.remove(cache_path)
    yield
    if os.path.exists(cache_path):
        os.remove(cache_path)


class TestSystem7FullScan:
    """Test full historical scan mode (latest_only=False)."""

    def create_spy_with_multiple_setups(self, num_setups=5):
        """Create SPY data with multiple setup signals spread across history."""
        dates = pd.date_range("2025-01-01", periods=100, freq="D")

        prices = np.linspace(400, 450, 100)
        lows = prices - 2
        highs = prices + 2

        # Create multiple setup days (every 20 days)
        setup_days = list(range(19, 100, 20))[:num_setups]

        min_50_values = [450.0] * 100  # Default: no setup (Low < min_50 is False)
        setup_flags = [False] * 100

        for day in setup_days:
            lows[day] = 380.0  # Below min_50
            min_50_values[day] = 390.0  # Setup condition: Low <= min_50
            setup_flags[day] = True  # Mark as setup

        spy_df = pd.DataFrame(
            {
                "Close": prices,
                "Low": lows,
                "High": highs,
                "atr50": [p * 0.02 for p in prices],
                "ATR50": [p * 0.02 for p in prices],
                "min_50": min_50_values,
                "Min_50": min_50_values,
                "max_70": [450.0] * 100,
                "Max_70": [450.0] * 100,
                "setup": setup_flags,  # Explicit setup flags
            },
            index=dates,
        )

        return spy_df

    @patch("core.system7.resolve_signal_entry_date")
    def test_full_scan_multiple_candidates(self, mock_resolve):
        """Test full scan returns multiple candidates (lines 275-340)."""

        # Mock to return day after setup
        def mock_entry_date(setup_date):
            return setup_date + pd.Timedelta(days=1)

        mock_resolve.side_effect = mock_entry_date

        spy_data = self.create_spy_with_multiple_setups(num_setups=5)

        # Skip prepare_data to preserve explicit setup flags
        data_dict = {"SPY": spy_data}

        result_tuple = generate_candidates_system7(
            data_dict,
            latest_only=False,  # Full historical scan
            include_diagnostics=True,
        )

        normalized, df_fast, diagnostics = result_tuple

        # Should have multiple candidates (one per setup)
        assert len(normalized) >= 3, f"Expected >=3 candidates, got {len(normalized)}"

        # Diagnostics should indicate full scan
        assert diagnostics.get("ranking_source") != "latest_only"

    @patch("core.system7.resolve_signal_entry_date")
    def test_full_scan_with_top_n_limit(self, mock_resolve):
        """Test full scan respects top_n limit per date (lines 303, 307-308).

        Note: top_n limits symbols per entry date, not total dates.
        For System7 (SPY only), each date has max 1 symbol.
        """

        def mock_entry_date(setup_date):
            return setup_date + pd.Timedelta(days=1)

        mock_resolve.side_effect = mock_entry_date

        spy_data = self.create_spy_with_multiple_setups(num_setups=5)

        # Skip prepare_data to preserve explicit setup flags
        data_dict = {"SPY": spy_data}

        result_tuple = generate_candidates_system7(
            data_dict,
            latest_only=False,
            top_n=1,  # Limit to 1 symbol per date
            include_diagnostics=True,
        )

        normalized, df_fast, diagnostics = result_tuple

        # Should have 5 candidates (one per setup day)
        # top_n limits symbols per date, not total dates
        assert len(normalized) == 5, f"Expected 5 candidates, got {len(normalized)}"

        # Each date should have exactly 1 symbol (SPY)
        for entry_date in normalized:
            assert (
                len(normalized[entry_date]) <= 1
            ), "Expected <=1 symbol per date (top_n=1)"
            assert "SPY" in normalized[entry_date]
            spy_payload = normalized[entry_date]["SPY"]
            assert "ATR50" in spy_payload

    @patch("core.system7.resolve_signal_entry_date")
    def test_full_scan_with_callbacks(self, mock_resolve):
        """Test full scan invokes callbacks (lines 360-361, 366-373)."""

        def mock_entry_date(setup_date):
            return setup_date + pd.Timedelta(days=1)

        mock_resolve.side_effect = mock_entry_date

        spy_data = self.create_spy_with_multiple_setups(num_setups=3)

        # Skip prepare_data to preserve explicit setup flags
        data_dict = {"SPY": spy_data}

        log_messages = []
        progress_calls = []

        def test_log_callback(msg):
            log_messages.append(msg)

        def test_progress_callback(current, total):
            progress_calls.append((current, total))

        result_tuple = generate_candidates_system7(
            data_dict,
            latest_only=False,
            include_diagnostics=True,
            log_callback=test_log_callback,
            progress_callback=test_progress_callback,
        )

        normalized, df_fast, diagnostics = result_tuple

        # Verify callbacks were invoked
        assert len(log_messages) > 0, "log_callback should be invoked"
        assert len(progress_calls) > 0, "progress_callback should be invoked"

    @patch("core.system7.resolve_signal_entry_date")
    def test_full_scan_date_grouping(self, mock_resolve):
        """Test full scan groups candidates by date (lines 292-297)."""
        # Mock: all setups map to same entry date
        fixed_entry = pd.Timestamp("2025-02-01")
        mock_resolve.return_value = fixed_entry

        spy_data = self.create_spy_with_multiple_setups(num_setups=5)

        # Skip prepare_data to preserve explicit setup flags
        data_dict = {"SPY": spy_data}

        result_tuple = generate_candidates_system7(
            data_dict,
            latest_only=False,
            include_diagnostics=True,
        )

        normalized, df_fast, diagnostics = result_tuple

        # All candidates should map to same entry date
        if len(normalized) > 0:
            assert fixed_entry in normalized or pd.Timestamp(fixed_entry) in normalized
            # Should have SPY in that date bucket
            for entry_date in normalized:
                assert "SPY" in normalized[entry_date]

    @patch("core.system7.resolve_signal_entry_date")
    def test_full_scan_without_diagnostics(self, mock_resolve):
        """Test full scan with include_diagnostics=False (line 388)."""

        def mock_entry_date(setup_date):
            return setup_date + pd.Timedelta(days=1)

        mock_resolve.side_effect = mock_entry_date

        spy_data = self.create_spy_with_multiple_setups(num_setups=3)

        # Skip prepare_data to preserve explicit setup flags
        data_dict = {"SPY": spy_data}

        result_tuple = generate_candidates_system7(
            data_dict,
            latest_only=False,
            include_diagnostics=False,  # Should return 2-tuple
        )

        # Should return 2-tuple
        assert len(result_tuple) == 2, "Without diagnostics, should return 2-tuple"

        normalized, df_fast = result_tuple

        # Verify results are valid
        assert isinstance(normalized, dict)
        assert len(normalized) >= 1

    @patch("core.system7.resolve_signal_entry_date")
    def test_full_scan_no_setups(self, mock_resolve):
        """Test full scan with no setup conditions met (line 349)."""
        mock_resolve.return_value = pd.Timestamp("2025-04-11")

        dates = pd.date_range("2025-01-01", periods=50, freq="D")

        # No setup conditions met (Low always > min_50)
        prices = np.linspace(400, 450, 50)
        spy_df = pd.DataFrame(
            {
                "Close": prices,
                "Low": prices + 10,  # Always above min_50
                "High": prices + 20,
                "atr50": [p * 0.02 for p in prices],
                "ATR50": [p * 0.02 for p in prices],
                "min_50": [390.0] * 50,
                "Min_50": [390.0] * 50,
                "max_70": [450.0] * 50,
                "Max_70": [450.0] * 50,
                "setup": [False] * 50,
            },
            index=dates,
        )

        # Skip prepare_data to preserve explicit setup flags
        data_dict = {"SPY": spy_df}

        result_tuple = generate_candidates_system7(
            data_dict,
            latest_only=False,
            include_diagnostics=True,
        )

        normalized, df_fast, diagnostics = result_tuple

        # Should return empty results
        assert len(normalized) == 0, "No setups should mean no candidates"
        assert df_fast is None or df_fast.empty
