"""
Partial tests for core.system7 module to improve coverage
Focus on utility functions that are testable in isolation
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from common.testing import set_test_determinism
from core.system7 import generate_candidates_system7, get_total_days_system7


class TestGetTotalDaysSystem7:
    """Test get_total_days_system7 utility function"""

    def setup_method(self):
        set_test_determinism()

    def test_get_total_days_with_empty_dict(self):
        """Test get_total_days_system7 with empty dictionary"""
        result = get_total_days_system7({})
        assert result == 0

    def test_get_total_days_with_none_dataframes(self):
        """Test get_total_days_system7 with None DataFrames"""
        data_dict = {"SPY": None, "AAPL": None}
        result = get_total_days_system7(data_dict)
        assert result == 0

    def test_get_total_days_with_empty_dataframes(self):
        """Test get_total_days_system7 with empty DataFrames"""
        data_dict = {"SPY": pd.DataFrame(), "AAPL": pd.DataFrame()}
        result = get_total_days_system7(data_dict)
        assert result == 0

    def test_get_total_days_with_date_column(self):
        """Test get_total_days_system7 with Date column"""
        data_dict = {
            "SPY": pd.DataFrame(
                {"Date": ["2023-01-01", "2023-01-02", "2023-01-03"], "Close": [100, 101, 102]}
            ),
            "AAPL": pd.DataFrame(
                {"Date": ["2023-01-02", "2023-01-03", "2023-01-04"], "Close": [150, 151, 152]}
            ),
        }
        result = get_total_days_system7(data_dict)
        # Unique dates: 2023-01-01, 2023-01-02, 2023-01-03, 2023-01-04
        assert result == 4

    def test_get_total_days_with_datetime_index(self):
        """Test get_total_days_system7 with datetime index"""
        spy_df = pd.DataFrame({"Close": [100, 101, 102]})
        spy_df.index = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])

        aapl_df = pd.DataFrame({"Close": [150, 151, 152]})
        aapl_df.index = pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04"])

        data_dict = {"SPY": spy_df, "AAPL": aapl_df}
        result = get_total_days_system7(data_dict)
        # Unique dates: 2023-01-01, 2023-01-02, 2023-01-03, 2023-01-04
        assert result == 4

    def test_get_total_days_with_overlapping_dates(self):
        """Test get_total_days_system7 with overlapping dates"""
        data_dict = {
            "SPY": pd.DataFrame({"Date": ["2023-01-01", "2023-01-02"], "Close": [100, 101]}),
            "AAPL": pd.DataFrame(
                {"Date": ["2023-01-01", "2023-01-02"], "Close": [150, 151]}  # Same dates
            ),
        }
        result = get_total_days_system7(data_dict)
        # Only unique dates count: 2023-01-01, 2023-01-02
        assert result == 2


class TestGenerateCandidatesSystem7:
    """Test generate_candidates_system7 function"""

    def setup_method(self):
        set_test_determinism()

    def test_generate_candidates_no_spy_data(self):
        """Test generate_candidates_system7 with no SPY data"""
        prepared_dict = {"AAPL": pd.DataFrame()}

        candidates_by_date, summary_df = generate_candidates_system7(prepared_dict)

        assert candidates_by_date == {}
        assert summary_df is None

    def test_generate_candidates_empty_spy_data(self):
        """Test generate_candidates_system7 with empty SPY DataFrame"""
        # Create empty DataFrame with required columns
        empty_spy = pd.DataFrame(columns=["setup", "ATR50", "Close"])
        prepared_dict = {"SPY": empty_spy}

        candidates_by_date, summary_df = generate_candidates_system7(prepared_dict)

        assert candidates_by_date == {}
        assert summary_df is None

    def test_generate_candidates_no_setup_signals(self):
        """Test generate_candidates_system7 with no setup signals"""
        spy_df = pd.DataFrame(
            {"setup": [0, 0, 0], "ATR50": [1.0, 1.1, 1.2], "Close": [100, 101, 102]}
        )
        spy_df.index = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])

        prepared_dict = {"SPY": spy_df}

        candidates_by_date, summary_df = generate_candidates_system7(prepared_dict)

        assert candidates_by_date == {}
        assert summary_df is None

    @patch("core.system7.resolve_signal_entry_date")
    def test_generate_candidates_with_setup_signals(self, mock_resolve_date):
        """Test generate_candidates_system7 with setup signals"""

        # Mock resolve_signal_entry_date to return next day
        def mock_resolve(date):
            return date + pd.Timedelta(days=1)

        mock_resolve_date.side_effect = mock_resolve

        spy_df = pd.DataFrame(
            {"setup": [1, 0, 1], "ATR50": [1.0, 1.1, 1.2], "Close": [100, 101, 102]}
        )
        spy_df.index = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])

        prepared_dict = {"SPY": spy_df}

        candidates_by_date, summary_df = generate_candidates_system7(prepared_dict)

        # Should have 2 setup signals
        assert len(candidates_by_date) == 2

        # Check first setup signal (2023-01-01 -> 2023-01-02 entry)
        expected_date1 = pd.Timestamp("2023-01-02")
        assert expected_date1 in candidates_by_date

        candidate1 = candidates_by_date[expected_date1][0]
        assert candidate1["symbol"] == "SPY"
        assert candidate1["entry_date"] == expected_date1
        assert candidate1["ATR50"] == 1.0
        assert candidate1["entry_price"] == 102  # Last price from Close column

        # Check second setup signal (2023-01-03 -> 2023-01-04 entry)
        expected_date2 = pd.Timestamp("2023-01-04")
        assert expected_date2 in candidates_by_date

        candidate2 = candidates_by_date[expected_date2][0]
        assert candidate2["symbol"] == "SPY"
        assert candidate2["entry_date"] == expected_date2
        assert candidate2["ATR50"] == 1.2
        assert candidate2["entry_price"] == 102  # Last price from Close column

    @patch("core.system7.resolve_signal_entry_date")
    def test_generate_candidates_with_top_n_limit(self, mock_resolve_date):
        """Test generate_candidates_system7 with top_n limit"""
        # Mock resolve_signal_entry_date to return same date for all signals
        mock_resolve_date.return_value = pd.Timestamp("2023-01-10")

        spy_df = pd.DataFrame(
            {
                "setup": [1, 1, 1],  # 3 setup signals
                "ATR50": [1.0, 1.1, 1.2],
                "Close": [100, 101, 102],
            }
        )
        spy_df.index = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])

        prepared_dict = {"SPY": spy_df}

        # Limit to top 2
        candidates_by_date, summary_df = generate_candidates_system7(prepared_dict, top_n=2)

        # Should only have 1 date with 2 candidates (limited by top_n)
        assert len(candidates_by_date) == 1

        expected_date = pd.Timestamp("2023-01-10")
        assert expected_date in candidates_by_date
        assert len(candidates_by_date[expected_date]) == 2  # Limited to 2

    @patch("core.system7.resolve_signal_entry_date")
    def test_generate_candidates_with_zero_top_n(self, mock_resolve_date):
        """Test generate_candidates_system7 with top_n=0"""
        mock_resolve_date.return_value = pd.Timestamp("2023-01-10")

        spy_df = pd.DataFrame({"setup": [1, 1], "ATR50": [1.0, 1.1], "Close": [100, 101]})
        spy_df.index = pd.to_datetime(["2023-01-01", "2023-01-02"])

        prepared_dict = {"SPY": spy_df}

        candidates_by_date, summary_df = generate_candidates_system7(prepared_dict, top_n=0)

        # top_n=0 should result in no candidates
        assert candidates_by_date == {}
        assert summary_df is None

    @patch("core.system7.resolve_signal_entry_date")
    def test_generate_candidates_with_invalid_top_n(self, mock_resolve_date):
        """Test generate_candidates_system7 with invalid top_n"""
        mock_resolve_date.return_value = pd.Timestamp("2023-01-10")

        spy_df = pd.DataFrame({"setup": [1], "ATR50": [1.0], "Close": [100]})
        spy_df.index = pd.to_datetime(["2023-01-01"])

        prepared_dict = {"SPY": spy_df}

        # Test with string top_n (should be treated as None)
        candidates_by_date, summary_df = generate_candidates_system7(prepared_dict, top_n="invalid")

        # Should work without limit
        assert len(candidates_by_date) == 1

    @patch("core.system7.resolve_signal_entry_date")
    def test_generate_candidates_with_invalid_entry_date(self, mock_resolve_date):
        """Test generate_candidates_system7 with invalid entry date"""
        # Mock resolve_signal_entry_date to return NaN
        mock_resolve_date.return_value = pd.NaT

        spy_df = pd.DataFrame({"setup": [1], "ATR50": [1.0], "Close": [100]})
        spy_df.index = pd.to_datetime(["2023-01-01"])

        prepared_dict = {"SPY": spy_df}

        candidates_by_date, summary_df = generate_candidates_system7(prepared_dict)

        # Should skip invalid entry dates
        assert candidates_by_date == {}
        assert summary_df is None

    def test_generate_candidates_with_callbacks(self):
        """Test generate_candidates_system7 with callbacks"""
        spy_df = pd.DataFrame({"setup": [0], "ATR50": [1.0], "Close": [100]})  # No setup signals
        spy_df.index = pd.to_datetime(["2023-01-01"])

        prepared_dict = {"SPY": spy_df}

        # Test with log callback
        log_messages = []

        def log_callback(msg):
            log_messages.append(msg)

        progress_calls = []

        def progress_callback(current, total):
            progress_calls.append((current, total))

        candidates_by_date, summary_df = generate_candidates_system7(
            prepared_dict, log_callback=log_callback, progress_callback=progress_callback
        )

        assert candidates_by_date == {}
        assert len(log_messages) == 1
        assert "候補日数" in log_messages[0]
        assert progress_calls == [(1, 1)]

    def test_generate_candidates_no_close_column(self):
        """Test generate_candidates_system7 without Close column"""
        spy_df = pd.DataFrame(
            {
                "setup": [1],
                "ATR50": [1.0],
                # No Close column
            }
        )
        spy_df.index = pd.to_datetime(["2023-01-01"])

        prepared_dict = {"SPY": spy_df}

        with patch("core.system7.resolve_signal_entry_date") as mock_resolve:
            mock_resolve.return_value = pd.Timestamp("2023-01-02")

            candidates_by_date, summary_df = generate_candidates_system7(prepared_dict)

            # Should still work but entry_price will be None
            assert len(candidates_by_date) == 1
            expected_date = pd.Timestamp("2023-01-02")
            candidate = candidates_by_date[expected_date][0]
            assert candidate["entry_price"] is None


if __name__ == "__main__":
    pytest.main([__file__])
