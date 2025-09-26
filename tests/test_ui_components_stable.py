"""
Simplified test file for ui_components.py with correct function signatures
Focus on stable, basic functionality testing to achieve coverage
"""

from __future__ import annotations

import pandas as pd
from unittest.mock import Mock, patch

from common import ui_components as ui_comp


class TestBasicFunctions:
    """Test basic utility functions"""

    def test_clean_date_column_basic(self):
        """Test basic clean_date_column functionality"""
        df = pd.DataFrame({"date": ["2023-01-01", "2023-01-02", "2023-01-03"], "value": [1, 2, 3]})

        result_df = ui_comp.clean_date_column(df, "date")

        assert "date" in result_df.columns
        assert pd.api.types.is_datetime64_any_dtype(result_df["date"])
        assert len(result_df) == 3

    def test_clean_date_column_with_invalid_dates(self):
        """Test clean_date_column with invalid dates"""
        df = pd.DataFrame({"date": ["2023-01-01", "invalid", "2023-01-03"], "value": [1, 2, 3]})

        result_df = ui_comp.clean_date_column(df, "date")

        assert "date" in result_df.columns
        assert pd.api.types.is_datetime64_any_dtype(result_df["date"])
        assert pd.isna(result_df.loc[1, "date"])

    @patch("streamlit.progress")
    @patch("streamlit.text")
    def test_log_with_progress(self, mock_text, mock_progress):
        """Test log_with_progress function"""
        progress_bar = Mock()
        mock_progress.return_value = progress_bar

        # Test basic logging
        ui_comp.log_with_progress("Test message", 0.5, progress_bar)

        mock_text.assert_called_once_with("Test message")
        progress_bar.progress.assert_called_once_with(0.5)

    def test_safe_filename_utility(self):
        """Test safe_filename function access"""
        # This function should be available from utils
        from common.utils import safe_filename

        result = safe_filename("Test/String*With?Invalid<>Chars")

        assert isinstance(result, str)
        assert "/" not in result
        assert "*" not in result
        assert "?" not in result

    def test_summarize_results_basic(self):
        """Test summarize_results with valid data"""
        results_df = pd.DataFrame(
            {
                "entry_date": pd.date_range("2023-01-01", periods=3),
                "exit_date": pd.date_range("2023-01-02", periods=3),
                "pnl": [100, -50, 75],
                "symbol": ["AAPL", "MSFT", "GOOGL"],
            }
        )

        capital = 10000.0

        result = ui_comp.summarize_results(results_df, capital)

        assert isinstance(result, dict)
        assert "trades" in result
        assert "total_return" in result
        assert result["trades"] == 3

    def test_summarize_results_empty(self):
        """Test summarize_results with empty dataframe"""
        empty_df = pd.DataFrame()
        capital = 10000.0

        result = ui_comp.summarize_results(empty_df, capital)

        assert isinstance(result, dict)
        assert result["trades"] == 0

    @patch("streamlit.info")
    def test_show_results_empty(self, mock_info):
        """Test show_results with empty data"""
        empty_df = pd.DataFrame()
        capital = 10000.0

        ui_comp.show_results(empty_df, capital, "TestSystem")

        mock_info.assert_called_once()

    @patch("streamlit.success")
    @patch("streamlit.subheader")
    def test_show_results_basic(self, mock_subheader, mock_success):
        """Test show_results with valid data"""
        results_df = pd.DataFrame(
            {
                "entry_date": pd.date_range("2023-01-01", periods=2),
                "exit_date": pd.date_range("2023-01-02", periods=2),
                "pnl": [100, -50],
                "symbol": ["AAPL", "MSFT"],
            }
        )
        capital = 10000.0

        ui_comp.show_results(results_df, capital, "TestSystem")

        mock_success.assert_called_once()

    def test_extract_zero_reason_from_logs_basic(self):
        """Test extract_zero_reason_from_logs function"""
        logs = [
            "2023-01-01: Processing started",
            "2023-01-01: No candidates found - insufficient volume",
            "2023-01-01: Processing completed",
        ]

        reason = ui_comp.extract_zero_reason_from_logs(logs)

        assert isinstance(reason, str)
        assert len(reason) > 0

    def test_extract_zero_reason_from_logs_empty(self):
        """Test extract_zero_reason_from_logs with empty logs"""
        logs = []

        reason = ui_comp.extract_zero_reason_from_logs(logs)

        assert isinstance(reason, str)

    def test_default_log_callback_basic(self):
        """Test default_log_callback function"""
        # Should not raise exception
        ui_comp.default_log_callback("Test message")

        # Test with different message types
        ui_comp.default_log_callback("")
        ui_comp.default_log_callback("Multi\nLine\nMessage")


class TestDataFetching:
    """Test data fetching functionality"""

    @patch("common.ui_components.get_cached_data")
    def test_fetch_data_basic(self, mock_get_cached):
        """Test basic fetch_data functionality"""
        mock_data = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT"],
                "date": pd.date_range("2023-01-01", periods=2),
                "close": [150.0, 250.0],
            }
        )
        mock_get_cached.return_value = mock_data

        result = ui_comp.fetch_data(["AAPL", "MSFT"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        mock_get_cached.assert_called_once()

    @patch("common.ui_components.get_cached_data")
    def test_fetch_data_empty_symbols(self, mock_get_cached):
        """Test fetch_data with empty symbol list"""
        mock_get_cached.return_value = pd.DataFrame()

        result = ui_comp.fetch_data([])

        assert isinstance(result, pd.DataFrame)

    @patch("common.ui_components.get_cached_data")
    def test_fetch_data_with_exception(self, mock_get_cached):
        """Test fetch_data when get_cached_data raises exception"""
        mock_get_cached.side_effect = Exception("Data fetch error")

        # Should not raise exception but return empty DataFrame or handle gracefully
        result = ui_comp.fetch_data(["AAPL"])

        # The function should handle exceptions gracefully
        assert result is None or isinstance(result, pd.DataFrame)


class TestStreamlitIntegration:
    """Test Streamlit component integration"""

    @patch("streamlit.subheader")
    def test_display_cache_health_dashboard_basic(self, mock_subheader):
        """Test basic display_cache_health_dashboard functionality"""
        with patch("common.cache_manager.CacheManager") as mock_cache_manager:
            mock_manager = Mock()
            mock_cache_manager.return_value = mock_manager
            mock_manager.get_cache_health.return_value = {
                "total_symbols": 100,
                "healthy_symbols": 95,
                "last_updated": "2023-01-01",
            }

            # Should not raise exception
            ui_comp.display_cache_health_dashboard()

            mock_subheader.assert_called()
