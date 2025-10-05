"""
Integration tests for ui_components module.

Tests realistic end-to-end scenarios combining multiple components.
Focuses on user workflows rather than individual function testing.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from common import ui_components
from common.testing import set_test_determinism


class TestBacktestWorkflow:
    """Test complete backtest workflow from data loading to results display."""

    def setup_method(self):
        set_test_determinism()

    @patch("common.ui_components.get_cached_data")
    def test_full_backtest_workflow(self, mock_get_cached):
        """Test complete workflow: fetch -> prepare -> backtest -> display."""
        # Setup mock data
        mock_data = pd.DataFrame(
            {
                "Date": pd.date_range("2023-01-01", periods=252),
                "Open": [100.0 + i * 0.1 for i in range(252)],
                "High": [101.0 + i * 0.1 for i in range(252)],
                "Low": [99.0 + i * 0.1 for i in range(252)],
                "Close": [100.5 + i * 0.1 for i in range(252)],
                "Volume": [1000000] * 252,
            }
        )
        mock_get_cached.return_value = mock_data

        # Mock strategy
        mock_strategy = Mock()
        mock_strategy.prepare_data = Mock(return_value=mock_data.copy())
        mock_strategy.generate_signals = Mock(
            return_value=pd.DataFrame(
                {"Date": pd.date_range("2023-01-01", periods=10), "signal": [1] * 10}
            )
        )
        mock_strategy.compute_entry = Mock(return_value=True)
        mock_strategy.compute_exit = Mock(return_value=False)

        # Step 1: Fetch data
        with patch("common.ui_components.get_all_tickers", return_value=["AAPL"]):
            data_dict = ui_components.fetch_data(["AAPL"])
            assert data_dict is not None

        # Step 2: Prepare backtest data
        with patch("common.ui_components.fetch_data", return_value=data_dict):
            prepared = ui_components.prepare_backtest_data(
                strategy=mock_strategy,
                symbols=["AAPL"],
                start_date="2023-01-01",
                end_date="2023-12-31",
                min_volume=100000,
            )
            assert prepared is not None

        # Step 3: Run backtest (with mocked execution)
        with patch("common.ui_components.run_backtest_with_logging") as mock_backtest:
            mock_results = pd.DataFrame(
                {
                    "entry_date": pd.date_range("2023-01-01", periods=5),
                    "exit_date": pd.date_range("2023-01-02", periods=5),
                    "pnl": [100, -50, 75, -25, 150],
                    "symbol": ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
                }
            )
            mock_backtest.return_value = (mock_results, mock_data, None)

            # Step 4: Display results (check it doesn't crash)
            with patch("streamlit.success"):
                ui_components.show_results(mock_results, 10000.0, "IntegrationTest")


class TestErrorHandling:
    """Test error handling in various scenarios."""

    def setup_method(self):
        set_test_determinism()

    @patch("common.ui_components.get_cached_data")
    def test_fetch_data_with_failures(self, mock_get_cached):
        """Test fetch_data handles partial failures gracefully."""
        # First call succeeds, second fails
        mock_get_cached.side_effect = [
            pd.DataFrame({"Close": [100, 101]}),
            Exception("Cache miss"),
            pd.DataFrame({"Close": [200, 201]}),
        ]

        # Should handle failures gracefully
        result = ui_components.fetch_data(["AAPL", "FAIL", "MSFT"])
        assert result is not None

    def test_show_results_with_invalid_data(self):
        """Test show_results handles invalid data."""
        # Create DataFrame with minimal required columns
        # (show_results requires entry_date, exit_date, pnl, symbol minimum)
        minimal_df = pd.DataFrame(
            {
                "entry_date": ["2023-01-01"],
                "exit_date": ["2023-01-02"],
                "pnl": [100],
                "symbol": ["AAPL"],
            }
        )

        with patch("streamlit.success"):
            # Should not crash with minimal data
            ui_components.show_results(minimal_df, 10000.0, "TestSystem")


class TestDataQuality:
    """Test data quality validation and cleaning."""

    def setup_method(self):
        set_test_determinism()

    def test_clean_date_column_various_formats(self):
        """Test date cleaning with various input formats."""
        test_cases = [
            # Standard format
            pd.DataFrame(
                {
                    "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                    "Close": [100, 101, 102],
                }
            ),
            # With NaN values
            pd.DataFrame({"Date": ["2023-01-01", None, "2023-01-03"], "Close": [100, 101, 102]}),
            # Empty strings
            pd.DataFrame({"Date": ["2023-01-01", "", "2023-01-03"], "Close": [100, 101, 102]}),
        ]

        for df in test_cases:
            result = ui_components.clean_date_column(df, "Date")
            assert isinstance(result, pd.DataFrame)
            # Should return valid DataFrame (may filter rows)


class TestPerformanceMetrics:
    """Test performance metrics calculation and display."""

    def setup_method(self):
        set_test_determinism()

    @pytest.mark.skip(reason="Requires GUI environment (matplotlib/tkinter)")
    @patch("streamlit.success")
    @patch("streamlit.subheader")
    def test_metrics_display_with_wins_and_losses(self, mock_subheader, mock_success):
        """Test metrics display with mixed win/loss trades."""
        results_df = pd.DataFrame(
            {
                "entry_date": pd.date_range("2023-01-01", periods=20),
                "exit_date": pd.date_range("2023-01-02", periods=20),
                "pnl": [100, -50, 75, -25, 150] * 4,
                "symbol": ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"] * 4,
            }
        )

        ui_components.show_results(results_df, 10000.0, "MetricsTest")

        # Verify some display happened (success or subheader)
        assert mock_success.called or mock_subheader.called

    @patch("streamlit.warning")
    def test_metrics_display_all_losses(self, mock_warning):
        """Test metrics display with only losing trades."""
        results_df = pd.DataFrame(
            {
                "entry_date": pd.date_range("2023-01-01", periods=10),
                "exit_date": pd.date_range("2023-01-02", periods=10),
                "pnl": [-50, -25, -100, -75, -30, -60, -45, -80, -55, -70],
                "symbol": ["AAPL"] * 10,
            }
        )

        with patch("streamlit.metric"):
            ui_components.show_results(results_df, 10000.0, "LossTest")


class TestConcurrentOperations:
    """Test concurrent data operations."""

    def setup_method(self):
        set_test_determinism()

    @patch("common.ui_components.get_cached_data")
    def test_fetch_multiple_symbols_concurrently(self, mock_get_cached):
        """Test fetching multiple symbols handles concurrency."""
        mock_get_cached.return_value = pd.DataFrame(
            {
                "Date": pd.date_range("2023-01-01", periods=100),
                "Close": [100.0] * 100,
                "Volume": [1000000] * 100,
            }
        )

        # Large symbol list to trigger concurrent processing
        symbols = [f"SYM{i:03d}" for i in range(50)]

        result = ui_components.fetch_data(symbols)
        assert result is not None
        # Should handle concurrent processing without errors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
