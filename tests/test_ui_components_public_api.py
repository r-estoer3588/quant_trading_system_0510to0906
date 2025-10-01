"""
Test public API of ui_components module.

This file tests only the PUBLIC API that is intended for external use.
Internal functions (prefixed with _) and deprecated features are not tested here.

Public API includes:
- run_backtest_app: Main backtest UI entry point
- prepare_backtest_data: Data preparation for backtesting
- fetch_data: Data fetching with caching
- show_results: Display backtest results
- show_signal_trade_summary: Display signal/trade summary
- clean_date_column: Date column cleaning utility
- save_signal_and_trade_logs: Save signal and trade logs
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from common import ui_components
from common.testing import set_test_determinism


class TestPublicUtilities:
    """Test public utility functions."""

    def setup_method(self):
        set_test_determinism()

    def test_clean_date_column_basic(self):
        """Test clean_date_column with basic input."""
        df = pd.DataFrame(
            {"Date": ["2023-01-01", "2023-01-02", "2023-01-03"], "Close": [100.0, 101.0, 102.0]}
        )
        result = ui_components.clean_date_column(df, "Date")
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_clean_date_column_empty(self):
        """Test clean_date_column with empty dataframe."""
        df = pd.DataFrame({"Date": [], "Close": []})
        result = ui_components.clean_date_column(df, "Date")
        assert isinstance(result, pd.DataFrame)


class TestDataFetching:
    """Test data fetching functions."""

    def setup_method(self):
        set_test_determinism()

    @patch("common.ui_components.get_cached_data")
    def test_fetch_data_basic(self, mock_get_cached):
        """Test fetch_data with mocked cache."""
        mock_data = pd.DataFrame(
            {
                "Date": pd.date_range("2023-01-01", periods=10),
                "Close": [100.0 + i for i in range(10)],
                "Volume": [1000000] * 10,
            }
        )
        mock_get_cached.return_value = mock_data

        symbols = ["AAPL", "MSFT"]
        result = ui_components.fetch_data(symbols)

        assert result is not None
        assert isinstance(result, dict)

    @patch("common.ui_components.get_cached_data")
    def test_fetch_data_empty_symbols(self, mock_get_cached):
        """Test fetch_data with empty symbol list."""
        result = ui_components.fetch_data([])
        assert result is not None


class TestBacktestDataPreparation:
    """Test backtest data preparation."""

    def setup_method(self):
        set_test_determinism()

    @patch("common.ui_components.fetch_data")
    @patch("common.ui_components.get_all_tickers")
    def test_prepare_backtest_data_basic(self, mock_tickers, mock_fetch):
        """Test prepare_backtest_data with mocked dependencies."""
        mock_tickers.return_value = ["AAPL", "MSFT", "GOOGL"]
        mock_fetch.return_value = {
            "AAPL": pd.DataFrame(
                {
                    "Date": pd.date_range("2023-01-01", periods=100),
                    "Close": [150.0] * 100,
                    "Volume": [1000000] * 100,
                }
            )
        }

        mock_strategy = Mock()
        mock_strategy.prepare_data = Mock(
            return_value=pd.DataFrame(
                {"Date": pd.date_range("2023-01-01", periods=100), "Close": [150.0] * 100}
            )
        )

        result = ui_components.prepare_backtest_data(
            strategy=mock_strategy,
            symbols=["AAPL"],
            start_date="2023-01-01",
            end_date="2023-12-31",
            min_volume=100000,
        )

        assert result is not None


class TestResultsDisplay:
    """Test results display functions."""

    def setup_method(self):
        set_test_determinism()

    @patch("streamlit.success")
    @patch("streamlit.metric")
    def test_show_results_basic(self, mock_metric, mock_success):
        """Test show_results with basic dataframe."""
        results_df = pd.DataFrame(
            {
                "entry_date": pd.date_range("2023-01-01", periods=5),
                "exit_date": pd.date_range("2023-01-02", periods=5),
                "pnl": [100, -50, 75, -25, 150],
                "symbol": ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"],
            }
        )

        # Should not raise exceptions
        ui_components.show_results(results_df, 10000.0, "TestSystem")

        # Basic assertions that UI was called
        assert mock_success.called or mock_metric.called

    @patch("streamlit.warning")
    def test_show_results_empty(self, mock_warning):
        """Test show_results with empty dataframe."""
        empty_df = pd.DataFrame()
        ui_components.show_results(empty_df, 10000.0, "TestSystem")
        # Should handle gracefully

    @patch("streamlit.info")
    def test_show_results_missing_columns(self, mock_info):
        df = pd.DataFrame({"pnl": [1, -1, 2]})  # 必須列欠損
        ui_components.show_results(df, 10000.0, "GuardSystem")
        assert mock_info.called  # 早期return


class TestSignalTradeSummary:
    """Test signal and trade summary functions."""

    def setup_method(self):
        set_test_determinism()

    @patch("streamlit.dataframe")
    def test_show_signal_trade_summary_basic(self, mock_dataframe):
        """Test show_signal_trade_summary with basic input."""
        signal_data = {
            "AAPL": pd.DataFrame(
                {"Date": pd.date_range("2023-01-01", periods=10), "Close": [150.0] * 10}
            )
        }

        results_df = pd.DataFrame(
            {
                "entry_date": ["2023-01-01", "2023-01-05"],
                "exit_date": ["2023-01-02", "2023-01-06"],
                "pnl": [100, -50],
                "symbol": ["AAPL", "MSFT"],
            }
        )

        result = ui_components.show_signal_trade_summary(
            signal_data, results_df, "System1", display_name="System 1 Test"
        )

        assert result is not None
        assert isinstance(result, pd.DataFrame)


class TestLoggingFunctions:
    """Test logging and file save functions."""

    def setup_method(self):
        set_test_determinism()

    @pytest.mark.skip(
        reason="save_signal_and_trade_logs uses st.download_button which requires Streamlit context"
    )
    @patch("os.makedirs")
    @patch("pandas.DataFrame.to_csv")
    def test_save_signal_and_trade_logs_basic(self, mock_to_csv, mock_makedirs):
        """Test save_signal_and_trade_logs basic functionality."""
        signal_df = pd.DataFrame({"system": ["System1", "System2"], "count": [10, 15]})

        results_df = pd.DataFrame(
            {
                "entry_date": ["2023-01-01", "2023-01-02"],
                "exit_date": ["2023-01-02", "2023-01-03"],
                "pnl": [100, -50],
                "symbol": ["AAPL", "MSFT"],
            }
        )

        # Should not raise exceptions
        ui_components.save_signal_and_trade_logs(signal_df, results_df, "TestSystem", 10000.0)

        # Verify CSV save was attempted
        assert mock_to_csv.called


class TestBacktestApp:
    """Test main backtest app entry point."""

    def setup_method(self):
        set_test_determinism()

    @patch("common.ui_components.prepare_backtest_data")
    @patch("common.ui_components.run_backtest_with_logging")
    def test_run_backtest_app_basic(self, mock_run, mock_prepare):
        """Test run_backtest_app with mocked dependencies."""
        # Mock prepare_backtest_data to return prepared data
        mock_prepared = {
            "AAPL": pd.DataFrame(
                {"Date": pd.date_range("2023-01-01", periods=100), "Close": [150.0] * 100}
            )
        }
        mock_prepare.return_value = mock_prepared

        # Mock run_backtest_with_logging to return results
        mock_results = pd.DataFrame(
            {
                "entry_date": ["2023-01-01"],
                "exit_date": ["2023-01-02"],
                "pnl": [100],
                "symbol": ["AAPL"],
            }
        )
        mock_merged = pd.DataFrame({"Date": ["2023-01-01"], "signal": [1]})
        mock_run.return_value = (mock_results, mock_merged, None)

        mock_strategy = Mock()

        result = ui_components.run_backtest_app(
            strategy=mock_strategy, system_name="TestSystem", limit_symbols=10, spy_df=None
        )

        # Should return a tuple
        assert result is not None
        assert isinstance(result, tuple)


class TestDefensiveGuards:
    """新規追加: 防御的ガードの挙動確認"""
    def setup_method(self):
        set_test_determinism()

    @patch("streamlit.info")
    def test_show_results_missing_columns(self, mock_info):
        df = pd.DataFrame({"pnl": [1, -1, 2]})  # 必須列欠損
        ui_components.show_results(df, 10000.0, "GuardSystem")
        assert mock_info.called  # 早期return

    def test_show_results_empty_df(self):
        empty_df = pd.DataFrame()
        # 例外発生せず終了すること（戻り値はNone）
        ui_components.show_results(empty_df, 5000.0, "GuardSystem")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
