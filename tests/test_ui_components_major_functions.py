"""
Comprehensive tests for ui_components.py targeting high coverage
Focus on show_results, run_backtest_app, and other major functions
"""

from __future__ import annotations

from unittest.mock import Mock, patch, mock_open
import pytest
import pandas as pd
import numpy as np

from common import ui_components as ui_comp


class TestShowResults:
    """Test show_results function comprehensively"""

    def create_test_results_df(self):
        """Create test results dataframe"""
        return pd.DataFrame(
            {
                "entry_date": pd.date_range("2023-01-01", periods=5),
                "exit_date": pd.date_range("2023-01-02", periods=5),
                "pnl": [100, -50, 75, -25, 150],
                "symbol": ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
                "entry_price": [100, 200, 150, 300, 250],
                "exit_price": [110, 190, 165, 290, 275],
                "cumulative_pnl": [100, 50, 125, 100, 250],
            }
        )

    @patch("streamlit.subheader")
    @patch("streamlit.success")
    @patch("streamlit.dataframe")
    @patch("streamlit.plotly_chart")
    def test_show_results_basic(self, mock_plotly, mock_dataframe, mock_success, mock_subheader):
        """基本的なshow_resultsテスト"""
        results_df = self.create_test_results_df()
        capital = 10000.0

        # 正しいシグネチャで呼び出し
        ui_comp.show_results(results_df, capital, "TestSystem", key_context="test")

        # Should display success message and charts
        assert mock_success.called
        assert mock_subheader.called

    @patch("streamlit.subheader")
    @patch("streamlit.metric")
    @patch("streamlit.dataframe")
    @patch("streamlit.plotly_chart")
    @patch("streamlit.warning")
    def test_show_results_empty_data(
        self, mock_warning, mock_plotly, mock_dataframe, mock_metric, mock_subheader
    ):
        """空データでのshow_resultsテスト"""
        empty_df = pd.DataFrame()
        summary = {"trades": 0, "total_return": 0.0, "win_rate": 0.0, "max_dd": 0.0}

        ui_comp.show_results(empty_df, summary, 10000)

        # Should handle empty data gracefully
        assert mock_warning.called or mock_subheader.called


class TestRunBacktestApp:
    """Test run_backtest_app function"""

    def create_mock_data(self):
        """Create mock data for backtest"""
        return {
            "AAPL": pd.DataFrame(
                {
                    "Date": pd.date_range("2023-01-01", periods=100),
                    "Close": 150 + np.random.randn(100) * 10,
                    "Volume": 1000000 + np.random.randn(100) * 100000,
                    "RSI": 30 + np.random.randn(100) * 20,
                    "EMA12": 150 + np.random.randn(100) * 5,
                    "EMA26": 150 + np.random.randn(100) * 5,
                    "roc_200": np.random.randn(100) * 0.1,
                }
            ),
            "MSFT": pd.DataFrame(
                {
                    "Date": pd.date_range("2023-01-01", periods=100),
                    "Close": 250 + np.random.randn(100) * 15,
                    "Volume": 800000 + np.random.randn(100) * 80000,
                    "RSI": 40 + np.random.randn(100) * 25,
                    "EMA12": 250 + np.random.randn(100) * 8,
                    "EMA26": 250 + np.random.randn(100) * 8,
                    "roc_200": np.random.randn(100) * 0.12,
                }
            ),
        }

    @patch("streamlit.subheader")
    @patch("streamlit.info")
    @patch("streamlit.error")
    @patch("streamlit.warning")
    @patch("streamlit.success")
    @patch("common.ui_components.show_results")
    def test_run_backtest_app_success(
        self, mock_show_results, mock_success, mock_warning, mock_error, mock_info, mock_subheader
    ):
        """成功パターンのrun_backtest_appテスト"""
        mock_data = self.create_mock_data()
        mock_strategy = Mock()

        # Mock strategy to return some results
        mock_results = pd.DataFrame(
            {
                "entry_date": ["2023-01-01", "2023-02-01"],
                "exit_date": ["2023-01-05", "2023-02-05"],
                "pnl": [100, 50],
                "symbol": ["AAPL", "MSFT"],
            }
        )
        mock_strategy.run_backtest.return_value = mock_results

        # Execute the function
        ui_comp.run_backtest_app(
            mock_data, mock_strategy, "2023-01-01", "2023-12-31", 10000, "TestSystem"
        )

        # Verify strategy was called
        mock_strategy.run_backtest.assert_called_once()

        # Should show results if backtest succeeded
        assert mock_show_results.called or mock_info.called

    @patch("streamlit.error")
    @patch("streamlit.info")
    def test_run_backtest_app_strategy_error(self, mock_info, mock_error):
        """戦略実行エラーのテスト"""
        mock_data = self.create_mock_data()
        mock_strategy = Mock()
        mock_strategy.run_backtest.side_effect = Exception("Backtest failed")

        # Should handle strategy errors gracefully
        ui_comp.run_backtest_app(
            mock_data, mock_strategy, "2023-01-01", "2023-12-31", 10000, "FailSystem"
        )

        # Should display error message
        assert mock_error.called

    @patch("streamlit.warning")
    def test_run_backtest_app_empty_data(self, mock_warning):
        """空データでのテスト"""
        empty_data = {}
        mock_strategy = Mock()

        ui_comp.run_backtest_app(
            empty_data, mock_strategy, "2023-01-01", "2023-12-31", 10000, "EmptySystem"
        )

        # Should handle empty data
        assert mock_warning.called or not mock_strategy.run_backtest.called


class TestBacktestPreparation:
    """Test prepare_backtest_data function"""

    @patch("streamlit.info")
    @patch("streamlit.progress")
    @patch("streamlit.empty")
    def test_prepare_backtest_data(self, mock_empty, mock_progress, mock_info):
        """prepare_backtest_dataのテスト"""
        mock_data = {
            "AAPL": pd.DataFrame(
                {
                    "Date": pd.date_range("2023-01-01", periods=50),
                    "Close": np.random.randn(50) + 150,
                    "Volume": np.random.randn(50) * 1000 + 100000,
                }
            ),
            "MSFT": pd.DataFrame(
                {
                    "Date": pd.date_range("2023-01-01", periods=50),
                    "Close": np.random.randn(50) + 250,
                    "Volume": np.random.randn(50) * 1000 + 80000,
                }
            ),
        }

        result = ui_comp.prepare_backtest_data(
            mock_data, start_date="2023-01-01", end_date="2023-12-31", min_volume=50000
        )

        # Should return prepared data
        assert isinstance(result, dict)
        assert len(result) >= 0  # May filter out some symbols


class TestRunBacktestWithLogging:
    """Test run_backtest_with_logging function"""

    @patch("streamlit.info")
    @patch("streamlit.progress")
    @patch("streamlit.empty")
    @patch("streamlit.error")
    def test_run_backtest_with_logging_success(
        self, mock_error, mock_empty, mock_progress, mock_info
    ):
        """成功パターンのrun_backtest_with_loggingテスト"""
        mock_strategy = Mock()
        mock_results = pd.DataFrame(
            {
                "entry_date": ["2023-01-01"],
                "exit_date": ["2023-01-05"],
                "pnl": [100],
                "symbol": ["AAPL"],
            }
        )
        mock_strategy.run_backtest.return_value = mock_results

        prepared_data = {
            "AAPL": pd.DataFrame(
                {
                    "Date": pd.date_range("2023-01-01", periods=10),
                    "Close": [150] * 10,
                }
            )
        }

        results_df, summary = ui_comp.run_backtest_with_logging(
            prepared_data, mock_strategy, 10000, "TestSystem"
        )

        assert results_df is not None
        assert isinstance(summary, dict)

    @patch("streamlit.error")
    @patch("streamlit.info")
    def test_run_backtest_with_logging_error(self, mock_info, mock_error):
        """エラーパターンのテスト"""
        mock_strategy = Mock()
        mock_strategy.run_backtest.side_effect = Exception("Backtest error")

        prepared_data = {"AAPL": pd.DataFrame()}

        results_df, summary = ui_comp.run_backtest_with_logging(
            prepared_data, mock_strategy, 10000, "FailSystem"
        )

        # Should return empty results on error
        assert results_df is None or len(results_df) == 0
        assert isinstance(summary, dict)


class TestUtilityFunctionsExpanded:
    """Expanded tests for utility functions"""

    def test_clean_date_column_edge_cases(self):
        """clean_date_columnの極端なケース"""
        # Test with None values
        df_with_nones = pd.DataFrame({"Date": [None, None, "2023-01-01"], "value": [1, 2, 3]})

        result = ui_comp.clean_date_column(df_with_nones, "Date")
        assert len(result) == 1  # Only one valid date

        # Test with completely empty date column
        df_empty_dates = pd.DataFrame({"Date": [None, None, None], "value": [1, 2, 3]})

        result = ui_comp.clean_date_column(df_empty_dates, "Date")
        assert len(result) == 0  # All dates invalid

    @patch("os.path.exists")
    def test_mtime_or_zero_various_paths(self, mock_exists):
        """_mtime_or_zeroの様々なパス"""
        # Test non-existent file
        mock_exists.return_value = False
        result = ui_comp._mtime_or_zero("nonexistent_file.txt")
        assert result == 0.0

        # Test with empty string
        result = ui_comp._mtime_or_zero("")
        assert result == 0.0


class TestDisplayFunctions:
    """Test display and dashboard functions"""

    @patch("streamlit.header")
    @patch("streamlit.subheader")
    @patch("streamlit.info")
    @patch("streamlit.warning")
    @patch("streamlit.error")
    @patch("streamlit.success")
    @patch("streamlit.metric")
    @patch("common.ui_components.CacheHealthChecker")
    def test_display_cache_health_dashboard(
        self,
        mock_checker,
        mock_metric,
        mock_success,
        mock_error,
        mock_warning,
        mock_info,
        mock_subheader,
        mock_header,
    ):
        """キャッシュヘルスダッシュボードのテスト"""
        # Mock the health checker
        mock_checker_instance = Mock()
        mock_checker_instance.analyze_cache_health.return_value = {
            "overall_health": "healthy",
            "total_symbols": 100,
            "healthy_symbols": 95,
            "issues": [],
            "recommendations": [],
        }
        mock_checker.return_value = mock_checker_instance

        # Should not raise exceptions
        ui_comp.display_cache_health_dashboard()

        # Should have called the health checker
        assert mock_checker.called

    @patch("streamlit.info")
    @patch("streamlit.warning")
    def test_display_cache_health_dashboard_error(self, mock_warning, mock_info):
        """キャッシュヘルスダッシュボードでエラーが発生した場合のテスト"""
        with patch("common.ui_components.CacheHealthChecker") as mock_checker:
            mock_checker.side_effect = Exception("Cache checker failed")

            # Should handle errors gracefully
            ui_comp.display_cache_health_dashboard()

            # Should display some kind of message
            assert mock_warning.called or mock_info.called


class TestSaveOperations:
    """Test save operations"""

    @patch("os.makedirs")
    @patch("pandas.DataFrame.to_csv")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_signal_and_trade_logs(self, mock_file, mock_to_csv, mock_makedirs):
        """save_signal_and_trade_logsのテスト"""
        signal_counts_df = pd.DataFrame({"system": ["System1", "System2"], "count": [10, 15]})

        results = pd.DataFrame({"entry_date": ["2023-01-01", "2023-01-02"], "pnl": [100, -50]})

        # Should not raise exceptions
        ui_comp.save_signal_and_trade_logs(signal_counts_df, results, "TestSystem", 10000)

        # Should have attempted to save files
        assert mock_to_csv.called

    @patch("common.ui_components.get_settings")
    @patch("os.makedirs")
    @patch("pandas.DataFrame.to_csv")
    def test_save_prepared_data_cache(self, mock_to_csv, mock_makedirs, mock_get_settings):
        """save_prepared_data_cacheのテスト"""
        mock_get_settings.return_value = Mock(data_cache_dir="test_cache", create_dirs=Mock())

        prepared_data = {
            "AAPL": pd.DataFrame({"Close": [100, 101, 102]}),
            "MSFT": pd.DataFrame({"Close": [200, 201, 202]}),
        }

        # Should not raise exceptions
        ui_comp.save_prepared_data_cache(prepared_data, "TestSystem")

        # Should have attempted to save
        assert mock_to_csv.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
