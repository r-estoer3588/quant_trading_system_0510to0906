"""
Focused tests for specific uncovered functions in ui_components.py
Targeting 50%+ coverage by testing show_results, signal processing, and display functions
"""

from __future__ import annotations

from unittest.mock import Mock, patch, MagicMock
import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from common import ui_components as ui_comp


class TestShowResultsFocused:
    """Focused tests for show_results function to boost coverage"""

    def create_complete_results_df(self):
        """Create complete results DataFrame with all expected columns"""
        return pd.DataFrame(
            {
                "entry_date": pd.to_datetime(["2023-01-01", "2023-01-05", "2023-01-10"]),
                "exit_date": pd.to_datetime(["2023-01-03", "2023-01-08", "2023-01-15"]),
                "pnl": [100.0, -50.0, 75.0],
                "symbol": ["AAPL", "MSFT", "GOOGL"],
                "entry_price": [150.0, 200.0, 2500.0],
                "exit_price": [160.0, 190.0, 2575.0],
                "cumulative_pnl": [100.0, 50.0, 125.0],
                "returns": [0.067, -0.05, 0.03],
                "trade_duration": [3, 3, 5],
            }
        )

    @patch("streamlit.subheader")
    @patch("streamlit.metric")
    @patch("streamlit.dataframe")
    @patch("streamlit.plotly_chart")
    @patch("streamlit.columns")
    def test_show_results_complete_data(
        self, mock_columns, mock_plotly, mock_dataframe, mock_metric, mock_subheader
    ):
        """完全なデータでのshow_results関数テスト"""
        results_df = self.create_complete_results_df()
        summary = {
            "trades": 3,
            "total_return": 125.0,
            "win_rate": 66.7,
            "max_dd": 50.0,
        }

        # Mock columns to return mock column objects
        mock_col1, mock_col2, mock_col3 = Mock(), Mock(), Mock()
        mock_columns.return_value = [mock_col1, mock_col2, mock_col3]

        # Execute the function
        ui_comp.show_results(results_df, summary, 10000)

        # Verify UI components were called
        assert mock_subheader.called
        assert mock_metric.called
        assert mock_dataframe.called


class TestSignalAndTradeProcessing:
    """Test signal and trade processing functions"""

    @patch("streamlit.info")
    def test_show_signal_trade_summary_basic(self, mock_info):
        """show_signal_trade_summaryの基本テスト"""
        signal_counts_df = pd.DataFrame(
            {
                "system": ["System1", "System2", "System3"],
                "count": [5, 3, 7],
                "date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-01"]),
            }
        )

        results = pd.DataFrame(
            {
                "entry_date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                "exit_date": pd.to_datetime(["2023-01-05", "2023-01-07"]),
                "pnl": [100, -50],
                "symbol": ["AAPL", "MSFT"],
            }
        )

        # Should not raise exceptions
        ui_comp.show_signal_trade_summary(signal_counts_df, results, "TestSystem", 10000)

        # Should display some information
        assert mock_info.called

    @patch("streamlit.dataframe")
    def test_display_roc200_ranking(self, mock_dataframe):
        """display_roc200_ranking関数のテスト"""
        data = {
            "AAPL": pd.DataFrame(
                {
                    "Date": pd.date_range("2023-01-01", periods=5),
                    "Close": [150, 151, 149, 152, 153],
                    "roc_200": [0.05, 0.03, -0.01, 0.04, 0.06],
                    "symbol": ["AAPL"] * 5,
                }
            ),
            "MSFT": pd.DataFrame(
                {
                    "Date": pd.date_range("2023-01-01", periods=5),
                    "Close": [250, 248, 251, 252, 255],
                    "roc_200": [0.02, -0.01, 0.03, 0.04, 0.07],
                    "symbol": ["MSFT"] * 5,
                }
            ),
        }

        # Should not raise exceptions
        ui_comp.display_roc200_ranking(data, num_stocks=2)

        # Should display dataframe
        assert mock_dataframe.called


class TestCacheAndSystemDisplays:
    """Test cache and system display functions"""

    @patch("streamlit.subheader")
    @patch("streamlit.metric")
    @patch("streamlit.warning")
    @patch("streamlit.info")
    @patch("common.ui_components.get_settings")
    def test_display_system_cache_coverage(
        self, mock_get_settings, mock_info, mock_warning, mock_metric, mock_subheader
    ):
        """display_system_cache_coverage関数のテスト"""
        # Mock settings
        mock_settings = Mock()
        mock_settings.data_cache_dir = "test_cache"
        mock_get_settings.return_value = mock_settings

        with patch("os.path.exists", return_value=True):
            with patch("os.listdir", return_value=["system1.csv", "system2.csv"]):
                with patch(
                    "pandas.read_csv", return_value=pd.DataFrame({"symbol": ["AAPL", "MSFT"]})
                ):
                    # Should not raise exceptions
                    ui_comp.display_system_cache_coverage()

                    # Should display some metrics
                    assert mock_subheader.called or mock_info.called


class TestDataPreparationFunctions:
    """Test data preparation and processing functions"""

    def create_test_data_dict(self):
        """Create test data dictionary"""
        return {
            "AAPL": pd.DataFrame(
                {
                    "Date": pd.date_range("2023-01-01", periods=100),
                    "Close": 150 + np.random.randn(100) * 5,
                    "Volume": 1000000 + np.random.randn(100) * 100000,
                    "RSI": 30 + np.random.randn(100) * 20,
                    "EMA12": 150 + np.random.randn(100) * 3,
                    "EMA26": 150 + np.random.randn(100) * 3,
                }
            ),
            "MSFT": pd.DataFrame(
                {
                    "Date": pd.date_range("2023-01-01", periods=100),
                    "Close": 250 + np.random.randn(100) * 8,
                    "Volume": 800000 + np.random.randn(100) * 80000,
                    "RSI": 40 + np.random.randn(100) * 25,
                }
            ),
        }

    @patch("streamlit.info")
    @patch("streamlit.progress")
    @patch("streamlit.empty")
    def test_prepare_backtest_data_with_volume_filter(self, mock_empty, mock_progress, mock_info):
        """ボリュームフィルタ付きのprepare_backtest_dataテスト"""
        data = self.create_test_data_dict()

        # Mock progress bar and log area
        mock_progress_bar = Mock()
        mock_log_area = Mock()
        mock_empty.return_value = mock_log_area
        mock_progress.return_value = mock_progress_bar

        result = ui_comp.prepare_backtest_data(
            data,
            start_date="2023-01-01",
            end_date="2023-12-31",
            min_volume=500000,  # Filter by volume
            log_area=mock_log_area,
            progress_bar=mock_progress_bar,
        )

        # Should return filtered data
        assert isinstance(result, dict)
        # Volume filter should be applied (implementation dependent)

    @patch("streamlit.success")
    @patch("streamlit.info")
    def test_prepare_backtest_data_date_filtering(self, mock_info, mock_success):
        """日付フィルタリングのテスト"""
        data = self.create_test_data_dict()

        # Test with specific date range
        result = ui_comp.prepare_backtest_data(
            data,
            start_date="2023-06-01",  # Mid-year start
            end_date="2023-08-31",  # Mid-year end
            min_volume=0,  # No volume filter
        )

        # Should return data within date range
        assert isinstance(result, dict)
        for symbol, df in result.items():
            if len(df) > 0:
                assert df["Date"].min() >= pd.to_datetime("2023-06-01")
                assert df["Date"].max() <= pd.to_datetime("2023-08-31")


class TestUtilityAndHelperFunctions:
    """Test utility and helper functions"""

    def test_extract_zero_reason_from_logs_comprehensive(self):
        """extract_zero_reason_from_logsの包括的テスト"""
        test_cases = [
            # Case 1: Found zero reason
            (["処理開始", "候補数: 0件", "理由: データ不十分"], "理由: データ不十分"),
            # Case 2: Different pattern
            (
                ["Starting analysis", "Found 0 entries", "Reason: No valid data"],
                "Reason: No valid data",
            ),
            # Case 3: No zero reason found
            (["正常処理", "候補数: 5件", "処理完了"], None),
            # Case 4: Empty logs
            ([], None),
            # Case 5: None input
            (None, None),
        ]

        for logs, expected in test_cases:
            result = ui_comp.extract_zero_reason_from_logs(logs)
            if expected is None:
                assert result is None or result == ""
            else:
                assert expected in str(result) if result else False

    @patch("builtins.open")
    @patch("os.makedirs")
    def test_save_prepared_data_cache_comprehensive(self, mock_makedirs, mock_open):
        """save_prepared_data_cacheの包括的テスト"""
        prepared_data = {
            "AAPL": pd.DataFrame(
                {
                    "Date": pd.date_range("2023-01-01", periods=10),
                    "Close": [150] * 10,
                    "Volume": [1000000] * 10,
                }
            ),
            "MSFT": pd.DataFrame(
                {
                    "Date": pd.date_range("2023-01-01", periods=10),
                    "Close": [250] * 10,
                    "Volume": [800000] * 10,
                }
            ),
        }

        with patch("common.ui_components.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.data_cache_dir = "test_cache"
            mock_get_settings.return_value = mock_settings

            with patch("pandas.DataFrame.to_csv") as mock_to_csv:
                # Should not raise exceptions
                ui_comp.save_prepared_data_cache(prepared_data, "TestSystem")

                # Should have attempted to save files
                assert mock_to_csv.called


class TestDisplayAnalysisResults:
    """Test display analysis result functions"""

    @patch("streamlit.success")
    @patch("streamlit.warning")
    @patch("streamlit.error")
    @patch("streamlit.metric")
    @patch("streamlit.dataframe")
    @patch("streamlit.write")
    def test_display_cache_analysis_results(
        self, mock_write, mock_dataframe, mock_metric, mock_error, mock_warning, mock_success
    ):
        """_display_cache_analysis_results関数のテスト"""
        # Test healthy result
        healthy_result = {
            "overall_health": "healthy",
            "total_symbols": 100,
            "healthy_symbols": 95,
            "unhealthy_symbols": 5,
            "issues": ["Minor issue 1", "Minor issue 2"],
            "recommendations": ["Recommendation 1"],
        }

        ui_comp._display_cache_analysis_results(healthy_result)

        # Should display success status for healthy cache
        assert mock_success.called or mock_write.called

        # Test unhealthy result
        unhealthy_result = {
            "overall_health": "unhealthy",
            "total_symbols": 100,
            "healthy_symbols": 60,
            "unhealthy_symbols": 40,
            "issues": ["Major issue 1", "Major issue 2", "Critical issue"],
            "recommendations": ["Fix recommendation 1", "Fix recommendation 2"],
        }

        ui_comp._display_cache_analysis_results(unhealthy_result)

        # Should display warning or error for unhealthy cache
        assert mock_warning.called or mock_error.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
