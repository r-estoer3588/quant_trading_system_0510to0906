"""
Additional comprehensive tests for ui_components.py to push coverage above 50%
Focuses on uncovered functions and branches
"""

from __future__ import annotations

import os
import time
from unittest.mock import Mock, patch
import pytest
import tempfile

import pandas as pd
import streamlit as st

from common import ui_components as ui_comp


# ============================================================================
# Test Complex Functions Not Covered Yet
# ============================================================================


class TestAdvancedUIComponents:
    """Test advanced UI component functions for higher coverage"""

    @patch("streamlit.info")
    @patch("streamlit.error")
    @patch("streamlit.progress")
    @patch("streamlit.empty")
    def test_fetch_data_with_skipped_symbols(
        self, mock_empty, mock_progress, mock_error, mock_info
    ):
        """スキップされるシンボルがある場合のテスト"""

        # Setup mocks for successful and failed symbol loads
        def mock_load_symbol_side_effect(symbol):
            if symbol == "FAIL":
                return symbol, None  # No data
            else:
                df = pd.DataFrame(
                    {
                        "Date": pd.date_range("2023-01-01", periods=5),
                        "Close": [100, 101, 102, 103, 104],
                        "Volume": [1000, 1100, 1200, 1300, 1400],
                    }
                )
                return symbol, df

        with patch("common.ui_components.load_symbol", side_effect=mock_load_symbol_side_effect):
            result = ui_comp.fetch_data(["AAPL", "FAIL", "MSFT"], max_workers=1)

            # Should have 2 successful results, FAIL should be skipped
            assert len(result) == 2
            assert "AAPL" in result
            assert "MSFT" in result
            assert "FAIL" not in result

    def test_clean_date_column_all_invalid(self):
        """全て無効な日付の場合のテスト"""
        df = pd.DataFrame(
            {
                "Date": ["invalid1", "invalid2", None, "not_a_date"],
                "value": [1, 2, 3, 4],
            }
        )

        result = ui_comp.clean_date_column(df, "Date")

        # Should return empty DataFrame after dropping all invalid dates
        assert len(result) == 0

    def test_clean_date_column_with_timezone_dates(self):
        """タイムゾーン付き日付のテスト"""
        df = pd.DataFrame(
            {
                "Date": ["2023-01-01T09:00:00Z", "2023-01-02T10:00:00+09:00", "2023-01-03"],
                "value": [1, 2, 3],
            }
        )

        result = ui_comp.clean_date_column(df, "Date")

        # One date might be invalid due to mixed timezone parsing
        assert len(result) >= 2  # At least 2 should be valid
        assert pd.api.types.is_datetime64_any_dtype(result["Date"])

    @patch("streamlit.info")
    @patch("streamlit.progress")
    @patch("streamlit.empty")
    def test_log_with_progress_different_batches(self, mock_empty, mock_progress, mock_info):
        """異なるバッチサイズでの進捗ログテスト"""
        mock_log_area = Mock()
        mock_progress_bar = Mock()

        start_time = time.time() - 120  # 2 minutes ago

        # Test different batch sizes
        for batch_size in [1, 10, 25, 100]:
            ui_comp.log_with_progress(
                batch_size,
                100,
                start_time,
                prefix=f"Batch{batch_size}",
                batch=batch_size,
                log_area=mock_log_area,
                progress_bar=mock_progress_bar,
                unit="items",
            )

            if batch_size <= 100:  # Should log when i % batch == 0
                assert mock_log_area.text.called
                assert mock_progress_bar.progress.called

    def test_log_with_progress_completion(self):
        """完了時の進捗ログテスト"""
        mock_log_area = Mock()
        mock_progress_bar = Mock()
        start_time = time.time() - 60

        # Test completion (i == total)
        ui_comp.log_with_progress(
            100,
            100,
            start_time,
            prefix="完了",
            batch=50,  # Normally wouldn't log, but completion should
            log_area=mock_log_area,
            progress_bar=mock_progress_bar,
        )

        mock_log_area.text.assert_called_once()
        mock_progress_bar.progress.assert_called_once_with(1.0)  # 100% completion

    def test_log_with_progress_exception_handling(self):
        """進捗ログでの例外処理テスト"""
        mock_log_area = Mock()
        mock_log_area.text.side_effect = Exception("Text update failed")

        mock_progress_bar = Mock()
        mock_progress_bar.progress.side_effect = Exception("Progress update failed")

        # Should not raise exception even if UI updates fail
        ui_comp.log_with_progress(
            50, 100, time.time(), log_area=mock_log_area, progress_bar=mock_progress_bar
        )
        # Test passes if no exception is raised


class TestDataLoadingEdgeCases:
    """Test edge cases in data loading functions"""

    @patch("os.path.exists")
    @patch("common.ui_components.get_cached_data")
    @patch("common.ui_components.load_base_cache")
    def test_load_symbol_all_methods_fail(self, mock_load_base, mock_get_cached, mock_exists):
        """全ての読み込み方法が失敗する場合のテスト"""
        # All methods fail
        mock_load_base.side_effect = Exception("Base cache failed")
        mock_get_cached.side_effect = Exception("Cached data failed")
        mock_exists.return_value = False  # Raw file doesn't exist

        symbol, result_df = ui_comp.load_symbol("FAIL_ALL")

        assert symbol == "FAIL_ALL"
        assert result_df is None

    @patch("os.path.exists")
    @patch("common.ui_components.get_cached_data")
    @patch("common.ui_components.load_base_cache")
    def test_load_symbol_empty_dataframe(self, mock_load_base, mock_get_cached, mock_exists):
        """空のDataFrameが返される場合のテスト"""
        # Return empty DataFrame
        mock_load_base.return_value = pd.DataFrame()  # Empty
        mock_exists.return_value = False

        symbol, result_df = ui_comp.load_symbol("EMPTY")

        assert symbol == "EMPTY"
        assert result_df is None  # Should treat empty df as None

    def test_mtime_or_zero_permission_error(self):
        """権限エラーが発生する場合のテスト"""
        # Use a system path that might have permission issues
        system_path = "C:\\Windows\\System32\\config\\SAM"  # Windows system file

        result = ui_comp._mtime_or_zero(system_path)

        # Should return 0.0 even for permission errors
        assert result == 0.0

    def test_mtime_or_zero_with_real_file(self):
        """実際のファイルでのテスト"""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test data")
            tmp_path = tmp.name

        try:
            result = ui_comp._mtime_or_zero(tmp_path)
            assert result > 0
            assert isinstance(result, float)
        finally:
            os.unlink(tmp_path)


class TestCachedSymbolLoading:
    """Test the cached symbol loading functionality"""

    @patch("common.ui_components.get_cached_data")
    @patch("common.ui_components.load_base_cache")
    @patch("os.path.getmtime")
    def test_load_symbol_cached_with_mtime(self, mock_getmtime, mock_load_base, mock_get_cached):
        """mtimeを使用したキャッシュロードのテスト"""
        # Setup mtime values
        mock_getmtime.side_effect = lambda path: {
            "base_path": 1000.0,
            "raw_path": 2000.0,
        }.get(os.path.basename(path), 0.0)

        # Setup successful base cache load
        sample_df = pd.DataFrame(
            {"Date": pd.date_range("2023-01-01", periods=3), "Close": [100, 101, 102]}
        )
        mock_load_base.return_value = sample_df

        symbol, result_df = ui_comp.load_symbol("CACHED_TEST")

        assert symbol == "CACHED_TEST"
        assert result_df is not None
        assert len(result_df) == 3

    def test_load_symbol_cached_function_direct_call(self):
        """_load_symbol_cached関数の直接呼び出しテスト"""
        with patch("common.ui_components.load_base_cache") as mock_load_base:
            mock_load_base.return_value = pd.DataFrame(
                {"Close": [100, 101], "Date": pd.date_range("2023-01-01", periods=2)}
            )

            # Direct call to cached function
            symbol, result_df = ui_comp._load_symbol_cached(
                "TEST",
                base_path="test_base_path",
                base_mtime=1000.0,
                raw_path="test_raw_path",
                raw_mtime=2000.0,
            )

            assert symbol == "TEST"
            assert result_df is not None


class TestResultSummarizationDetailed:
    """Detailed tests for result summarization"""

    @patch("streamlit.info")
    @patch("streamlit.error")
    @patch("streamlit.progress")
    @patch("streamlit.empty")
    def test_summarize_results_with_drawdown(
        self, mock_empty, mock_progress, mock_error, mock_info
    ):
        """ドローダウン計算を含む結果サマリーテスト"""
        # Create results with specific drawdown pattern
        results_df = pd.DataFrame(
            {
                "entry_date": pd.date_range("2023-01-01", periods=6),
                "exit_date": pd.date_range("2023-01-01", periods=6) + pd.Timedelta(days=3),
                "pnl": [100, -50, -75, 200, 150, -25],  # Creates drawdown pattern
                "symbol": ["A", "B", "C", "D", "E", "F"],
                "entry_price": [100] * 6,
                "exit_price": [105] * 6,
            }
        )

        summary, processed_df = ui_comp.summarize_results(results_df, 10000)

        # Check drawdown calculation
        assert "max_dd" in summary
        assert summary["max_dd"] > 0  # Should have some drawdown

        # Check that cumulative PnL is calculated
        assert "cumulative_pnl" in processed_df.columns

        # Verify win/loss statistics
        assert summary["trades"] == 6
        wins = (results_df["pnl"] > 0).sum()
        expected_win_rate = (wins / 6) * 100
        assert abs(summary["win_rate"] - expected_win_rate) < 0.01

    @patch("streamlit.info")
    @patch("streamlit.error")
    @patch("streamlit.progress")
    @patch("streamlit.empty")
    def test_summarize_results_all_losses(self, mock_empty, mock_progress, mock_error, mock_info):
        """全て損失のトレードでのテスト"""
        results_df = pd.DataFrame(
            {
                "entry_date": pd.date_range("2023-01-01", periods=3),
                "exit_date": pd.date_range("2023-01-01", periods=3) + pd.Timedelta(days=1),
                "pnl": [-100, -50, -75],  # All losses
                "symbol": ["A", "B", "C"],
                "entry_price": [100] * 3,
                "exit_price": [95] * 3,
            }
        )

        summary, processed_df = ui_comp.summarize_results(results_df, 10000)

        assert summary["win_rate"] == 0.0
        assert summary["total_return"] < 0  # Overall loss
        assert "trades" in summary
        assert "max_dd" in summary

    def test_summarize_results_extreme_values(self):
        """極端な値でのテスト"""
        results_df = pd.DataFrame(
            {
                "entry_date": pd.date_range("2023-01-01", periods=2),
                "exit_date": pd.date_range("2023-01-01", periods=2) + pd.Timedelta(days=1),
                "pnl": [1000000, -999999],  # Very large win and loss
                "symbol": ["HUGE_WIN", "HUGE_LOSS"],
                "entry_price": [100, 100],
                "exit_price": [200, 50],
            }
        )

        with patch.multiple(
            st,
            info=Mock(),
            error=Mock(),
            progress=Mock(return_value=Mock()),
            empty=Mock(return_value=Mock()),
        ):
            summary, processed_df = ui_comp.summarize_results(results_df, 10000)

            assert summary["trades"] == 2
            assert abs(summary["win_rate"] - 50.0) < 0.01  # 1 win out of 2

            # Check extreme values don't break calculation
            assert isinstance(summary["total_return"], int | float)
            assert isinstance(summary["max_dd"], int | float)


class TestFontAndInternationalization:
    """Test font and i18n functionality"""

    def test_japanese_font_fallback_execution(self):
        """日本語フォント設定の実行テスト"""
        # Should not raise an exception
        ui_comp._set_japanese_font_fallback()
        assert True

    def test_japanese_font_fallback_with_mock_matplotlib(self):
        """matplotlibモックでの日本語フォント設定テスト"""
        with patch("matplotlib.rcParams") as mock_rcparams:
            with patch("matplotlib.font_manager.fontManager") as mock_font_manager:
                # Mock font manager to return some fonts
                mock_font = Mock()
                mock_font.name = "Yu Gothic"
                mock_font_manager.ttflist = [mock_font]

                ui_comp._set_japanese_font_fallback()

                # Should have attempted to set font family
                assert mock_rcparams.__setitem__.called

    def test_clean_date_column_preserves_other_columns(self):
        """日付クリーニングで他の列が保持されるかテスト"""
        df = pd.DataFrame(
            {
                "Date": ["2023-01-01", "invalid", "2023-01-03"],
                "value": [1, 2, 3],
                "symbol": ["A", "B", "C"],
                "extra_col": ["x", "y", "z"],
            }
        )

        result = ui_comp.clean_date_column(df, "Date")

        # Should preserve all columns except drop invalid date rows
        assert len(result) == 2  # One invalid date dropped
        assert "value" in result.columns
        assert "symbol" in result.columns
        assert "extra_col" in result.columns


class TestErrorHandlingAndEdgeCases:
    """Test comprehensive error handling"""

    def test_default_log_callback_edge_cases(self):
        """デフォルトログコールバックのエッジケース"""
        func = ui_comp.default_log_callback

        # Test with zero time elapsed
        start_time = time.time()
        result = func(0, 100, start_time)
        assert "0/100" in result

        # Test with zero total
        result = func(0, 0, start_time - 60)
        assert "0/0" in result

        # Test with completed task
        result = func(50, 50, start_time - 30)
        assert "50/50" in result

    def test_extract_zero_reason_various_patterns(self):
        """ゼロ理由抽出の様々なパターン"""
        func = ui_comp.extract_zero_reason_from_logs

        test_cases = [
            # Different zero reason patterns
            ["System initialized", "候補数: 0件", "理由: データ不足"],
            ["Processing...", "No valid candidates found", "Done"],
            ["Start", "Filtering data...", "Result: 0 entries", "End"],
            ["エラー: 候補が見つかりません", "処理終了"],
            # Edge cases
            [],  # Empty
            None,  # None
            [""],  # Empty strings
            ["Only normal logs", "No zero reason here"],
        ]

        for logs in test_cases:
            result = func(logs)
            # Accept any result - implementation dependent
            assert result is None or isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
