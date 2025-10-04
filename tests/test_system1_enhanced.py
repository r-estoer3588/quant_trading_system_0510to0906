"""
Enhanced tests for core.system1 module targeting main functions for improved coverage
Focus on the main pipeline functions and integration tests
"""

from __future__ import annotations

import pandas as pd
import pytest
from unittest.mock import patch

from common.testing import set_test_determinism

# Import functions directly to avoid dependency issues
try:
    from core.system1 import (
        prepare_data_vectorized_system1,
        generate_candidates_system1,
        get_total_days_system1,
        summarize_system1_diagnostics,
    )

    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


class TestSystem1MainFunctions:
    """Test main System1 functions for improved coverage"""

    def setup_method(self):
        set_test_determinism()
        if not IMPORTS_AVAILABLE:
            pytest.skip("core.system1 imports not available")

    def test_prepare_data_vectorized_system1_fast_path_success(self):
        """Test fast path with precomputed indicators"""
        # Create mock data with all required indicators
        mock_data = {
            "AAPL": pd.DataFrame(
                {
                    "Close": [100, 110, 120],
                    "dollarvolume20": [30_000_000, 32_000_000, 35_000_000],
                    "sma200": [95, 100, 105],
                    "roc200": [5, 8, 12],
                },
                index=pd.date_range("2023-01-01", periods=3),
            ),
            "MSFT": pd.DataFrame(
                {
                    "Close": [200, 210, 220],
                    "dollarvolume20": [40_000_000, 42_000_000, 45_000_000],
                    "sma200": [190, 195, 200],
                    "roc200": [10, 15, 18],
                },
                index=pd.date_range("2023-01-01", periods=3),
            ),
        }

        # Mock the check_precomputed_indicators to return valid data
        with patch("core.system1.check_precomputed_indicators") as mock_check:
            mock_check.return_value = (mock_data, [])

            result = prepare_data_vectorized_system1(raw_data_dict=mock_data, reuse_indicators=True)

        assert isinstance(result, dict)
        assert "AAPL" in result
        assert "MSFT" in result

        # Check that filter and setup columns were added
        for _symbol, df in result.items():
            assert "filter" in df.columns
            assert "setup" in df.columns
            assert all(df["filter"].notna())
            assert all(df["setup"].notna())

    def test_prepare_data_vectorized_system1_with_symbols_list(self):
        """Test normal processing path with symbols list"""
        symbols = ["AAPL", "MSFT"]

        # Mock _compute_indicators function
        def mock_compute_indicators(symbol):
            df = pd.DataFrame(
                {
                    "Close": [100, 110],
                    "dollarvolume20": [30_000_000, 32_000_000],
                    "sma200": [95, 100],
                    "roc200": [5, 8],
                    "filter": [True, True],
                    "setup": [True, True],
                },
                index=pd.date_range("2023-01-01", periods=2),
            )
            return symbol, df

        with patch("core.system1.process_symbols_batch") as mock_batch:
            mock_batch.return_value = ({"AAPL": mock_compute_indicators("AAPL")[1]}, [])

            result = prepare_data_vectorized_system1(
                raw_data_dict=None, symbols=symbols, reuse_indicators=False
            )

        assert isinstance(result, dict)
        assert "AAPL" in result

    def test_prepare_data_vectorized_system1_empty_input(self):
        """Test behavior with empty input"""
        result = prepare_data_vectorized_system1(raw_data_dict=None, symbols=None)
        assert result == {}

    def test_generate_candidates_system1_success(self):
        """Test candidate generation with valid data"""
        # Create mock prepared data
        prepared_data = {
            "AAPL": pd.DataFrame(
                {
                    "Close": [100, 110, 120],
                    "roc200": [5, 8, 12],
                    "setup": [True, True, True],
                },
                index=pd.date_range("2023-01-01", periods=3),
            ),
            "MSFT": pd.DataFrame(
                {
                    "Close": [200, 210, 220],
                    "roc200": [10, 15, 18],
                    "setup": [True, True, True],
                },
                index=pd.date_range("2023-01-01", periods=3),
            ),
        }

        candidates_by_date, candidates_df, diagnostics = generate_candidates_system1(
            prepared_data, top_n=10
        )

        assert isinstance(candidates_by_date, dict)
        assert len(candidates_by_date) > 0

        if candidates_df is not None:
            assert isinstance(candidates_df, pd.DataFrame)
            assert "symbol" in candidates_df.columns
            assert "roc200" in candidates_df.columns
            assert "close" in candidates_df.columns
            assert "date" in candidates_df.columns

    def test_generate_candidates_system1_empty_data(self):
        """Test candidate generation with empty data"""
        candidates_by_date, candidates_df, diagnostics = generate_candidates_system1({})
        assert candidates_by_date == {}
        assert candidates_df is None
        assert isinstance(diagnostics, dict)

    def test_generate_candidates_system1_no_setup_conditions(self):
        """Test candidate generation when no setup conditions are met"""
        prepared_data = {
            "AAPL": pd.DataFrame(
                {
                    "Close": [100, 110, 120],
                    "roc200": [5, 8, 12],
                    "setup": [False, False, False],  # No setups
                },
                index=pd.date_range("2023-01-01", periods=3),
            )
        }

        candidates_by_date, candidates_df, _ = generate_candidates_system1(prepared_data, top_n=10)

        assert isinstance(candidates_by_date, dict)
        # Should still return dict structure even if no candidates
        assert candidates_df is None or candidates_df.empty

    def test_get_total_days_system1(self):
        """Test get_total_days_system1 function"""
        data_dict = {
            "AAPL": pd.DataFrame(
                {"Close": [100, 110, 120, 130, 140]}, index=pd.date_range("2023-01-01", periods=5)
            ),
            "MSFT": pd.DataFrame(
                {"Close": [200, 210, 220]}, index=pd.date_range("2023-01-01", periods=3)
            ),
        }

        total_days = get_total_days_system1(data_dict)
        assert total_days == 5  # Maximum length

    def test_get_total_days_system1_empty_dict(self):
        """Test get_total_days_system1 with empty dict"""
        total_days = get_total_days_system1({})
        assert total_days == 0

    @pytest.mark.skip(reason="_compute_indicators is internal implementation")
    def test_compute_indicators_success(self):
        """Test _compute_indicators with valid cached data"""
        # This tests internal implementation details - skip for now
        pass

    @pytest.mark.skip(reason="_compute_indicators is internal implementation")
    def test_compute_indicators_missing_data(self):
        """Test _compute_indicators when cache returns None"""
        pass

    @pytest.mark.skip(reason="_compute_indicators is internal implementation")
    def test_compute_indicators_missing_indicators(self):
        """Test _compute_indicators when required indicators are missing"""
        pass

    @pytest.mark.skip(reason="_compute_indicators is internal implementation")
    def test_compute_indicators_exception_handling(self):
        """Test _compute_indicators exception handling"""
        pass

    def test_prepare_data_vectorized_system1_with_callbacks(self):
        """Test prepare_data_vectorized_system1 with callbacks"""
        symbols = ["AAPL"]
        progress_calls = []
        log_calls = []

        def progress_callback(msg):
            progress_calls.append(msg)

        def log_callback(msg):
            log_calls.append(msg)

        with patch("core.system1.process_symbols_batch") as mock_batch:
            mock_batch.return_value = ({}, [])

            prepare_data_vectorized_system1(
                raw_data_dict=None,
                symbols=symbols,
                progress_callback=progress_callback,
                log_callback=log_callback,
                reuse_indicators=False,
            )

        assert len(log_calls) > 0

    def test_generate_candidates_system1_with_callbacks(self):
        """Test generate_candidates_system1 with callbacks and progress"""
        if not IMPORTS_AVAILABLE:
            pytest.skip("core.system1 imports not available")

        prepared_data = {
            "AAPL": pd.DataFrame(
                {
                    "Close": [100],
                    "roc200": [5],
                    "setup": [True],
                },
                index=pd.date_range("2023-01-01", periods=1),
            )
        }

        progress_calls = []
        log_calls = []

        def progress_callback(msg):
            progress_calls.append(msg)

        def log_callback(msg):
            log_calls.append(msg)

        result = generate_candidates_system1(  # type: ignore[name-defined]
            prepared_data, top_n=10, progress_callback=progress_callback, log_callback=log_callback
        )

        assert isinstance(result, tuple)
        assert len(result) == 3

        assert len(log_calls) > 0

    def test_summarize_system1_diagnostics_basic(self):
        """Diagnostics helper should normalize counters and reasons."""

        if not IMPORTS_AVAILABLE:
            pytest.skip("core.system1 imports not available")

        diag = {
            "filter_pass": "12",
            "setup_flag_true": 7.0,
            "fallback_pass": None,
            "roc200_positive": 9,
            "final_pass": "0",
            "top_n": 10,
            "exclude_reasons": {"setup": 5, "roc200": 3, "filter": 1},
        }

        summary = summarize_system1_diagnostics(diag)

        assert summary["filter_pass"] == 12
        assert summary["setup_flag_true"] == 7
        assert summary["fallback_pass"] == 0
        assert summary["roc200_positive"] == 9
        assert summary["final_pass"] == 0
        assert summary.get("top_n") == 10
        assert summary.get("exclude_reasons") == {
            "setup": 5,
            "roc200": 3,
            "filter": 1,
        }


if __name__ == "__main__":
    pytest.main([__file__])
