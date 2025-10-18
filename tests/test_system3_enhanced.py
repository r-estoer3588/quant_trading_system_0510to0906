"""
Enhanced tests for core.system3 module to improve test coverage
Focus on main functions with mock-based testing for System3 three-day pullback strategy
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from common.testing import set_test_determinism

# Import functions directly to avoid dependency issues
try:
    from core.system3 import (
        generate_candidates_system3,
        get_total_days_system3,
        prepare_data_vectorized_system3,
    )

    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


class TestSystem3MainFunctions:
    """Test System3 main functions with mock-based approaches"""

    def setup_method(self):
        set_test_determinism()
        if not IMPORTS_AVAILABLE:
            pytest.skip("core.system3 imports not available")

    def test_prepare_data_vectorized_system3_fast_path_success(self):
        """Test prepare_data_vectorized_system3 fast path with precomputed indicators"""
        mock_df = pd.DataFrame(
            {
                "Close": [100, 95, 90],  # 3-day decline
                "sma150": [98, 96, 94],
                "atr10": [2.0, 2.1, 2.2],
                "drop3d": [0.05, 0.08, 0.15],  # System3 uses drop3d
                "avgvolume50": [2_000_000, 2_100_000, 2_200_000],
            },
            index=pd.date_range("2023-01-01", periods=3),
        )

        raw_data = {"AAPL": mock_df}

        with (
            patch("core.system3.check_precomputed_indicators") as mock_check,
            patch("core.system3.process_symbols_batch") as mock_batch,
        ):
            mock_check.return_value = (raw_data, [])
            mock_batch.return_value = (raw_data, [])

            result = prepare_data_vectorized_system3(raw_data_dict=raw_data, symbols=["AAPL"], reuse_indicators=True)

        assert len(result) == 1
        assert "AAPL" in result

    def test_prepare_data_vectorized_system3_with_symbols_list(self):
        """Test prepare_data_vectorized_system3 with symbols parameter"""
        symbols = ["AAPL", "TSLA"]

        with patch("core.system3.process_symbols_batch") as mock_batch:
            mock_df = pd.DataFrame({"Close": [100], "filter": [True], "setup": [False]})
            mock_batch.return_value = ({"AAPL": mock_df}, [])

            result = prepare_data_vectorized_system3(raw_data_dict=None, symbols=symbols, reuse_indicators=False)

        assert "AAPL" in result
        mock_batch.assert_called_once()

    def test_generate_candidates_system3_success(self):
        """Test generate_candidates_system3 with valid setup data"""
        dates = pd.date_range("2023-01-01", periods=3)
        mock_df = pd.DataFrame(
            {
                "Close": [100, 95, 90],
                "Drop3D": [0.05, 0.10, 0.15],  # System3 ranks by Drop3D descending
                "ATR10": [2.0, 2.1, 2.2],
                "setup": [True, False, True],
            },
            index=dates,
        )

        prepared_dict = {"AAPL": mock_df, "TSLA": mock_df.copy()}

        candidates_by_date, candidates_df = generate_candidates_system3(prepared_dict, top_n=5)

        assert isinstance(candidates_by_date, dict)

    def test_generate_candidates_system3_empty_data(self):
        """Test generate_candidates_system3 with empty data"""
        candidates_by_date, candidates_df = generate_candidates_system3({})

        assert candidates_by_date == {}
        assert candidates_df is None

    def test_get_total_days_system3(self):
        """Test get_total_days_system3 function"""
        dates1 = pd.date_range("2023-01-01", periods=5)
        dates2 = pd.date_range("2023-01-03", periods=3)

        data_dict = {
            "AAPL": pd.DataFrame({"Close": range(5)}, index=dates1),
            "TSLA": pd.DataFrame({"Close": range(3)}, index=dates2),
        }

        total_days = get_total_days_system3(data_dict)

        expected_dates = len(set(dates1.union(dates2)))
        assert total_days == expected_dates

    def test_get_total_days_system3_empty_dict(self):
        """Test get_total_days_system3 with empty dict"""
        total_days = get_total_days_system3({})
        assert total_days == 0
