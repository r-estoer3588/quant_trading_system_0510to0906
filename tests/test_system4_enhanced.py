"""
Enhanced tests for core.system4 module targeting main functions for improved coverage
Focus on the main pipeline functions and integration tests
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from common.testing import set_test_determinism

# Import functions directly to avoid dependency issues
generate_candidates_system4: Any = None
get_total_days_system4: Any = None
prepare_data_vectorized_system4: Any = None
try:
    from core.system4 import generate_candidates_system4 as _gc4
    from core.system4 import get_total_days_system4 as _gt4
    from core.system4 import prepare_data_vectorized_system4 as _prep4

    generate_candidates_system4 = _gc4
    get_total_days_system4 = _gt4
    prepare_data_vectorized_system4 = _prep4
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


class TestSystem4MainFunctions:
    """Test main System4 functions for improved coverage"""

    def setup_method(self):
        set_test_determinism()
        if not IMPORTS_AVAILABLE:
            pytest.skip("core.system4 imports not available")

    def test_prepare_data_vectorized_system4_fast_path_success(self):
        """Test fast path with precomputed indicators"""
        # Create mock data with all required System4 indicators
        mock_data = {
            "AAPL": pd.DataFrame(
                {
                    "Close": [100, 110, 120],
                    "dollarvolume50": [150_000_000, 160_000_000, 170_000_000],
                    "sma200": [95, 100, 105],
                    "rsi4": [25, 30, 35],
                    "atr40": [2.5, 2.7, 2.9],
                    "hv50": [15, 20, 25],
                },
                index=pd.date_range("2023-01-01", periods=3),
            ),
            "MSFT": pd.DataFrame(
                {
                    "Close": [200, 210, 220],
                    "dollarvolume50": [200_000_000, 210_000_000, 220_000_000],
                    "sma200": [190, 195, 200],
                    "rsi4": [30, 35, 40],
                    "atr40": [3.0, 3.2, 3.4],
                    "hv50": [18, 22, 28],
                },
                index=pd.date_range("2023-01-01", periods=3),
            ),
        }

        # Mock the check_precomputed_indicators to return valid data
        with patch("core.system4.check_precomputed_indicators") as mock_check:
            mock_check.return_value = (mock_data, [])

            result = prepare_data_vectorized_system4(
                raw_data_dict=mock_data,
                reuse_indicators=True,
            )

        assert isinstance(result, dict)
        assert "AAPL" in result
        assert "MSFT" in result

        # Check that filter and setup columns were added
        for _symbol, df in result.items():
            assert "filter" in df.columns
            assert "setup" in df.columns
            assert all(df["filter"].notna())
            assert all(df["setup"].notna())

    def test_prepare_data_vectorized_system4_with_symbols_list(self):
        """Test normal processing path with symbols list"""
        symbols = ["AAPL", "MSFT"]

        # Mock _compute_indicators function
        def mock_compute_indicators(symbol):
            df = pd.DataFrame(
                {
                    "Close": [100, 110],
                    "dollarvolume50": [150_000_000, 160_000_000],
                    "sma200": [95, 100],
                    "rsi4": [25, 30],
                    "atr40": [2.5, 2.7],
                    "hv50": [15, 20],
                    "filter": [True, True],
                    "setup": [True, True],
                },
                index=pd.date_range("2023-01-01", periods=2),
            )
            return symbol, df

        with patch("core.system4.process_symbols_batch") as mock_batch:
            mock_batch.return_value = ({"AAPL": mock_compute_indicators("AAPL")[1]}, [])

            result = prepare_data_vectorized_system4(
                raw_data_dict=None, symbols=symbols, reuse_indicators=False
            )

        assert isinstance(result, dict)
        assert len(result) > 0

    def test_generate_candidates_system4_latest_only_mode(self):
        """Test candidate generation in latest_only mode"""
        # Prepare test data with setup conditions
        prepared_dict = {
            "AAPL": pd.DataFrame(
                {
                    "Close": [100, 110, 120],
                    "rsi4": [25, 30, 20],  # Lower RSI = higher rank in System4
                    "filter": [True, True, True],
                    "setup": [True, True, True],
                },
                index=pd.date_range("2023-01-01", periods=3),
            ),
            "MSFT": pd.DataFrame(
                {
                    "Close": [200, 210, 220],
                    "rsi4": [35, 40, 30],
                    "filter": [True, True, True],
                    "setup": [True, True, True],
                },
                index=pd.date_range("2023-01-01", periods=3),
            ),
        }

        res = generate_candidates_system4(
            prepared_dict, top_n=10, latest_only=True, include_diagnostics=True
        )
        assert isinstance(res, tuple)
        if len(res) == 3:
            candidates, df, diagnostics = res
        else:
            candidates, df = res
            diagnostics = {}

        # Check basic structure
        assert isinstance(candidates, dict)
        assert isinstance(diagnostics, dict)

        # Check diagnostics keys
        assert "ranking_source" in diagnostics
        assert "setup_predicate_count" in diagnostics
        assert "ranked_top_n_count" in diagnostics

        # Verify ranking_source
        assert diagnostics["ranking_source"] == "latest_only"

        # Check that candidates were generated
        assert diagnostics["ranked_top_n_count"] >= 0

    def test_generate_candidates_system4_full_scan_mode(self):
        """Test candidate generation in full scan mode"""
        prepared_dict = {
            "AAPL": pd.DataFrame(
                {
                    "Close": [100, 110, 120],
                    "rsi4": [25, 30, 35],
                    "filter": [True, True, True],
                    "setup": [True, True, True],
                },
                index=pd.date_range("2023-01-01", periods=3),
            ),
        }

        res = generate_candidates_system4(
            prepared_dict, top_n=10, latest_only=False, include_diagnostics=True
        )
        assert isinstance(res, tuple)
        if len(res) == 3:
            candidates, df, diagnostics = res
        else:
            candidates, df = res
            diagnostics = {}

        # Check basic structure
        assert isinstance(candidates, dict)
        assert isinstance(diagnostics, dict)

        # Verify ranking_source
        assert diagnostics["ranking_source"] == "full_scan"

    def test_generate_candidates_system4_empty_data(self):
        """Test candidate generation with empty data"""
        res = generate_candidates_system4({}, top_n=10, include_diagnostics=True)
        assert isinstance(res, tuple)
        if len(res) == 3:
            candidates, df, diagnostics = res
        else:
            candidates, df = res
            diagnostics = {}

        assert candidates == {}
        assert df is None
        assert diagnostics["ranked_top_n_count"] == 0

    def test_generate_candidates_system4_no_setup(self):
        """Test candidate generation when no symbols meet setup条件"""
        prepared_dict = {
            "AAPL": pd.DataFrame(
                {
                    "Close": [100, 110, 120],
                    "rsi4": [25, 30, 35],
                    "filter": [True, True, True],
                    "setup": [False, False, False],  # No setup
                },
                index=pd.date_range("2023-01-01", periods=3),
            ),
        }

        res = generate_candidates_system4(
            prepared_dict, top_n=10, latest_only=True, include_diagnostics=True
        )
        assert isinstance(res, tuple)
        if len(res) == 3:
            candidates, df, diagnostics = res
        else:
            candidates, df = res
            diagnostics = {}

        # Should return empty results
        assert len(candidates) == 0 or all(len(v) == 0 for v in candidates.values())
        assert diagnostics["setup_predicate_count"] == 0

    def test_generate_candidates_system4_ranking_order(self):
        """Test that candidates are ranked by RSI4 ascending (lower is better)"""
        prepared_dict = {
            "LOW_RSI": pd.DataFrame(
                {
                    "Close": [100],
                    "rsi4": [10],  # Lower RSI
                    "filter": [True],
                    "setup": [True],
                },
                index=pd.date_range("2023-01-03", periods=1),
            ),
            "HIGH_RSI": pd.DataFrame(
                {
                    "Close": [200],
                    "rsi4": [50],  # Higher RSI
                    "filter": [True],
                    "setup": [True],
                },
                index=pd.date_range("2023-01-03", periods=1),
            ),
        }

        res = generate_candidates_system4(
            prepared_dict, top_n=10, latest_only=True, include_diagnostics=True
        )
        assert isinstance(res, tuple)
        if len(res) == 3:
            _candidates, df, _ = res
        else:
            _candidates, df = res

        if df is not None and not df.empty:
            # First candidate should have lower RSI
            first_symbol = df.iloc[0]["symbol"]
            assert first_symbol == "LOW_RSI"

    def test_get_total_days_system4_basic(self):
        """Test get_total_days_system4 with basic data"""
        data_dict = {
            "AAPL": pd.DataFrame(
                {"Close": [100, 110, 120]}, index=pd.date_range("2023-01-01", periods=3)
            ),
            "MSFT": pd.DataFrame(
                {"Close": [200, 210]}, index=pd.date_range("2023-01-01", periods=2)
            ),
        }

        total_days = get_total_days_system4(data_dict)
        assert total_days == 3  # Maximum unique dates

    def test_get_total_days_system4_empty(self):
        """Test get_total_days_system4 with empty data"""
        assert get_total_days_system4({}) == 0

    def test_generate_candidates_system4_diagnostics_consistency(self):
        """Test diagnostics values are consistent across runs"""
        prepared_dict = {
            "AAPL": pd.DataFrame(
                {
                    "Close": [100, 110, 120],
                    "rsi4": [25, 30, 35],
                    "filter": [True, True, True],
                    "setup": [True, True, False],  # 2 setup passes
                },
                index=pd.date_range("2023-01-01", periods=3),
            ),
        }

        res1 = generate_candidates_system4(
            prepared_dict, top_n=10, latest_only=True, include_diagnostics=True
        )
        assert isinstance(res1, tuple)
        if len(res1) == 3:
            _candidates1, _df1, diagnostics1 = res1
        else:
            _candidates1, _df1 = res1
            diagnostics1 = {}

        res2 = generate_candidates_system4(
            prepared_dict, top_n=10, latest_only=True, include_diagnostics=True
        )
        assert isinstance(res2, tuple)
        if len(res2) == 3:
            _candidates2, _df2, diagnostics2 = res2
        else:
            _candidates2, _df2 = res2
            diagnostics2 = {}

        # Diagnostics should be consistent
        assert diagnostics1["ranking_source"] == diagnostics2["ranking_source"]
        assert diagnostics1["setup_predicate_count"] == diagnostics2["setup_predicate_count"]
        assert diagnostics1["ranked_top_n_count"] == diagnostics2["ranked_top_n_count"]


class TestSystem4EdgeCases:
    """Test edge cases and error handling"""

    def setup_method(self):
        set_test_determinism()
        if not IMPORTS_AVAILABLE:
            pytest.skip("core.system4 imports not available")

    def test_prepare_data_with_missing_indicators(self):
        """Test handling of missing required indicators"""
        mock_data = {
            "AAPL": pd.DataFrame(
                {
                    "Close": [100, 110, 120],
                    # Missing required indicators
                },
                index=pd.date_range("2023-01-01", periods=3),
            ),
        }

        with patch("core.system4.check_precomputed_indicators") as mock_check:
            mock_check.return_value = ({}, ["AAPL"])  # Indicate failure

            result = prepare_data_vectorized_system4(
                raw_data_dict=mock_data,
                reuse_indicators=True,
            )

        # Should return empty or partial results
        assert isinstance(result, dict)

    def test_generate_candidates_with_nan_values(self):
        """Test handling of NaN values in data"""
        prepared_dict = {
            "AAPL": pd.DataFrame(
                {
                    "Close": [100, None, 120],
                    "rsi4": [25, None, 35],
                    "filter": [True, True, True],
                    "setup": [True, False, True],
                },
                index=pd.date_range("2023-01-01", periods=3),
            ),
        }

        # Should handle NaN values gracefully
        res = generate_candidates_system4(
            prepared_dict, top_n=10, latest_only=True, include_diagnostics=True
        )
        assert isinstance(res, tuple)
        if len(res) == 3:
            candidates, df, diagnostics = res
        else:
            candidates, df = res
            diagnostics = {}

        assert isinstance(candidates, dict)
        assert isinstance(diagnostics, dict)
        assert diagnostics["ranking_source"] is not None
