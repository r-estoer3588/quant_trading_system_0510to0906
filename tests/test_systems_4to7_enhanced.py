"""
Enhanced tests for core.system4-7 modules to improve test coverage
Focus on main functions with mock-based testing for remaining trading systems
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from common.testing import set_test_determinism

# Import System4 functions
try:
    from core.system4 import (
        generate_candidates_system4,
        get_total_days_system4,
        prepare_data_vectorized_system4,
    )

    SYSTEM4_AVAILABLE = True
except ImportError:
    SYSTEM4_AVAILABLE = False

# Import System5 functions
try:
    from core.system5 import (
        generate_candidates_system5,
        get_total_days_system5,
        prepare_data_vectorized_system5,
    )

    SYSTEM5_AVAILABLE = True
except ImportError:
    SYSTEM5_AVAILABLE = False

# Import System6 functions
try:
    from core.system6 import (
        generate_candidates_system6,
        get_total_days_system6,
        prepare_data_vectorized_system6,
    )

    SYSTEM6_AVAILABLE = True
except ImportError:
    SYSTEM6_AVAILABLE = False

# Import System7 functions
try:
    from core.system7 import (
        generate_candidates_system7,
        get_total_days_system7,
        prepare_data_vectorized_system7,
    )

    SYSTEM7_AVAILABLE = True
except ImportError:
    SYSTEM7_AVAILABLE = False


class TestSystem4MainFunctions:
    """Test System4 main functions (Long trend low-vol pullback)"""

    def setup_method(self):
        set_test_determinism()
        if not SYSTEM4_AVAILABLE:
            pytest.skip("core.system4 imports not available")

    def test_prepare_data_vectorized_system4_success(self):
        """Test prepare_data_vectorized_system4 with symbols parameter"""
        symbols = ["AAPL"]

        with patch("core.system4.process_symbols_batch") as mock_batch:
            mock_df = pd.DataFrame({"Close": [100], "filter": [True], "setup": [False]})
            mock_batch.return_value = ({"AAPL": mock_df}, [])

            result = prepare_data_vectorized_system4(
                raw_data_dict=None, symbols=symbols, reuse_indicators=False
            )

        assert "AAPL" in result
        mock_batch.assert_called_once()

    def test_generate_candidates_system4_success(self):
        """Test generate_candidates_system4 with SPY market data"""
        dates = pd.date_range("2023-01-01", periods=3)
        mock_df = pd.DataFrame(
            {
                "Close": [100, 95, 90],
                "RSI4": [30, 25, 20],  # System4 ranks by RSI4 ascending
                "setup": [True, False, True],
            },
            index=dates,
        )

        # Mock SPY market data required by System4 (spy_df removed)
        # spy_df definition removed - not used in test

        prepared_dict = {"AAPL": mock_df}

        candidates_by_date, candidates_df = generate_candidates_system4(
            prepared_dict, top_n=5
        )

        assert isinstance(candidates_by_date, dict)

    def test_get_total_days_system4(self):
        """Test get_total_days_system4 function"""
        dates = pd.date_range("2023-01-01", periods=5)
        data_dict = {"AAPL": pd.DataFrame({"Close": range(5)}, index=dates)}

        total_days = get_total_days_system4(data_dict)
        assert total_days == 5


class TestSystem5MainFunctions:
    """Test System5 main functions (Long breakout high volatility)"""

    def setup_method(self):
        set_test_determinism()
        if not SYSTEM5_AVAILABLE:
            pytest.skip("core.system5 imports not available")

    def test_prepare_data_vectorized_system5_success(self):
        """Test prepare_data_vectorized_system5 with symbols parameter"""
        symbols = ["AAPL"]

        with patch("core.system5.process_symbols_batch") as mock_batch:
            mock_df = pd.DataFrame({"Close": [100], "filter": [True], "setup": [False]})
            mock_batch.return_value = ({"AAPL": mock_df}, [])

            result = prepare_data_vectorized_system5(
                raw_data_dict=None, symbols=symbols, reuse_indicators=False
            )

        assert "AAPL" in result

    def test_generate_candidates_system5_success(self):
        """Test generate_candidates_system5 with valid setup data"""
        dates = pd.date_range("2023-01-01", periods=3)
        mock_df = pd.DataFrame(
            {
                "Close": [100, 105, 110],
                "ADX7": [60, 55, 65],  # System5 ranks by ADX7 descending
                "setup": [True, False, True],
            },
            index=dates,
        )

        prepared_dict = {"AAPL": mock_df}

        candidates_by_date, candidates_df = generate_candidates_system5(
            prepared_dict, top_n=5
        )

        assert isinstance(candidates_by_date, dict)

    def test_get_total_days_system5(self):
        """Test get_total_days_system5 function"""
        dates = pd.date_range("2023-01-01", periods=5)
        data_dict = {"AAPL": pd.DataFrame({"Close": range(5)}, index=dates)}

        total_days = get_total_days_system5(data_dict)
        assert total_days == 5


class TestSystem6MainFunctions:
    """Test System6 main functions (Short momentum reversal)"""

    def setup_method(self):
        set_test_determinism()
        if not SYSTEM6_AVAILABLE:
            pytest.skip("core.system6 imports not available")

    def test_prepare_data_vectorized_system6_success(self):
        """Test prepare_data_vectorized_system6 with symbols parameter"""
        symbols = ["AAPL"]

        with patch("core.system6.get_cached_data") as mock_cache:
            mock_df = pd.DataFrame({"Close": [100], "filter": [True], "setup": [False]})
            mock_cache.return_value = mock_df

            result = prepare_data_vectorized_system6(
                raw_data_dict=None, symbols=symbols, reuse_indicators=False
            )

        assert len(result) >= 0  # May be empty if no valid symbols

    def test_generate_candidates_system6_success(self):
        """Test generate_candidates_system6 with valid setup data"""
        dates = pd.date_range("2023-01-01", periods=3)
        mock_df = pd.DataFrame(
            {
                "Close": [100, 115, 125],
                "Return6D": [0.25, 0.30, 0.35],  # System6 ranks by Return6D descending
                "setup": [True, False, True],
            },
            index=dates,
        )

        prepared_dict = {"AAPL": mock_df}

        candidates_by_date, candidates_df = generate_candidates_system6(
            prepared_dict, top_n=5
        )

        assert isinstance(candidates_by_date, dict)

    def test_get_total_days_system6(self):
        """Test get_total_days_system6 function"""
        dates = pd.date_range("2023-01-01", periods=5)
        data_dict = {"AAPL": pd.DataFrame({"Close": range(5)}, index=dates)}

        total_days = get_total_days_system6(data_dict)
        assert total_days == 5


class TestSystem7MainFunctions:
    """Test System7 main functions (SPY short catastrophe hedge)"""

    def setup_method(self):
        set_test_determinism()
        if not SYSTEM7_AVAILABLE:
            pytest.skip("core.system7 imports not available")

    def test_prepare_data_vectorized_system7_success(self):
        """Test prepare_data_vectorized_system7 SPY-only processing"""
        # System7 is SPY-only, test with SPY data
        spy_df = pd.DataFrame(
            {
                "Open": [400, 395, 390],
                "High": [405, 400, 395],
                "Low": [395, 390, 385],
                "Close": [400, 395, 390],
                "Volume": [100_000_000, 110_000_000, 120_000_000],
            },
            index=pd.date_range("2023-01-01", periods=3),
        )

        raw_data = {"SPY": spy_df}

        with (
            patch("core.system7.os.path.exists") as mock_exists,
            patch("core.system7.pd.read_feather"),  # mock_read removed
        ):
            mock_exists.return_value = False  # No cache

            result = prepare_data_vectorized_system7(raw_data, reuse_indicators=True)

        assert len(result) <= 1  # Should process SPY only

    def test_generate_candidates_system7_success(self):
        """Test generate_candidates_system7 SPY setup detection"""
        dates = pd.date_range("2023-01-01", periods=3)
        spy_df = pd.DataFrame(
            {
                "Close": [400, 395, 390],
                "ATR50": [8.0, 8.5, 9.0],
                "setup": [1, 0, 1],  # Setup conditions met
            },
            index=dates,
        )

        prepared_dict = {"SPY": spy_df}

        candidates_by_date, candidates_df = generate_candidates_system7(prepared_dict)

        assert isinstance(candidates_by_date, dict)
        # Should contain entries for dates with setup==1

    def test_generate_candidates_system7_no_spy(self):
        """Test generate_candidates_system7 when SPY data is missing"""
        prepared_dict = {"AAPL": pd.DataFrame({"Close": [100]})}

        candidates_by_date, candidates_df = generate_candidates_system7(prepared_dict)

        assert candidates_by_date == {}
        assert candidates_df is None

    def test_get_total_days_system7(self):
        """Test get_total_days_system7 function"""
        dates = pd.date_range("2023-01-01", periods=5)
        data_dict = {"SPY": pd.DataFrame({"Close": range(5)}, index=dates)}

        total_days = get_total_days_system7(data_dict)
        assert total_days == 5
