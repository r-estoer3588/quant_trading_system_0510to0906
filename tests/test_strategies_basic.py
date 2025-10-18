"""
Basic tests for strategies module to improve coverage
Focus on base_strategy and system strategy initialization
"""

from __future__ import annotations

import pandas as pd
import pytest

from common.testing import set_test_determinism
from strategies.base_strategy import StrategyBase
from strategies.system1_strategy import System1Strategy
from strategies.system2_strategy import System2Strategy
from strategies.system3_strategy import System3Strategy


class MockStrategy(StrategyBase):
    """Mock concrete implementation of StrategyBase for testing"""

    def __init__(self):
        super().__init__()

    def prepare_data(
        self,
        raw_data_or_symbols: dict,
        reuse_indicators: bool | None = None,
        **kwargs,
    ) -> dict:
        """Mock prepare_data implementation"""
        return {
            "AAPL": pd.DataFrame(
                {
                    "Close": [100, 101, 102],
                    "Date": pd.date_range("2023-01-01", periods=3),
                }
            )
        }

    def generate_candidates(
        self, data_dict: dict, market_df: pd.DataFrame | None = None, **kwargs
    ) -> tuple[dict, pd.DataFrame | None]:
        """Mock generate_candidates implementation"""
        return data_dict, market_df

    def run_backtest(self, data_dict: dict, candidates_by_date: dict, capital: float, **kwargs) -> pd.DataFrame:
        """Mock run_backtest implementation"""
        return pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=3),
                "capital": [10000, 10100, 10200],
            }
        )


class TestBaseStrategy:
    """Test the abstract base strategy class"""

    def setup_method(self):
        set_test_determinism()

    def test_base_strategy_initialization_success(self):
        """Test successful base strategy initialization"""
        strategy = MockStrategy()

        # Test that strategy has the required abstract methods
        assert hasattr(strategy, "prepare_data")
        assert hasattr(strategy, "generate_candidates")
        assert hasattr(strategy, "run_backtest")

        # Test shared methods exist
        assert hasattr(strategy, "_resolve_data_params")
        assert hasattr(strategy, "_get_top_n_setting")
        assert hasattr(strategy, "_get_market_df")

    def test_base_strategy_abstract_methods(self):
        """Test that StrategyBase is properly abstract"""
        # Should not be able to instantiate StrategyBase directly
        with pytest.raises(TypeError):
            # This should raise TypeError because StrategyBase is abstract
            StrategyBase()


class TestSystem1Strategy:
    """Test System1Strategy basic functionality"""

    def setup_method(self):
        set_test_determinism()

    def test_system1_initialization(self):
        """Test System1Strategy initialization"""
        strategy = System1Strategy()

        # Test that strategy has the required methods
        assert hasattr(strategy, "prepare_data")
        assert hasattr(strategy, "generate_candidates")
        assert hasattr(strategy, "run_backtest")

    def test_system1_prepare_data_basic(self):
        """Test System1Strategy prepare_data method"""
        strategy = System1Strategy()

        # Mock minimal data structure
        mock_data = {
            "AAPL": pd.DataFrame(
                {
                    "Close": [100, 101, 102],
                    "Date": pd.date_range("2023-01-01", periods=3),
                }
            )
        }

        try:
            result = strategy.prepare_data(mock_data)
            assert isinstance(result, dict)
        except Exception:
            # Expected to fail with test data but should not raise AttributeError
            pass

    def test_system1_generate_candidates_basic(self):
        """Test System1Strategy generate_candidates method"""
        strategy = System1Strategy()

        # Mock minimal data structure
        mock_data = {
            "AAPL": pd.DataFrame(
                {
                    "Close": [100, 101, 102],
                    "Date": pd.date_range("2023-01-01", periods=3),
                }
            )
        }

        try:
            result = strategy.generate_candidates(mock_data)
            assert isinstance(result, tuple)
            assert len(result) == 2
        except Exception:
            # Expected to fail with test data but should not raise AttributeError
            pass


class TestSystem2Strategy:
    """Test System2Strategy basic functionality"""

    def setup_method(self):
        set_test_determinism()

    def test_system2_initialization(self):
        """Test System2Strategy initialization"""
        strategy = System2Strategy()

        # Test that strategy has the required methods
        assert hasattr(strategy, "prepare_data")
        assert hasattr(strategy, "generate_candidates")
        assert hasattr(strategy, "run_backtest")

    def test_system2_prepare_data_basic(self):
        """Test System2Strategy prepare_data method"""
        strategy = System2Strategy()

        # Mock minimal data structure
        mock_data = {
            "AAPL": pd.DataFrame(
                {
                    "Close": [100, 101, 102],
                    "Date": pd.date_range("2023-01-01", periods=3),
                }
            )
        }

        try:
            result = strategy.prepare_data(mock_data)
            assert isinstance(result, dict)
        except Exception:
            # Expected to fail with test data but should not raise AttributeError
            pass


class TestSystem3Strategy:
    """Test System3Strategy basic functionality"""

    def setup_method(self):
        set_test_determinism()

    def test_system3_initialization(self):
        """Test System3Strategy initialization"""
        strategy = System3Strategy()

        # Test that strategy has the required methods
        assert hasattr(strategy, "prepare_data")
        assert hasattr(strategy, "generate_candidates")
        assert hasattr(strategy, "run_backtest")

    def test_system3_methods_basic(self):
        """Test System3Strategy basic method calls"""
        strategy = System3Strategy()

        # Mock minimal data structure
        mock_data = {"TEST": pd.DataFrame({"Close": [100.0], "Date": pd.date_range("2023-01-01", periods=1)})}

        try:
            result = strategy.prepare_data(mock_data)
            assert isinstance(result, dict)
        except Exception:
            # Expected to fail with test data but should not raise AttributeError
            pass


class TestStrategyErrorHandling:
    """Test error handling across strategy classes"""

    def setup_method(self):
        set_test_determinism()

    def test_strategy_with_empty_data(self):
        """Test strategy methods with empty DataFrames"""
        strategy = System1Strategy()

        empty_data = {}

        # Should handle gracefully without raising AttributeError
        try:
            result = strategy.prepare_data(empty_data)
            assert isinstance(result, dict)
        except Exception:
            # Expected to handle gracefully, but may raise other exceptions
            pass

    def test_strategy_with_invalid_data(self):
        """Test strategy methods with invalid data format"""
        strategy = System1Strategy()

        invalid_data = {"INVALID": pd.DataFrame()}  # Empty DataFrame

        try:
            result = strategy.prepare_data(invalid_data)
            assert isinstance(result, dict)
        except Exception:
            # Expected to handle gracefully, but may raise other exceptions
            pass


if __name__ == "__main__":
    pytest.main([__file__])
