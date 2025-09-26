"""
Basic tests for strategies module to improve coverage
Focus on base_strategy and system strategy initialization
"""

from __future__ import annotations

import pandas as pd
from unittest.mock import Mock, patch
import pytest
from datetime import datetime

from strategies.base_strategy import StrategyBase
from strategies.system1_strategy import System1Strategy
from strategies.system2_strategy import System2Strategy
from strategies.system3_strategy import System3Strategy
from common.testing import set_test_determinism


class MockStrategy(StrategyBase):
    """Mock concrete implementation of StrategyBase for testing"""

    def __init__(self):
        super().__init__()

    def entry_rules(self, df: pd.DataFrame, meta: dict) -> pd.DataFrame:
        return df

    def exit_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def position_size(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class TestBaseStrategy:
    """Test the abstract base strategy class"""

    def setup_method(self):
        set_test_determinism()

    def test_base_strategy_initialization_success(self):
        """Test successful base strategy initialization"""
        with patch("strategies.base_strategy.get_settings") as mock_get_settings:
            with patch("strategies.base_strategy.get_system_params") as mock_get_params:
                # Mock settings
                mock_settings = Mock()
                mock_settings.risk = Mock()
                mock_settings.risk.risk_pct = 0.02
                mock_get_settings.return_value = mock_settings

                # Mock system params
                mock_params = {"test_param": "test_value"}
                mock_get_params.return_value = mock_params

                strategy = MockStrategy()

                assert hasattr(strategy, "config")
                assert strategy.config is not None

    def test_base_strategy_initialization_import_error(self):
        """Test base strategy initialization with import error"""
        with patch(
            "strategies.base_strategy.get_settings", side_effect=ImportError("Mock import error")
        ):
            with patch("strategies.base_strategy.logger") as mock_logger:
                strategy = MockStrategy()

                assert strategy.config == {}
                mock_logger.error.assert_called()

    def test_base_strategy_initialization_settings_error(self):
        """Test base strategy initialization with settings error"""
        with patch("strategies.base_strategy.get_settings") as mock_get_settings:
            mock_get_settings.side_effect = Exception("Mock settings error")

            with patch("strategies.base_strategy.logger") as mock_logger:
                strategy = MockStrategy()

                assert strategy.config == {}
                mock_logger.error.assert_called()

    def test_base_strategy_abstract_methods(self):
        """Test that StrategyBase is properly abstract"""
        # Should not be able to instantiate StrategyBase directly
        with pytest.raises(TypeError):
            StrategyBase()


class TestSystem1Strategy:
    """Test System1Strategy basic functionality"""

    def setup_method(self):
        set_test_determinism()

    def test_system1_initialization(self):
        """Test System1Strategy initialization"""
        with patch("strategies.system1_strategy.get_settings") as mock_settings:
            with patch("strategies.system1_strategy.get_system_params") as mock_params:
                mock_settings.return_value = Mock()
                mock_params.return_value = {}

                strategy = System1Strategy()
                assert strategy is not None
                assert hasattr(strategy, "config")

    def test_system1_entry_rules_basic(self):
        """Test System1Strategy entry_rules method"""
        with patch("strategies.system1_strategy.get_settings"):
            with patch("strategies.system1_strategy.get_system_params"):
                strategy = System1Strategy()

                # Create sample test data
                test_data = pd.DataFrame(
                    {
                        "symbol": ["AAPL", "MSFT"],
                        "close": [150.0, 300.0],
                        "sma50": [145.0, 295.0],
                        "rsi14": [65.0, 45.0],
                    }
                )

                meta = {"current_date": datetime(2023, 1, 1)}

                with patch("core.system1.entry_rules") as mock_entry:
                    mock_entry.return_value = test_data

                    result = strategy.entry_rules(test_data, meta)

                    assert result is not None
                    assert isinstance(result, pd.DataFrame)
                    mock_entry.assert_called_once()

    def test_system1_exit_rules_basic(self):
        """Test System1Strategy exit_rules method"""
        with patch("strategies.system1_strategy.get_settings"):
            with patch("strategies.system1_strategy.get_system_params"):
                strategy = System1Strategy()

                test_data = pd.DataFrame(
                    {"symbol": ["AAPL"], "close": [150.0], "entry_price": [145.0]}
                )

                with patch("core.system1.exit_rules") as mock_exit:
                    mock_exit.return_value = test_data

                    result = strategy.exit_rules(test_data)

                    assert result is not None
                    assert isinstance(result, pd.DataFrame)
                    mock_exit.assert_called_once()


class TestSystem2Strategy:
    """Test System2Strategy basic functionality"""

    def setup_method(self):
        set_test_determinism()

    def test_system2_initialization(self):
        """Test System2Strategy initialization"""
        with patch("strategies.system2_strategy.get_settings"):
            with patch("strategies.system2_strategy.get_system_params"):
                strategy = System2Strategy()
                assert strategy is not None

    def test_system2_entry_rules_basic(self):
        """Test System2Strategy entry_rules method"""
        with patch("strategies.system2_strategy.get_settings"):
            with patch("strategies.system2_strategy.get_system_params"):
                strategy = System2Strategy()

                test_data = pd.DataFrame({"symbol": ["AAPL"], "close": [150.0]})
                meta = {}

                with patch("core.system2.entry_rules") as mock_entry:
                    mock_entry.return_value = test_data

                    result = strategy.entry_rules(test_data, meta)
                    assert isinstance(result, pd.DataFrame)


class TestSystem3Strategy:
    """Test System3Strategy basic functionality"""

    def setup_method(self):
        set_test_determinism()

    def test_system3_initialization(self):
        """Test System3Strategy initialization"""
        with patch("strategies.system3_strategy.get_settings"):
            with patch("strategies.system3_strategy.get_system_params"):
                strategy = System3Strategy()
                assert strategy is not None

    def test_system3_methods_basic(self):
        """Test System3Strategy basic method calls"""
        with patch("strategies.system3_strategy.get_settings"):
            with patch("strategies.system3_strategy.get_system_params"):
                strategy = System3Strategy()

                test_data = pd.DataFrame({"symbol": ["TEST"], "close": [100.0]})

                # Test entry_rules
                with patch("core.system3.entry_rules") as mock_entry:
                    mock_entry.return_value = test_data
                    result = strategy.entry_rules(test_data, {})
                    assert isinstance(result, pd.DataFrame)

                # Test exit_rules
                with patch("core.system3.exit_rules") as mock_exit:
                    mock_exit.return_value = test_data
                    result = strategy.exit_rules(test_data)
                    assert isinstance(result, pd.DataFrame)


class TestStrategyErrorHandling:
    """Test error handling across strategy classes"""

    def setup_method(self):
        set_test_determinism()

    def test_strategy_with_core_import_error(self):
        """Test strategy behavior when core modules fail to import"""
        with patch("strategies.system1_strategy.get_settings"):
            with patch("strategies.system1_strategy.get_system_params"):
                strategy = System1Strategy()

                # Mock core import failure
                with patch(
                    "strategies.system1_strategy.core.system1.entry_rules",
                    side_effect=ImportError("Core import failed"),
                ):
                    test_data = pd.DataFrame({"symbol": ["TEST"]})

                    # Should handle gracefully
                    try:
                        strategy.entry_rules(test_data, {})
                    except ImportError:
                        pass  # Expected behavior

    def test_strategy_with_empty_data(self):
        """Test strategy methods with empty DataFrames"""
        with patch("strategies.system1_strategy.get_settings"):
            with patch("strategies.system1_strategy.get_system_params"):
                strategy = System1Strategy()

                empty_df = pd.DataFrame()

                with patch("core.system1.entry_rules") as mock_entry:
                    mock_entry.return_value = empty_df

                    result = strategy.entry_rules(empty_df, {})
                    assert isinstance(result, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__])
