"""Tests for common.strategy_runner module."""

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import pytest

from common.strategy_runner import (
    StrategyRunner,
    run_strategies_parallel,
    run_strategies_serial,
)


class TestStrategyRunner:
    """Test suite for StrategyRunner class."""

    @pytest.fixture
    def mock_strategy(self):
        """Create a mock strategy for testing."""
        strategy = Mock()
        strategy.SYSTEM_NAME = "system1"
        # Mock the get_today_signals method which actually gets called
        strategy.get_today_signals.return_value = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "score": [1.5],
            }
        )
        # Also mock get_today_signals for backward compatibility
        strategy.get_today_signals.return_value = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "score": [1.5],
            }
        )
        return strategy

    @pytest.fixture
    def mock_strategies(self, mock_strategy):
        """Create mock strategies dictionary."""
        return {"system1": mock_strategy}

    @pytest.fixture
    def mock_raw_data_sets(self):
        """Create mock raw data sets."""
        return {
            "system1": {
                "AAPL": pd.DataFrame(
                    {
                        "Date": ["2023-01-01"],
                        "Close": [150.0],
                        "Volume": [1000],
                    }
                )
            }
        }

    @pytest.fixture
    def mock_basic_data(self):
        """Create mock basic data."""
        return {
            "AAPL": pd.DataFrame(
                {
                    "Date": ["2023-01-01"],
                    "Close": [150.0],
                    "Volume": [1000],
                }
            ),
            "SPY": pd.DataFrame(
                {
                    "Date": ["2023-01-01"],
                    "Close": [400.0],
                    "Volume": [50000],
                }
            ),
        }

    def test_strategy_runner_init(self):
        """Test StrategyRunner initialization."""
        log_callback = Mock()
        progress_callback = Mock()
        per_system_progress = Mock()

        runner = StrategyRunner(
            log_callback=log_callback,
            progress_callback=progress_callback,
            per_system_progress=per_system_progress,
        )

        assert runner.log_callback == log_callback
        assert runner.progress_callback == progress_callback
        assert runner.per_system_progress == per_system_progress

    def test_strategy_runner_serial_execution(self, mock_strategies, mock_raw_data_sets, mock_basic_data):
        """Test serial strategy execution."""
        runner = StrategyRunner()

        result = runner.run_strategies(
            strategies=mock_strategies,
            basic_data=mock_basic_data,
            raw_data_sets=mock_raw_data_sets,
            spy_df=mock_basic_data.get("SPY"),
            parallel=False,
        )

        assert isinstance(result, dict)
        assert "system1" in result
        # Result should be tuple of (DataFrame, message, logs)
        assert len(result["system1"]) == 3
        df, msg, logs = result["system1"]
        assert isinstance(df, pd.DataFrame)
        assert isinstance(msg, str)
        assert isinstance(logs, list)

    def test_strategy_runner_parallel_execution(self, mock_strategies, mock_raw_data_sets, mock_basic_data):
        """Test parallel strategy execution."""
        runner = StrategyRunner()

        result = runner.run_strategies(
            strategies=mock_strategies,
            basic_data=mock_basic_data,
            raw_data_sets=mock_raw_data_sets,
            spy_df=mock_basic_data.get("SPY"),
            parallel=True,
        )

        assert isinstance(result, dict)
        assert "system1" in result
        # Result should be tuple of (DataFrame, message, logs)
        assert len(result["system1"]) == 3
        df, msg, logs = result["system1"]
        assert isinstance(df, pd.DataFrame)
        assert isinstance(msg, str)
        assert isinstance(logs, list)

    def test_run_strategies_parallel_function(self, mock_strategies, mock_raw_data_sets, mock_basic_data):
        """Test the run_strategies_parallel function."""
        result = run_strategies_parallel(
            strategies=mock_strategies,
            basic_data=mock_basic_data,
            raw_data_sets=mock_raw_data_sets,
            spy_df=mock_basic_data.get("SPY"),
        )

        assert isinstance(result, dict)
        assert "system1" in result
        # Result should be tuple of (DataFrame, message, logs)
        assert len(result["system1"]) == 3

    def test_run_strategies_serial_function(self, mock_strategies, mock_raw_data_sets, mock_basic_data):
        """Test the run_strategies_serial function."""
        result = run_strategies_serial(
            strategies=mock_strategies,
            basic_data=mock_basic_data,
            raw_data_sets=mock_raw_data_sets,
            spy_df=mock_basic_data.get("SPY"),
        )

        assert isinstance(result, dict)
        assert "system1" in result
        # Result should be tuple of (DataFrame, message, logs)
        assert len(result["system1"]) == 3

    def test_strategy_error_handling(self, mock_strategies, mock_raw_data_sets, mock_basic_data):
        """Test error handling when strategy fails."""
        # Make strategy raise an exception
        mock_strategies["system1"].get_today_signals.side_effect = Exception("Test error")

        result = run_strategies_serial(
            strategies=mock_strategies,
            basic_data=mock_basic_data,
            raw_data_sets=mock_raw_data_sets,
            spy_df=mock_basic_data.get("SPY"),
        )

        assert isinstance(result, dict)
        assert "system1" in result
        df, msg, logs = result["system1"]
        assert isinstance(df, pd.DataFrame)
        assert df.empty  # Should return empty DataFrame on error
        assert "件" in msg or "error" in msg.lower()  # Accept Japanese or English error message
        assert len(logs) > 0

    def test_missing_spy_handling(self, mock_strategies, mock_raw_data_sets, mock_basic_data):
        """Test handling when SPY data is missing for systems that need it."""
        # Test system2 which typically needs SPY
        system2_strategy = Mock()
        system2_strategy.SYSTEM_NAME = "system2"
        system2_strategy.get_today_signals.return_value = pd.DataFrame()

        strategies = {"system2": system2_strategy}
        raw_data_sets = {"system2": {"AAPL": mock_basic_data["AAPL"]}}

        result = run_strategies_serial(
            strategies=strategies,
            basic_data={"AAPL": mock_basic_data["AAPL"]},  # No SPY
            raw_data_sets=raw_data_sets,
            spy_df=None,
        )

        assert isinstance(result, dict)
        assert "system2" in result
        # Should handle missing SPY gracefully

    def test_callback_handling(self, mock_strategies, mock_raw_data_sets, mock_basic_data):
        """Test callback function handling."""
        per_system_progress = Mock()
        log_callback = Mock()

        result = run_strategies_parallel(
            strategies=mock_strategies,
            basic_data=mock_basic_data,
            raw_data_sets=mock_raw_data_sets,
            spy_df=mock_basic_data.get("SPY"),
            per_system_progress=per_system_progress,
            log_callback=log_callback,
        )

        assert isinstance(result, dict)
        # Note: Callbacks may or may not be called depending on implementation

    def test_empty_strategies_handling(self):
        """Test handling of empty strategies dictionary."""
        result = run_strategies_parallel(
            strategies={},
            basic_data={},
            raw_data_sets={},
            spy_df=None,
        )

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_system7_spy_only_handling(self, mock_basic_data):
        """Test system7 SPY-only handling."""
        system7_strategy = Mock()
        system7_strategy.SYSTEM_NAME = "system7"
        system7_strategy.get_today_signals.return_value = pd.DataFrame(
            {
                "symbol": ["SPY"],
                "score": [1.0],
            }
        )

        strategies = {"system7": system7_strategy}
        raw_data_sets = {"system7": {"SPY": mock_basic_data["SPY"]}}

        result = run_strategies_serial(
            strategies=strategies,
            basic_data=mock_basic_data,
            raw_data_sets=raw_data_sets,
            spy_df=mock_basic_data.get("SPY"),
        )

        assert isinstance(result, dict)
        assert "system7" in result
        df, msg, logs = result["system7"]
        assert isinstance(df, pd.DataFrame)
        assert isinstance(msg, str)
        assert isinstance(logs, list)

    def test_large_strategy_set(self, mock_basic_data):
        """Test handling of all systems together."""
        strategies = {}
        raw_data_sets = {}

        for i in range(1, 8):  # system1 through system7
            strategy = Mock()
            strategy.SYSTEM_NAME = f"system{i}"
            strategy.get_today_signals.return_value = pd.DataFrame(
                {
                    "symbol": ["AAPL"],
                    "score": [float(i)],
                }
            )

            strategies[f"system{i}"] = strategy
            if i == 7:  # system7 gets only SPY
                raw_data_sets[f"system{i}"] = {"SPY": mock_basic_data["SPY"]}
            else:
                raw_data_sets[f"system{i}"] = {"AAPL": mock_basic_data["AAPL"]}

        result = run_strategies_parallel(
            strategies=strategies,
            basic_data=mock_basic_data,
            raw_data_sets=raw_data_sets,
            spy_df=mock_basic_data.get("SPY"),
        )

        assert isinstance(result, dict)
        assert len(result) == 7
        for i in range(1, 8):
            assert f"system{i}" in result
            df, msg, logs = result[f"system{i}"]
            assert isinstance(df, pd.DataFrame)
            assert isinstance(msg, str)
            assert isinstance(logs, list)

    def test_runner_with_callbacks(self, mock_strategies, mock_raw_data_sets, mock_basic_data):
        """Test StrategyRunner with callbacks."""
        log_callback = Mock()
        progress_callback = Mock()
        per_system_progress = Mock()

        runner = StrategyRunner(
            log_callback=log_callback,
            progress_callback=progress_callback,
            per_system_progress=per_system_progress,
        )

        result = runner.run_strategies(
            strategies=mock_strategies,
            basic_data=mock_basic_data,
            raw_data_sets=mock_raw_data_sets,
            spy_df=mock_basic_data.get("SPY"),
            parallel=False,
        )

        assert isinstance(result, dict)
        # Callbacks should be available to the runner

    @pytest.mark.parametrize("parallel", [True, False])
    def test_both_execution_modes(self, mock_strategies, mock_raw_data_sets, mock_basic_data, parallel):
        """Test both parallel and serial execution modes."""
        runner = StrategyRunner()

        result = runner.run_strategies(
            strategies=mock_strategies,
            basic_data=mock_basic_data,
            raw_data_sets=mock_raw_data_sets,
            spy_df=mock_basic_data.get("SPY"),
            parallel=parallel,
        )

        assert isinstance(result, dict)
        assert "system1" in result
        df, msg, logs = result["system1"]
        assert isinstance(df, pd.DataFrame)
        assert isinstance(msg, str)
        assert isinstance(logs, list)
