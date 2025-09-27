"""
Test for rolling issue aggregator functionality.
"""

import os
from unittest.mock import patch

import pytest

from common.cache_manager import report_rolling_issue, _RollingIssueAggregator


class TestRollingIssueAggregator:
    """Test rolling issue aggregation functionality."""

    def test_compact_mode_disabled_by_default(self):
        """Test that compact mode is disabled by default."""
        # Reset the singleton
        _RollingIssueAggregator._instance = None

        aggregator = _RollingIssueAggregator()
        assert aggregator.compact_mode is False
        assert aggregator.verbose_head == 20

    def test_compact_mode_enabled_with_env_var(self):
        """Test compact mode activation with environment variable."""
        with patch.dict(os.environ, {"COMPACT_TODAY_LOGS": "1"}):
            # Reset the singleton
            _RollingIssueAggregator._instance = None

            aggregator = _RollingIssueAggregator()
            assert aggregator.compact_mode is True

    def test_verbose_head_configuration(self):
        """Test verbose head configuration via environment."""
        with patch.dict(os.environ, {"ROLLING_ISSUES_VERBOSE_HEAD": "5"}):
            # Reset the singleton
            _RollingIssueAggregator._instance = None

            aggregator = _RollingIssueAggregator()
            assert aggregator.verbose_head == 5

    def test_report_rolling_issue_direct(self):
        """Test direct report_rolling_issue function."""
        # Reset the singleton
        _RollingIssueAggregator._instance = None

        # Test without compact mode (should not accumulate)
        report_rolling_issue("test_category", "AAPL", "test message")

        # This should pass without errors
        assert True

    def test_aggregator_issue_accumulation(self):
        """Test issue accumulation in compact mode."""
        with patch.dict(
            os.environ, {"COMPACT_TODAY_LOGS": "1", "ROLLING_ISSUES_VERBOSE_HEAD": "2"}
        ):
            # Reset the singleton
            _RollingIssueAggregator._instance = None

            aggregator = _RollingIssueAggregator()

            # Report multiple issues
            aggregator.report_issue("missing_rolling", "AAPL")
            aggregator.report_issue("missing_rolling", "MSFT")
            aggregator.report_issue("missing_rolling", "GOOGL")
            aggregator.report_issue("insufficient_data", "TSLA")

            # Check accumulation
            assert len(aggregator.issues["missing_rolling"]) == 3
            assert len(aggregator.issues["insufficient_data"]) == 1
            assert aggregator.warning_count == 4

    def test_summary_output_formatting(self):
        """Test summary output formatting."""
        with patch.dict(os.environ, {"COMPACT_TODAY_LOGS": "1"}):
            # Reset the singleton
            _RollingIssueAggregator._instance = None

            aggregator = _RollingIssueAggregator()

            # Add test data
            test_symbols = [f"SYM{i}" for i in range(15)]
            for symbol in test_symbols:
                aggregator.report_issue("missing_rolling", symbol)

            # Manual call to _output_summary for testing
            with patch.object(aggregator.logger, "info") as mock_info:
                aggregator._output_summary()

                # Verify summary was called
                assert mock_info.called
                call_args = [call.args[0] for call in mock_info.call_args_list]

                # Should have summary header and total
                summary_calls = [arg for arg in call_args if "Rolling Cache Issues Summary" in arg]
                assert len(summary_calls) == 1

                total_calls = [arg for arg in call_args if "Total issues reported:" in arg]
                assert len(total_calls) == 1


if __name__ == "__main__":
    pytest.main([__file__])
