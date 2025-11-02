"""
Final push tests for cache_manager.py to reach 80% coverage
Focus on critical uncovered methods
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from common.cache_manager import CacheManager
from common.cache_manager_old import _RollingIssueAggregator
from common.testing import set_test_determinism


class TestCacheManagerCriticalMethods:
    """Test critical uncovered methods for final coverage push"""

    def setup_method(self):
        set_test_determinism()
        self.temp_dir = Path(tempfile.mkdtemp())

        self.mock_settings = Mock()
        self.mock_settings.cache = Mock()
        self.mock_settings.cache.full_dir = str(self.temp_dir / "full")
        self.mock_settings.cache.rolling_dir = str(self.temp_dir / "rolling")
        self.mock_settings.cache.rolling = Mock()
        self.mock_settings.cache.rolling.meta_file = "meta.json"
        self.mock_settings.cache.rolling.base_lookback_days = 250
        self.mock_settings.cache.rolling.buffer_days = 50
        self.mock_settings.cache.file_format = "csv"

        with patch(
            "common.cache_manager.get_settings", return_value=self.mock_settings
        ):
            self.manager = CacheManager(self.mock_settings)

    def teardown_method(self):
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_get_rolling_health_summary_with_meta(self):
        """Test get_rolling_health_summary with meta file present"""
        # Create mock meta file
        meta_content = {"last_update": "2023-01-01", "version": "1.0"}
        meta_path = self.temp_dir / "rolling" / "meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        with open(meta_path, "w") as f:
            json.dump(meta_content, f)

        # Create some rolling cache files
        rolling_dir = self.temp_dir / "rolling"
        (rolling_dir / "AAPL.csv").touch()
        (rolling_dir / "MSFT.csv").touch()

        with patch("common.cache_manager.logger") as mock_logger:
            result = self.manager.get_rolling_health_summary()

            assert isinstance(result, dict)
            mock_logger.info.assert_called()  # Should log health check start

    def test_get_rolling_health_summary_no_meta(self):
        """Test get_rolling_health_summary without meta file"""
        # Ensure no meta file exists
        assert not self.manager.rolling_meta_path.exists()

        with patch("common.cache_manager.logger") as mock_logger:
            result = self.manager.get_rolling_health_summary()

            assert isinstance(result, dict)
            mock_logger.info.assert_called()

    def test_get_rolling_health_summary_corrupt_meta(self):
        """Test get_rolling_health_summary with corrupt meta file"""
        # Create corrupt meta file
        meta_path = self.temp_dir / "rolling" / "meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        with open(meta_path, "w") as f:
            f.write("invalid json content")

        with patch("common.cache_manager.logger") as mock_logger:
            result = self.manager.get_rolling_health_summary()

            assert isinstance(result, dict)
            mock_logger.warning.assert_called()  # Should warn about corrupt file


class TestRollingIssueAggregatorCompact:
    """Test RollingIssueAggregator compact mode scenarios"""

    def test_compact_mode_with_environment_vars(self):
        """Test compact mode initialization with environment variables"""
        with patch.dict(
            os.environ, {"COMPACT_TODAY_LOGS": "1", "ROLLING_ISSUES_VERBOSE_HEAD": "5"}
        ):
            # Reset singleton to force re-initialization
            _RollingIssueAggregator._instance = None

            aggregator = _RollingIssueAggregator()
            assert aggregator.compact_mode
            assert aggregator.verbose_head == 5

            # Test reporting with compact mode
            with patch.object(aggregator.logger, "warning") as mock_warning:
                with patch.object(aggregator.logger, "debug") as mock_debug:
                    # Report multiple issues
                    for i in range(7):
                        aggregator.report_issue("test_cat", f"SYM{i}", f"msg{i}")

                    # First 5 should be warnings, rest debug
                    assert mock_warning.call_count == 5
                    assert mock_debug.call_count == 2

    def test_output_summary_execution(self):
        """Test _output_summary method execution"""
        with patch.dict(os.environ, {"COMPACT_TODAY_LOGS": "1"}):
            _RollingIssueAggregator._instance = None
            aggregator = _RollingIssueAggregator()

            # Add test data
            aggregator.issues["missing_data"].extend(["SYM1", "SYM2"])
            aggregator.issues["corrupt_data"].append("SYM3")

            with patch.object(aggregator.logger, "info") as mock_info:
                aggregator._output_summary()

                # Should log summary information
                assert mock_info.call_count >= 2

    def test_non_compact_mode(self):
        """Test non-compact mode (default behavior)"""
        with patch.dict(os.environ, {"COMPACT_TODAY_LOGS": "0"}):
            _RollingIssueAggregator._instance = None
            aggregator = _RollingIssueAggregator()

            assert not aggregator.compact_mode

            with patch.object(aggregator.logger, "warning") as mock_warning:
                aggregator.report_issue("test", "SYMBOL", "message")
                mock_warning.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
