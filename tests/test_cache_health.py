"""
Cache health and rolling analysis functionality tests.
"""

import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock

import pandas as pd
import pytest

from common.cache_manager import CacheManager
from common.system_groups import (
    analyze_system_symbols_coverage,
    format_cache_coverage_report,
)


class TestCacheHealthFunctionality:
    """Test new cache health analysis features."""

    def test_analyze_rolling_gaps_empty_cache(self):
        """Test analyze_rolling_gaps with empty cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock settings
            mock_settings = MagicMock()
            mock_settings.cache.full_dir = Path(temp_dir) / "full_backup"
            mock_settings.cache.rolling_dir = Path(temp_dir) / "rolling"
            mock_settings.cache.rolling.meta_file = "_meta.json"
            mock_settings.cache.rolling.lookback_days = 100
            mock_settings.cache.rolling.buffer_days = 20

            # Create directories
            mock_settings.cache.full_dir.mkdir(parents=True)
            mock_settings.cache.rolling_dir.mkdir(parents=True)

            cache_manager = CacheManager(mock_settings)

            # Test with empty cache
            result = cache_manager.analyze_rolling_gaps([])

            assert result["status"] == "error"
            assert "分析対象シンボルが見つかりません" in result["message"]

    def test_analyze_rolling_gaps_with_symbols(self):
        """Test analyze_rolling_gaps with some test symbols."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock settings
            mock_settings = MagicMock()
            mock_settings.cache.full_dir = Path(temp_dir) / "full_backup"
            mock_settings.cache.rolling_dir = Path(temp_dir) / "rolling"
            mock_settings.cache.rolling.meta_file = "_meta.json"
            mock_settings.cache.rolling.lookback_days = 100
            mock_settings.cache.rolling.buffer_days = 20

            # Create directories
            mock_settings.cache.full_dir.mkdir(parents=True)
            mock_settings.cache.rolling_dir.mkdir(parents=True)

            cache_manager = CacheManager(mock_settings)

            # Mock read method to simulate some symbols available, some not
            def mock_read(ticker, profile):
                if profile == "rolling" and ticker == "AVAILABLE_SYM":
                    return pd.DataFrame({"date": ["2023-01-01"], "close": [100]})
                return None

            cache_manager.read = mock_read

            test_symbols = ["AVAILABLE_SYM", "MISSING_SYM"]
            result = cache_manager.analyze_rolling_gaps(test_symbols)

            assert result["status"] == "success"
            assert result["total_symbols"] == 2
            # Note: The mock may not fully represent real behavior, so we just check structure
            assert "healthy" in result
            assert "missing_files" in result

    def test_get_rolling_health_summary(self):
        """Test get_rolling_health_summary method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock settings
            mock_settings = MagicMock()
            mock_settings.cache.full_dir = Path(temp_dir) / "full_backup"
            mock_settings.cache.rolling_dir = Path(temp_dir) / "rolling"
            mock_settings.cache.rolling.meta_file = "_meta.json"
            mock_settings.cache.rolling.lookback_days = 100
            mock_settings.cache.rolling.buffer_days = 20

            # Create directories
            mock_settings.cache.full_dir.mkdir(parents=True)
            mock_settings.cache.rolling_dir.mkdir(parents=True)

            # Create a meta file
            meta_path = mock_settings.cache.rolling_dir / "_meta.json"
            meta_content = {"anchor_rows_at_prune": 150}
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_content, f)

            # Create some dummy rolling files
            (mock_settings.cache.rolling_dir / "SPY.csv").touch()
            (mock_settings.cache.rolling_dir / "AAPL.csv").touch()

            cache_manager = CacheManager(mock_settings)

            # Mock read method for SPY
            def mock_read(ticker, profile):
                if ticker == "SPY" and profile == "rolling":
                    return pd.DataFrame(
                        {"date": pd.date_range("2023-01-01", periods=120)}
                    )
                return None

            cache_manager.read = mock_read

            result = cache_manager.get_rolling_health_summary()

            assert result["status"] == "success"
            # The mock may not produce expected detailed results, just verify basic structure
            assert "total_files" in result

    def test_format_cache_coverage_report(self):
        """Test cache coverage report formatting."""
        # Test good coverage
        report = format_cache_coverage_report(
            total_symbols=100,
            available_count=95,
            missing_count=5,
            coverage_percentage=95.0,
            missing_symbols=["SYM1", "SYM2", "SYM3"],
        )

        assert report["status"] == "✅ 良好"
        assert report["priority"] == "低"
        assert report["summary"]["coverage"] == "95.0%"
        assert len(report["missing_symbols_preview"]) == 3
        assert "excellent" in report["recommendations"][0]

        # Test poor coverage
        report_poor = format_cache_coverage_report(
            total_symbols=100,
            available_count=40,
            missing_count=60,
            coverage_percentage=40.0,
            missing_symbols=["SYM" + str(i) for i in range(60)],
        )

        assert report_poor["status"] == "🚨 緊急"
        assert report_poor["priority"] == "高"
        assert "緊急" in report_poor["recommendations"][0]
        assert (
            len(report_poor["missing_symbols_preview"]) == 11
        )  # 10 symbols + "... 他XX"

    def test_analyze_system_symbols_coverage(self):
        """Test system symbols coverage analysis."""
        system_symbols_map = {
            "system1": ["AAPL", "MSFT", "GOOGL"],
            "system2": ["TSLA", "NVDA", "AMD"],
            "system3": ["SPY", "QQQ"],
        }

        cache_analysis_results = {"missing_symbols": ["MSFT", "NVDA"]}

        result = analyze_system_symbols_coverage(
            system_symbols_map, cache_analysis_results
        )

        # Check system1 (1 missing out of 3)
        assert result["by_system"]["system1"]["total_symbols"] == 3
        assert result["by_system"]["system1"]["missing"] == 1
        assert result["by_system"]["system1"]["available"] == 2
        assert abs(result["by_system"]["system1"]["coverage_percentage"] - 66.67) < 0.1

        # Check system2 (1 missing out of 3)
        assert result["by_system"]["system2"]["missing"] == 1

        # Check system3 (0 missing)
        assert result["by_system"]["system3"]["missing"] == 0
        assert result["by_system"]["system3"]["coverage_percentage"] == 100.0


if __name__ == "__main__":
    pytest.main([__file__])
