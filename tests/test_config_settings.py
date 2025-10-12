"""
Tests for config.settings module to improve coverage
Focus on utility functions and configuration loading
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest

from common.testing import set_test_determinism
from config.settings import (
    DataConfig,
    RiskConfig,
    UIConfig,
    _as_path,
    _env_float,
    _env_int,
    _load_config_generic,
    _positive_int_or_none,
    get_settings,
)


class TestEnvironmentUtilities:
    """Test environment variable utility functions"""

    def setup_method(self):
        set_test_determinism()

    def test_env_int_with_valid_env(self):
        """Test _env_int with valid environment variable"""
        with patch.dict(os.environ, {"TEST_INT": "42"}):
            result = _env_int("TEST_INT", 10)
            assert result == 42

    def test_env_int_with_invalid_env(self):
        """Test _env_int with invalid environment variable"""
        with patch.dict(os.environ, {"TEST_INT": "invalid"}):
            result = _env_int("TEST_INT", 10)
            assert result == 10  # Should return default

    def test_env_int_with_missing_env(self):
        """Test _env_int with missing environment variable"""
        with patch.dict(os.environ, {}, clear=True):
            result = _env_int("MISSING_VAR", 15)
            assert result == 15  # Should return default

    def test_env_float_with_valid_env(self):
        """Test _env_float with valid environment variable"""
        with patch.dict(os.environ, {"TEST_FLOAT": "3.14"}):
            result = _env_float("TEST_FLOAT", 1.0)
            assert abs(result - 3.14) < 0.001

    def test_env_float_with_invalid_env(self):
        """Test _env_float with invalid environment variable"""
        with patch.dict(os.environ, {"TEST_FLOAT": "not_a_float"}):
            result = _env_float("TEST_FLOAT", 2.5)
            assert result == 2.5  # Should return default


class TestPositiveIntOrNone:
    """Test _positive_int_or_none utility function"""

    def setup_method(self):
        set_test_determinism()

    def test_positive_int_or_none_with_none(self):
        """Test _positive_int_or_none with None input"""
        result = _positive_int_or_none(None)
        assert result is None

    def test_positive_int_or_none_with_positive_int(self):
        """Test _positive_int_or_none with positive integer"""
        result = _positive_int_or_none(42)
        assert result == 42

    def test_positive_int_or_none_with_zero(self):
        """Test _positive_int_or_none with zero"""
        result = _positive_int_or_none(0)
        assert result is None  # Zero is not positive

    def test_positive_int_or_none_with_negative(self):
        """Test _positive_int_or_none with negative integer"""
        result = _positive_int_or_none(-5)
        assert result is None  # Negative is not positive

    def test_positive_int_or_none_with_string(self):
        """Test _positive_int_or_none with string input"""
        result = _positive_int_or_none("123")
        assert result == 123

        result = _positive_int_or_none("  456  ")  # With whitespace
        assert result == 456

        result = _positive_int_or_none("")  # Empty string
        assert result is None

        result = _positive_int_or_none("not_a_number")
        assert result is None


class TestAsPath:
    """Test _as_path utility function"""

    def setup_method(self):
        set_test_determinism()

    def test_as_path_with_absolute_path(self):
        """Test _as_path with absolute path"""
        base = Path("C:/base/dir")
        abs_path = "C:/absolute/path"

        result = _as_path(base, abs_path)

        assert result.is_absolute()
        assert Path(abs_path) == result

    def test_as_path_with_relative_path(self):
        """Test _as_path with relative path"""
        base = Path("C:/base/dir")
        rel_path = "relative/path"

        result = _as_path(base, rel_path)

        assert result == base / rel_path
        expected = base / rel_path
        assert expected == result

    def test_as_path_with_path_object(self):
        """Test _as_path with Path object"""
        base = Path("/base/dir")
        path_obj = Path("config/file.json")

        result = _as_path(base, path_obj)

        assert result == base / path_obj


class TestLoadConfigGeneric:
    """Test _load_config_generic utility function"""

    def setup_method(self):
        set_test_determinism()
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_load_config_generic_with_existing_file(self):
        """Test _load_config_generic with existing config file"""
        config_data = {"key": "value", "number": 42}
        config_file = self.temp_dir / "config.json"

        with open(config_file, "w") as f:
            json.dump(config_data, f)

        with patch.dict(os.environ, {"TEST_CONFIG": str(config_file)}):
            result = _load_config_generic(
                "TEST_CONFIG", Path("default.json"), json.load
            )

        assert result == config_data

    def test_load_config_generic_with_missing_file(self):
        """Test _load_config_generic with missing config file"""
        with patch.dict(os.environ, {}, clear=True):
            default_path = self.temp_dir / "missing.json"
            result = _load_config_generic("MISSING_CONFIG", default_path, json.load)

        assert result == {}

    def test_load_config_generic_with_invalid_json(self):
        """Test _load_config_generic with invalid JSON file"""
        config_file = self.temp_dir / "invalid.json"

        with open(config_file, "w") as f:
            f.write("invalid json content")

        with patch.dict(os.environ, {"TEST_CONFIG": str(config_file)}):
            result = _load_config_generic(
                "TEST_CONFIG", Path("default.json"), json.load
            )

        assert result == {}


class TestDataclassConfigurations:
    """Test configuration dataclass instantiation"""

    def setup_method(self):
        set_test_determinism()

    def test_risk_config_defaults(self):
        """Test RiskConfig default values"""
        config = RiskConfig()

        assert config.risk_pct == 0.02
        assert config.max_positions == 10
        assert config.max_pct == 0.10

    def test_risk_config_custom_values(self):
        """Test RiskConfig with custom values"""
        config = RiskConfig(risk_pct=0.05, max_positions=15, max_pct=0.15)

        assert config.risk_pct == 0.05
        assert config.max_positions == 15
        assert config.max_pct == 0.15

    def test_data_config_defaults(self):
        """Test DataConfig default values"""
        config = DataConfig()

        assert config.vendor == "EODHD"
        assert config.eodhd_base == "https://eodhistoricaldata.com"
        assert config.api_key_env == "EODHD_API_KEY"
        assert config.max_workers == 20
        assert config.batch_size == 100

    def test_ui_config_defaults(self):
        """Test UIConfig default values"""
        config = UIConfig()

        assert config.default_capital == 100000
        assert config.default_long_ratio == 0.5
        assert "system1" in config.long_allocations
        assert "system2" in config.short_allocations
        assert sum(config.long_allocations.values()) == 1.0  # Should sum to 1
        assert sum(config.short_allocations.values()) == 1.0


class TestGetSettings:
    """Test main get_settings function"""

    def setup_method(self):
        set_test_determinism()

    def test_get_settings_basic(self):
        """Test get_settings returns valid configuration"""
        settings = get_settings()

        assert hasattr(settings, "risk")
        assert hasattr(settings, "data")
        assert hasattr(settings, "ui")
        assert isinstance(settings.risk, RiskConfig)
        assert isinstance(settings.data, DataConfig)
        assert isinstance(settings.ui, UIConfig)

    def test_get_settings_caching(self):
        """Test get_settings uses caching (lru_cache)"""
        settings1 = get_settings()
        settings2 = get_settings()

        # Should be the same object due to caching
        assert settings1 is settings2

    @patch("config.settings._load_config_generic")
    def test_get_settings_with_config_override(self, mock_load_config):
        """Test get_settings with configuration overrides"""
        # Mock configuration override
        mock_load_config.return_value = {
            "risk": {"max_positions": 25},
            "ui": {"default_capital": 200000},
        }

        # Clear cache to force reload
        get_settings.cache_clear()

        settings = get_settings()

        # Verify the settings still work (exact values depend on implementation)
        assert isinstance(settings.risk, RiskConfig)
        assert isinstance(settings.ui, UIConfig)


if __name__ == "__main__":
    pytest.main([__file__])
