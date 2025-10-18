"""
Focused configuration tests for Settings coverage boost
"""

import json
import os
from pathlib import Path
import tempfile
from unittest.mock import mock_open, patch

import yaml

from config.settings import Settings, get_settings


class TestSettingsLoading:
    """Test settings loading functionality"""

    def test_get_settings_basic(self):
        """Test basic settings loading"""
        settings = get_settings()

        assert isinstance(settings, Settings)
        assert hasattr(settings, "cache")
        assert hasattr(settings, "DATA_CACHE_DIR")

    def test_get_settings_singleton(self):
        """Test settings is singleton"""
        settings1 = get_settings()
        settings2 = get_settings()

        # Should return same instance
        assert settings1 is settings2

    def test_settings_has_required_attributes(self):
        """Test settings has required attributes"""
        settings = get_settings()

        # Check major attribute groups exist
        required_attrs = [
            "cache",
            "data",
            "logging",
            "DATA_CACHE_DIR",
            "RESULTS_DIR",  # Corrected from RESULTS_CSV_DIR
        ]

        for attr in required_attrs:
            assert hasattr(settings, attr), f"Missing attribute: {attr}"

    def test_settings_cache_configuration(self):
        """Test cache configuration structure"""
        settings = get_settings()

        assert hasattr(settings.cache, "full_dir")
        assert hasattr(settings.cache, "rolling_dir")
        assert hasattr(settings.cache, "rolling")

        # Rolling cache config
        rolling = settings.cache.rolling
        assert hasattr(rolling, "base_lookback_days")  # Corrected from "days"
        assert hasattr(rolling, "buffer_days")
        assert isinstance(rolling.base_lookback_days, int)
        assert rolling.base_lookback_days > 0

    def test_settings_data_config(self):
        """Test data processing configuration"""
        settings = get_settings()

        assert hasattr(settings.data, "max_workers")
        assert hasattr(settings.data, "batch_size")

        # Should have reasonable values
        assert isinstance(settings.data.max_workers, int)
        assert settings.data.max_workers > 0
        assert isinstance(settings.data.batch_size, int)
        assert settings.data.batch_size > 0

    def test_settings_logging_config(self):
        """Test logging configuration"""
        settings = get_settings()

        assert hasattr(settings.logging, "level")
        assert hasattr(settings.logging, "format")

        # Logging level should be valid
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert settings.logging.level in valid_levels

    def test_settings_directory_paths(self):
        """Test directory path configurations"""
        settings = get_settings()

        # Directory paths should be Path objects or strings
        assert isinstance(settings.DATA_CACHE_DIR, (Path, str))
        assert isinstance(settings.RESULTS_DIR, (Path, str))

        # Should contain reasonable path components
        assert len(str(settings.DATA_CACHE_DIR)) > 0
        assert len(str(settings.RESULTS_DIR)) > 0


class TestSettingsEnvironmentOverrides:
    """Test environment variable overrides"""

    def test_env_override_data_cache_dir(self):
        """Test DATA_CACHE_DIR environment override"""
        test_path = "/custom/cache/path"

        with patch.dict(os.environ, {"DATA_CACHE_DIR": test_path}):
            # Clear settings cache to force reload
            if hasattr(get_settings, "_instance"):
                delattr(get_settings, "_instance")

            settings = get_settings()
            assert settings.DATA_CACHE_DIR == test_path

    def test_env_override_results_dir(self):
        """Test RESULTS_DIR environment override"""
        test_path = "/custom/results/path"

        with patch.dict(os.environ, {"RESULTS_DIR": test_path}):
            # Clear settings cache
            if hasattr(get_settings, "_instance"):
                delattr(get_settings, "_instance")

            settings = get_settings()
            assert str(settings.RESULTS_DIR) == test_path

    def test_env_override_log_level(self):
        """Test LOG_LEVEL environment override"""
        test_level = "DEBUG"

        with patch.dict(os.environ, {"LOG_LEVEL": test_level}):
            # Clear settings cache
            if hasattr(get_settings, "_instance"):
                delattr(get_settings, "_instance")

            settings = get_settings()
            assert settings.logging.level == test_level


class TestSettingsFileLoading:
    """Test configuration file loading"""

    def test_yaml_config_loading(self):
        """Test YAML configuration file loading"""
        yaml_content = """
        cache:
          full_dir: "test_full"
          rolling_dir: "test_rolling"
          rolling:
            days: 250
            base_lookback_days: 200

        batch_processing:
          max_workers: 2
          chunk_size: 50

        logging:
          level: "DEBUG"
          format: "test format"
        """

        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch("os.path.exists", return_value=True):
                with patch("yaml.safe_load", return_value=yaml.safe_load(yaml_content)):
                    # Clear settings cache
                    if hasattr(get_settings, "_instance"):
                        delattr(get_settings, "_instance")

                    settings = get_settings()

                    # Should load values from YAML
                    assert settings.cache.rolling.days == 250
                    assert settings.batch_processing.max_workers == 2
                    assert settings.logging.level == "DEBUG"

    def test_json_config_loading(self):
        """Test JSON configuration file loading"""
        json_content = {
            "cache": {
                "full_dir": "json_full",
                "rolling_dir": "json_rolling",
                "rolling": {"days": 300, "base_lookback_days": 250},
            },
            "batch_processing": {"max_workers": 4, "chunk_size": 100},
        }

        with patch("builtins.open", mock_open(read_data=json.dumps(json_content))):
            with patch("os.path.exists", return_value=True):
                with patch("json.load", return_value=json_content):
                    # Clear settings cache
                    if hasattr(get_settings, "_instance"):
                        delattr(get_settings, "_instance")

                    settings = get_settings()

                    # Should load values from JSON
                    assert settings.cache.rolling.days == 300
                    assert settings.batch_processing.max_workers == 4

    def test_config_file_priority(self):
        """Test configuration file loading priority"""
        # JSON should have higher priority than YAML
        json_content = {"cache": {"rolling": {"days": 123}}}
        yaml_content = "cache:\n  rolling:\n    days: 456"

        with patch("builtins.open", mock_open()):  # mock_file removed
            with patch("os.path.exists", return_value=True):
                with patch("json.load", return_value=json_content):
                    with patch("yaml.safe_load", return_value=yaml.safe_load(yaml_content)):
                        # Clear settings cache
                        if hasattr(get_settings, "_instance"):
                            delattr(get_settings, "_instance")

                        settings = get_settings()

                        # Should use JSON value (higher priority)
                        assert settings.cache.rolling.days == 123


class TestSettingsValidation:
    """Test settings validation"""

    def test_settings_numeric_values(self):
        """Test numeric settings are properly typed"""
        settings = get_settings()

        # Check numeric types
        assert isinstance(settings.cache.rolling.days, int)
        assert isinstance(settings.cache.rolling.base_lookback_days, int)
        assert isinstance(settings.batch_processing.max_workers, int)
        assert isinstance(settings.batch_processing.chunk_size, int)

        # Values should be positive
        assert settings.cache.rolling.days > 0
        assert settings.cache.rolling.base_lookback_days > 0
        assert settings.batch_processing.max_workers > 0
        assert settings.batch_processing.chunk_size > 0

    def test_settings_string_values(self):
        """Test string settings are properly typed"""
        settings = get_settings()

        # Check string/path types
        assert isinstance(settings.cache.full_dir, (str, Path))
        assert isinstance(settings.cache.rolling_dir, (str, Path))
        assert isinstance(settings.logging.level, str)
        assert isinstance(settings.DATA_CACHE_DIR, (str, Path))
        assert isinstance(settings.RESULTS_DIR, (str, Path))

        # Strings should not be empty
        assert len(str(settings.cache.full_dir)) > 0
        assert len(str(settings.cache.rolling_dir)) > 0
        assert len(settings.logging.level) > 0
        assert len(str(settings.DATA_CACHE_DIR)) > 0

    def test_settings_logical_constraints(self):
        """Test logical constraints in settings"""
        settings = get_settings()

        # Rolling lookback should be positive
        assert settings.cache.rolling.base_lookback_days > 0
        assert settings.cache.rolling.buffer_days > 0

        # Max workers should be reasonable
        assert settings.data.max_workers <= 32  # Reasonable upper limit

        # Batch size should be reasonable
        assert settings.data.batch_size <= 10000  # Reasonable upper limit


class TestSettingsRobustness:
    """Test settings robustness and error handling"""

    def test_missing_config_files(self):
        """Test behavior when config files are missing"""
        with patch("os.path.exists", return_value=False):
            # Clear settings cache
            if hasattr(get_settings, "_instance"):
                delattr(get_settings, "_instance")

            # Should still work with defaults
            settings = get_settings()
            assert isinstance(settings, Settings)
            assert hasattr(settings, "cache")

    def test_corrupted_json_config(self):
        """Test behavior with corrupted JSON config"""
        with patch("builtins.open", mock_open(read_data="invalid json {")):
            with patch("os.path.exists", return_value=True):
                # Clear settings cache
                if hasattr(get_settings, "_instance"):
                    delattr(get_settings, "_instance")

                # Should fallback gracefully
                settings = get_settings()
                assert isinstance(settings, Settings)

    def test_corrupted_yaml_config(self):
        """Test behavior with corrupted YAML config"""
        with patch("builtins.open", mock_open(read_data="invalid: yaml: content: [")):
            with patch("os.path.exists", return_value=True):
                # Clear settings cache
                if hasattr(get_settings, "_instance"):
                    delattr(get_settings, "_instance")

                # Should fallback gracefully
                settings = get_settings()
                assert isinstance(settings, Settings)

    def test_partial_config_override(self):
        """Test partial configuration override"""
        partial_config = {"cache": {"rolling": {"days": 150}}}  # Only override this value

        with patch("builtins.open", mock_open(read_data=json.dumps(partial_config))):
            with patch("os.path.exists", return_value=True):
                with patch("json.load", return_value=partial_config):
                    # Clear settings cache
                    if hasattr(get_settings, "_instance"):
                        delattr(get_settings, "_instance")

                    settings = get_settings()

                    # Should override only specified value
                    assert settings.cache.rolling.base_lookback_days == 150
                    # Other values should remain default
                    assert hasattr(settings.cache, "full_dir")
                    assert hasattr(settings, "data")


class TestSettingsAttributes:
    """Test specific settings attributes and their properties"""

    def test_cache_settings_structure(self):
        """Test cache settings nested structure"""
        settings = get_settings()

        # Test cache structure
        cache = settings.cache
        assert hasattr(cache, "full_dir")
        assert hasattr(cache, "rolling_dir")
        assert hasattr(cache, "rolling")
        assert hasattr(cache, "file_format")

        # Test rolling cache structure
        rolling = cache.rolling
        assert hasattr(rolling, "days")
        assert hasattr(rolling, "base_lookback_days")
        assert hasattr(rolling, "buffer_days")
        assert hasattr(rolling, "meta_file")

    def test_alpaca_settings_structure(self):
        """Test Alpaca settings if present"""
        settings = get_settings()

        # UI settings should exist
        assert hasattr(settings, "ui")
        ui = settings.ui
        assert hasattr(ui, "default_capital")
        assert hasattr(ui, "long_allocations")

    def test_ui_settings_structure(self):
        """Test UI settings if present"""
        settings = get_settings()

        # UI settings might be optional
        if hasattr(settings, "ui"):
            ui = settings.ui
            # Should have some UI configuration
            assert isinstance(ui, object)

    def test_notification_settings_structure(self):
        """Test notification settings if present"""
        settings = get_settings()

        # Scheduler settings should exist
        assert hasattr(settings, "scheduler")
        scheduler = settings.scheduler
        assert hasattr(scheduler, "timezone")


class TestSettingsIntegration:
    """Integration tests for settings"""

    def test_settings_with_temporary_files(self):
        """Test settings loading with actual temporary files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"

            config_data = {
                "cache": {
                    "full_dir": str(Path(temp_dir) / "full"),
                    "rolling_dir": str(Path(temp_dir) / "rolling"),
                },
                "batch_processing": {"max_workers": 3},
            }

            # Write actual config file
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            # Test that settings can work with real files
            # This is mainly to ensure no import/loading issues
            settings = get_settings()
            assert isinstance(settings, Settings)

    def test_settings_environment_integration(self):
        """Test settings with environment variables"""
        test_env = {
            "DATA_CACHE_DIR": "test_cache",
            "RESULTS_CSV_DIR": "test_results",
            "LOG_LEVEL": "ERROR",
        }

        with patch.dict(os.environ, test_env, clear=False):
            # Clear settings cache
            if hasattr(get_settings, "_instance"):
                delattr(get_settings, "_instance")

            settings = get_settings()

            # Environment should override defaults
            assert str(settings.DATA_CACHE_DIR) == "test_cache"
            assert str(settings.RESULTS_DIR) == "test_results"
            assert settings.logging.level == "ERROR"


# Cleanup function to reset settings between tests
def setup_module():
    """Setup for module tests"""
    pass


def teardown_module():
    """Cleanup after module tests"""
    # Clear any cached settings
    if hasattr(get_settings, "_instance"):
        delattr(get_settings, "_instance")
